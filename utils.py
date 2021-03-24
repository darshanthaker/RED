import numpy as np
import re
import torch
import torch.nn as nn
import math
import argparse
import os

from PIL import Image
from typing import Union, Tuple
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from kymatio.torch import Scattering2D
from pdb import set_trace

EPS = {'mnist': {1: 10., \
       2: 2., \
       np.infty: 0.3}, \
       'cifar': {1: 12., \
        2: 0.5, \
        np.infty: 0.03},
        'yale': {1: 4.5, \
        2: 0.75,
        np.infty: 0.02}}
STEP = {'mnist': {1: 0.8, \
        2: 0.1, \
        np.infty: 0.01}, \
        'cifar': {1: 1.0, \
        2: 0.02, \
        np.infty: 0.003}, \
        'yale': {1: 1.0, \
        2: 0.02, \
        np.infty: 0.003}}
SIZE_MAP = {'yale': 20, \
            'cifar': 200, \
            'mnist': 200}

def get_parser(parser):
    parser.add_argument('--dataset', default='mnist', type=str, help='Dataset to use for experiments')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate for training network')
    parser.add_argument('--num_epochs', default=5, type=int, help='Number of epochs to train network')
    parser.add_argument('--bsz', default=128, type=int, help='Batch size')
    parser.add_argument('--arch', default='carlini_cnn', type=str, help='Network architecture')
    parser.add_argument('--pretrained_path', default="", type=str, help='Path to find pretrained model')
    parser.add_argument('--embedding', default=None, type=str, help='Embedding to use')
    parser.add_argument('--J', default=2, type=float, help='Scattering Transform J')
    parser.add_argument('--L', default=7, type=float, help='Scattering Transform L')
    parser.add_argument('--regularizer', default=4, type=int, help='Regularizer to use for IRLS')
    parser.add_argument('--toolchain', default=[1, 2, np.infty], nargs='+', type=float, help='Toolchain')
    parser.add_argument('--test_lp', default=2, type=float, help='Lp perturbation type to apply to test points')
    parser.add_argument('--lambda1', default=5, type=float, help='Lambda1 for IRLS')
    parser.add_argument('--lambda2', default=15, type=float, help='Lambda2 for IRLS')
    parser.add_argument('--del_threshold', default=0.2, type=float, help='Del threshold for IRLS')
    return parser 


# Format: data is a list of 38 subjects, each who have m_i x 192 x 168 images.
def parse_yale_data(resize=True):
    PATH = 'files/CroppedYale'

    data = [0 for _ in range(38)]

    for folder in os.listdir(PATH):
        if folder.startswith('.'):
            continue
        pid = int(folder[5:7])
        # Because 14 doesn't exist -.-
        if pid > 14:
            pid -= 2
        else:
            pid -= 1
        person_path = os.path.join(PATH, folder)
        person_images = list()
        for im_name in os.listdir(person_path):
            if not im_name.endswith('pgm') or 'Ambient' in im_name:
                continue
            im_path = os.path.join(PATH, folder, im_name)
            if resize:
                im = np.array(Image.open(im_path).resize((35, 40), resample=Image.LANCZOS))
            else:
                im = np.array(Image.open(im_path))
            im = im[:, :, None]
            person_images.append(im)
        person_images = np.array(person_images)
        data[pid] = person_images

    train = [0 for i in range(len(data))]
    test = [0 for i in range(len(data))]
    for pid in range(len(data)):
        try:
            N = data[pid].shape[0]
        except:
            set_trace()
        indices = list(range(N))
        np.random.shuffle(indices)
        #m = 44 # ~70%
        m = 55
        train[pid] = data[pid][:m, :, :, :]
        test[pid] = data[pid][m:, :, :, :]
    return data, train, test

def pgd_linf(model, X, y, epsilon=0.3, alpha=0.01, num_iter = 50, randomize = 0, restarts = 0, device = "cuda:1"):
    """ Construct FGSM adversarial examples on the examples X"""
    # ipdb.set_trace()
   
    max_delta = torch.zeros_like(X)
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = (delta.data * 2.0 - 1.0) * epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)    
    for t in range(num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.grad.zero_()
    max_delta = delta.detach()
    
    for i in range (restarts):
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = (delta.data * 2.0 - 1.0) * epsilon

        for t in range(num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them            
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.grad.zero_()

        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks        
        max_delta[incorrect] = delta.detach()[incorrect]
    return max_delta

class YaleDataset(Dataset):

    def __init__(self, X, y, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            assert False
        else:
            x = self.X[idx, :]
            y = self.y[idx]

            if self.transform:
                x = self.transform(x)
            return x, y

class ScatteringTransform(object):

    def __init__(self, J, L=8, shape=(32, 32)):
        self.J = J
        self.L = L
        self.shape = shape
        self.scattering = Scattering2D(J=J, L=L, shape=shape)
        if torch.cuda.is_available():
            self.scattering = self.scattering.cuda()

    def __call__(self, sample):
        if len(sample.shape) == 3:
            (C, N1, N2) = sample.shape
        else:
            (B, C, N1, N2) = sample.shape
        if torch.cuda.is_available():
            sample = sample.cuda()
        embed_sample = self.scattering(sample)
        new_C = int(C*(1 + self.L*self.J + (self.L**2*self.J*(self.J - 1))/2))
        if len(sample.shape) == 3:
            embed_sample = embed_sample.reshape((new_C, \
                    N1//2**self.J, N2//2**self.J))
        else:
            embed_sample = embed_sample.reshape((-1, new_C, \
                    N1//2**self.J, N2//2**self.J))
        return embed_sample
