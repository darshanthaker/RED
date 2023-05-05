import numpy as np
import re
import torch
import torch.nn as nn
import math
import argparse
import os
import random

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
        'yale': {1: 7.5, \
        2: 1.0,
        np.infty: 0.03}, \
        'synthetic': {1: 20., \
        2: 4., \
        np.infty: 0.6},
        'imagenet': {1: 12., \
        2: 0.5, \
        np.infty: 0.3}, \
        'fashionmnist': {1: 10., \
        2: 2., \
        np.infty: 0.3}
        }
STEP = {'mnist': {1: 0.8, \
        2: 0.1, \
        np.infty: 0.01},
        'cifar': {1: 1.0, \
        2: 0.05, \
        np.infty: 0.003}, \
        'yale': {1: 1.0, \
        2: 0.02, \
        np.infty: 0.003}, \
        'synthetic': {1: 0.8, \
        2: 0.1, \
        np.infty: 0.01},
        'imagenet': {1: 1.0, \
        2: 0.02, \
        np.infty: 0.003}, \
        'fashionmnist' : {1: 0.8, \
        2: 0.1, \
        np.infty: 0.01}
        }
SIZE_MAP = {'yale': 20, \
            'cifar': 200, \
            'mnist': 200, \
            'imagenet': 20, \
            'synthetic': 200, \
            'fashionmnist': 200}

def get_parser(parser):
    parser.add_argument('--dataset', default='mnist', type=str, help='Dataset to use for experiments')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate for training network')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train network')
    parser.add_argument('--bsz', default=128, type=int, help='Batch size')
    parser.add_argument('--arch', default='carlini_cnn', type=str, help='Network architecture')
    parser.add_argument('--pretrained_path', default="", type=str, help='Path to find pretrained model')
    parser.add_argument('--embedding', default=None, type=str, help='Embedding to use')
    parser.add_argument('--J', default=2, type=float, help='Scattering Transform J')
    parser.add_argument('--L', default=7, type=float, help='Scattering Transform L')
    parser.add_argument('--regularizer', default=4, type=int, help='Regularizer to use for IRLS')
    parser.add_argument('--toolchain', default=[1, 2, np.infty], nargs='+', type=float, help='Toolchain')
    parser.add_argument('--test_lp', default=2, type=float, help='Lp perturbation type to apply to test points')
    parser.add_argument('--lp_variant', default=None, type=str, help='Lp perturbation type variant to apply to test points')
    parser.add_argument('--lambda1', default=5, type=float, help='Lambda1 for IRLS')
    parser.add_argument('--lambda2', default=15, type=float, help='Lambda2 for IRLS')
    parser.add_argument('--del_threshold', default=0.2, type=float, help='Del threshold for IRLS')
    parser.add_argument('--solver', default='active', type=str, help='Solver to use')
    parser.add_argument('--use_cheat_grad', action='store_true', help='Whether or not to use test example in Jacobian computation')
    parser.add_argument('--realizable', action='store_true', help='Realizable or not')
    parser.add_argument('--make_realizable', action='store_true', help='Make Realizable or not')
    parser.add_argument('--encoder_num_epochs', default=100, type=int, help='# of training epochs for encoder')
    parser.add_argument('--decoder_num_epochs', default=10, type=int, help='# of training epochs for decoder')
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

def pgd_linf(model, X, y, epsilon=0.3, alpha=0.01, num_iter = 50, randomize = 0, restarts = 0, device = "cuda:0"):
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

def l1_dir_topk(grad, delta, X, gap, k = 10) :
    #Check which all directions can still be increased such that
    #they haven't been clipped already and have scope of increasing
    # ipdb.set_trace()
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]
    # print (batch_size)
    neg1 = (grad < 0)*(X_curr <= gap)
#     neg1 = (grad < 0)*(X_curr == 0)
    neg2 = (grad > 0)*(X_curr >= 1-gap)
#     neg2 = (grad > 0)*(X_curr == 1)
    neg3 = X_curr <= 0
    neg4 = X_curr >= 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(batch_size,1,-1)
    grad_check = grad.view(batch_size,1,-1)
    grad_check[u] = 0

    kval = kthlargest(grad_check.abs().float(), k, dim = 2)[0].unsqueeze(1)
    k_hot = (grad_check.abs() >= kval).float() * grad_check.sign()
    return k_hot.view(batch_size, channels, pix, pix)


def pgd_l1_topk(model, X,y, epsilon = 10, alpha = 0.8, num_iter = 50, k_map = 0, gap = 0.05, device = "cuda:0", restarts = 0, randomize = 0):
    #Gap : Dont attack pixels closer than the gap value to 0 or 1
    # ipdb.set_trace()
    gap = gap
    max_delta = torch.zeros_like(X)
    if randomize:
        delta = torch.from_numpy(np.random.laplace(size=X.shape)).float().to(device)
        delta.data = delta.data*epsilon/norms_l1(delta.detach())
        delta.requires_grad = True
    else:
        delta = torch.zeros_like(X, requires_grad = True)
    alpha_l_1_default = alpha

    for t in range (num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X+delta), y)
        loss.backward()
        if k_map == 0:
            k = random.randint(5,20)
            alpha   = (alpha_l_1_default/k)
        elif k_map == 1:
            k = random.randint(10,40)
            alpha   = (alpha_l_1_default/k)
        else:
            k = 10
            alpha = alpha_l_1_default
        delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, gap,k)
        if (norms_l1(delta) > epsilon).any():
            delta.data = proj_l1ball(delta.data, epsilon, device)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
        delta.grad.zero_() 

    max_delta = delta.detach()

    #Restarts    
    for k in range(restarts):
        delta = torch.rand_like(X,requires_grad = True)
        delta.data = (2*delta.data - 1.0)
        delta.data = delta.data*epsilon/norms_l1(delta.detach())
        for t in range (num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them
            loss = nn.CrossEntropyLoss()(model(X+delta), y)
            loss.backward()
            if k_map == 0:
                k = random.randint(5,20)
                alpha   = (alpha_l_1_default/k)
            elif k_map == 1:
                k = random.randint(10,40)
                alpha   = (alpha_l_1_default/k)
            else:
                k = 10
                alpha = alpha_l_1_default
            delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, gap,k)
            if (norms_l1(delta) > epsilon).any():
                delta.data = proj_l1ball(delta.data, epsilon, device)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
            delta.grad.zero_() 
        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect]   

    return max_delta

def proj_l1ball(x, epsilon=10, device = "cuda:0"):
#     print (epsilon)
    # print (device)
    assert epsilon > 0
    # compute the vector of absolute values
    u = x.abs()
    if (u.sum(dim = (1,2,3)) <= epsilon).all():
        # print (u.sum(dim = (1,2,3)))
         # check if x is already a solution
        return x

    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    y = proj_simplex(u, s=epsilon, device = device)
    # compute the solution to the original problem on v
    y *= x.sign()
    y *= epsilon/norms_l1(y)
    return y

def proj_simplex(v, s=1, device = "cuda:0"):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]

    # check if we are already on the simplex    
    '''
    #Not checking this as we are calling this from the previous function only
    if v.sum(dim = (1,2,3)) == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    '''
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = v.view(batch_size,1,-1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending = True)
    cssv = u.cumsum(dim = 2)
    # get the number of > 0 components of the optimal solution
    vec = u * torch.arange(1, n+1).float().to(device)
    comp = (vec > (cssv - s)).float()

    u = comp.cumsum(dim = 2)
    w = (comp-1).cumsum(dim = 2)
    u = u + w
    rho = torch.argmax(u, dim = 2)
    rho = rho.view(batch_size)
    c = torch.FloatTensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).to(device)
    c = c-s
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = torch.div(c,(rho.float() + 1))
    theta = theta.view(batch_size,1,1,1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w

def kthlargest(tensor, k, dim=-1):
    val, idx = tensor.topk(k, dim = dim)
    return val[:,:,-1], idx[:,:,-1]

def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def norms_l2(Z):
    return norms(Z)

def norms_l2_squeezed(Z):
    return norms(Z).squeeze(1).squeeze(1).squeeze(1)

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def norms_l1_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None].squeeze(1).squeeze(1).squeeze(1)

def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float()

def norms_l0_squeezed(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float().squeeze(1).squeeze(1).squeeze(1)

def norms_linf(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]

def pgd_l2(model, X, y, epsilon=2.0, alpha=0.1, num_iter = 100, restarts = 0, device = "cuda:0", randomize = 0):
    # ipdb.set_trace()
    max_delta = torch.zeros_like(X)
    if random:
        delta = torch.rand_like(X, requires_grad=True) 
        delta.data *= (2.0*delta.data - 1.0)
        delta.data = delta.data*epsilon/norms_l2(delta.detach()) 
    else:
        delta = torch.zeros_like(X, requires_grad=True) 
    for t in range(num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data +=  correct*alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data *=  epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.data =   torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]     
        delta.grad.zero_()  

    max_delta = delta.detach()

    #restarts

    for k in range (restarts):
        delta = torch.rand_like(X, requires_grad=True) 
        delta.data *= (2.0*delta.data - 1.0)*epsilon 
        delta.data /= norms(delta.detach()).clamp(min=epsilon)
        for t in range(num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data +=  correct*alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.grad.zero_()  

        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect] 

    return max_delta    

def blocksoft_thres(vec, lam):
    norm_vec = np.linalg.norm(vec)
    if norm_vec <= lam:
        return np.zeros(vec.shape)
    else:
        return (1 - lam/norm_vec)*vec


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

import time

def timing(func):
    """
        Timing decorator that will take any function and time it. Will print to 
        stdout how long the function took in seconds.
        Example usage:
        @timing
        def some_func():
            # Some long function.
        When running some_func(), it will print out that:
            'some_func took x seconds' 
        This will be useful to time functions easily by adding a decorator.
    """

    def wrapper(*arg, **kwargs):
        t1 = time.time()
        ret_val = func(*arg, **kwargs)
        t2 = time.time()
        print("{} took {} seconds".format(func.__name__, t2 - t1))
        return ret_val

    return wrapper
