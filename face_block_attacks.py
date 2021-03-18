import numpy as np
import cvxpy as cp
import sys
import os
import pickle
import scipy.io
import torch
import torchvision
import seaborn as sns; #sns.set(rc={'text.usetex' : True}); sns.set_style("ticks")
import torch.nn as nn
import re
import matplotlib.pyplot as plt
import foolbox
import utils
import copy

from PIL import Image
from cvxpy.atoms.norm import norm
from cvxpy.atoms.affine.hstack import hstack
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from neural_net import NN, CNN
from pdb import set_trace
from scipy.linalg import orth
from scipy.sparse.linalg import svds
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from advertorch.attacks import SparseL1DescentAttack, L2PGDAttack, LinfPGDAttack
from foolbox.attacks import FGSM, L2PGD, L1PGD
from foolbox.criteria import Misclassification
from foolbox.utils import samples
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from kymatio.torch import Scattering2D
from functional import pgd
from irls import BlockSparseIRLSSolver

DATASET = 'mnist'
#DATASET = 'yale'
ARCH = 'carlini_cnn'
#ARCH = 'dense'
EMBEDDING = None
ALG = 4
LP = 1
TOOLCHAIN = [1, 2, np.infty]
EPS = {'mnist': {1: 10., \
       2: 2., \
       np.infty: 0.3}, \
       'cifar': {1: 12., \
        2: 0.5, \
        np.infty: 0.03},
        'yale': {1: 15., \
        2: 5.,
        np.infty: 0.1}}
#EPS = {'mnist': {1: 0.3, \
#       2: 0.3, \
#       np.infty: 0.3}, \
#       'cifar': {1: 12., \
#        2: 0.5, \
#        np.infty: 0.03}}
STEP = {'mnist': {1: 1.0, \
        2: 0.5, \
        np.infty: 0.01}, \
        'cifar': {1: 1.0, \
        2: 0.02, \
        np.infty: 0.003}, \
        'yale': {1: 1.0, \
        2: 0.02, \
        np.infty: 0.003}}
EPS = EPS[DATASET]
STEP = STEP[DATASET]
MEAN_MAP = {'yale': 0.2728, \
            'cifar': [0.4913997551666284, 0.48215855929893703, 0.4465309133731618], \
            'mnist':  0.1307}
STD_MAP = {'yale': 0.2453, \
           'cifar': [0.24703225141799082, 0.24348516474564, 0.26158783926049628], \
            'mnist': 0.3081}
SIZE_MAP = {'yale': 20, \
            'cifar': 200, \
            'mnist': 200}



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

            #x = utils.to_tensor_custom(x)

            if self.transform:
                x = self.transform(x)
            return x, y

class Trainer(object):

    def __init__(self, arch='carlini_cnn', dataset='yale', bsz=128, embedding=None):
        self.arch = arch
        self.use_cnn = 'cnn' in arch
        self.dataset = dataset
        self.embedding = embedding
        self.J = 2
        self.L = 8
        if self.dataset == 'yale':
            self.d = 1400
            self.num_classes = 38
            self.in_channels = 1
        elif self.dataset == 'cifar':
            self.input_shape = (32, 32)
            self.d = 32*32*3
            self.num_classes = 10
            if self.embedding is None:
                self.in_channels = 3
            else:
                self.in_channels = int(1 + self.L*self.J + (self.L**2*self.J*(self.J - 1))/2)
        elif self.dataset == 'mnist':
            self.input_shape= (28, 28)
            self.d = 28*28
            self.num_classes = 10
            if self.embedding is None:
                self.in_channels = 1
            else:
                self.in_channels = int(1 + self.L*self.J + (self.L**2*self.J*(self.J - 1))/2)
        if self.use_cnn:
            d_arch = '{}_{}'.format(self.arch, self.dataset)
            self.net = CNN(arch=d_arch, embedding=self.embedding, in_channels=self.in_channels,
                num_classes=self.num_classes)
        else:
            self.net = NN(self.d, 256, self.num_classes, linear=False) 
        #self.scattering = utils.ScatteringTransform(J=self.J, L=self.L, shape=self.input_shape)
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.bsz = bsz
        self.loss_fn = nn.CrossEntropyLoss()
        #self.loss_fn = nn.MultiMarginLoss()
        self.train_loader, self.test_loader = self.preprocess_data()

    # Normalize image to [0, 1] and then standardize.
    def preprocess_data(self):
        if self.embedding is None: 
            transform = transforms.Compose(
                    [transforms.ToTensor()])
                    # transforms.Normalize((MEAN_MAP[self.dataset]), (STD_MAP[self.dataset]))])
        elif self.embedding == 'scattering':
            transform = transforms.Compose(
                    [transforms.ToTensor(),
                     self.scattering])
                    # transforms.Normalize((MEAN_MAP[self.dataset]), (STD_MAP[self.dataset]))])

        if self.dataset == 'yale':
            data = parse_yale_data()
            raw_train_full, raw_test_full = split_train_test(data)
            self.train_y = []
            self.test_y = []
            for i in range(len(raw_train_full)):
                self.train_y += [i for _ in range(len(raw_train_full[i]))]
                self.test_y += [i for _ in range(len(raw_test_full[i]))]
            raw_train_X = np.vstack(raw_train_full)
            raw_test_X = np.vstack(raw_test_full)

            self.train_dataset = YaleDataset(raw_train_X, self.train_y, transform=transform)
            self.test_dataset = YaleDataset(raw_test_X, self.test_y, transform=transform)
        elif self.dataset == 'cifar':
            self.train_dataset = torchvision.datasets.CIFAR10('files/', train=True, \
                download=True, transform=transform)
            self.test_dataset = torchvision.datasets.CIFAR10('files/', train=False, \
                download=True, transform=transform)
        elif self.dataset == 'mnist':
            self.train_dataset = torchvision.datasets.MNIST('files/', train=True, \
                download=True, transform=transform) 
            self.test_dataset = torchvision.datasets.MNIST('files/', train=False, \
                download=True, transform=transform)
        self.N_train = len(self.train_dataset)
        self.N_test = len(self.test_dataset)

        train_X = [list() for i in range(self.num_classes)]
        test_X = [list() for i in range(self.num_classes)]
        train_y = [list() for i in range(self.num_classes)]
        test_y = [list() for i in range(self.num_classes)]
        for i in range(self.N_train):
            x, y = self.train_dataset[i]
            if len(train_X[y]) >= SIZE_MAP[self.dataset]:
                continue
            train_X[y].append(x.numpy())
            train_y[y].append(y)
        for i in range(self.N_test):
            x, y = self.test_dataset[i]
            test_X[y].append(x.numpy())
            test_y[y].append(y)
        train_X = [np.array(val) for val in train_X]
        test_X = [np.array(val) for val in test_X]
        self.train_y = np.concatenate([np.array(val) for val in train_y])
        self.test_y = np.concatenate([np.array(val) for val in test_y])

        self.train_full = train_X
        self.test_full = test_X
        self.train_X = np.vstack(train_X)
        self.test_X = np.vstack(test_X)
        self.N_train = self.train_X.shape[0]
        self.N_test = self.test_X.shape[0]

        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.bsz,
            shuffle=True, worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=100,
            shuffle=True, worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))
        return train_loader, test_loader

    def train(self, num_epochs, lr):
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.5)
        for epoch in range(num_epochs): 
            for batch_idx, data in enumerate(self.train_loader):
                bx = data[0]
                by = data[1]
                if not self.use_cnn:
                    bx = bx.flatten(1)

                if torch.cuda.is_available():
                    bx = bx.cuda()
                    by = by.cuda()

                self.net.train()
                output = self.net.forward(bx.float())
                loss = self.loss_fn(output, by)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if epoch % 1 == 0:
                train_acc = self.evaluate(test=False)
                test_acc = self.evaluate(test=True)
                print("[{}] Loss: {}. Train Accuracy: {}%. Test Accuracy: {}%.".format(epoch, 
                    loss, train_acc, test_acc))

        save_path = 'pretrained_model_ce_{}_{}.pth'.format(self.arch, self.dataset)
        torch.save(self.net.state_dict(), save_path)
        print("Saved model to {}".format(save_path))

    def evaluate(self, test=True, given_examples=None):
        if test:
            data_loader = self.test_loader
        else:
            data_loader = self.train_loader
        self.net.eval()
        with torch.no_grad():
            loss_fn = nn.CrossEntropyLoss(reduction='sum')
            num_correct = 0
            if given_examples is not None:
                data_loader = [given_examples]
            for (bx, by) in data_loader:
                if given_examples is not None:
                    bx = torch.from_numpy(bx)
                    by = torch.from_numpy(by)
                if not self.use_cnn:
                    bx = bx.squeeze()
                    bx = bx.flatten(1)
                if torch.cuda.is_available():
                    bx = bx.cuda()
                    by = by.cuda()
                output = self.net.forward(bx.float())
                pred = output.data.argmax(1)
                #print("pred: {}".format(pred))
                #print("by : {}".format(by))
                #if given_examples is not None:
                #    set_trace()
                num_correct += pred.eq(by.data.view_as(pred)).sum().item()
        if given_examples is not None:
            acc = num_correct / len(by) * 100.
        else:
            acc = num_correct / len(data_loader.dataset) * 100.
            
        return acc

    def compute_train_dictionary(self, normalize_cols=True):
        train = self.train_full
        dictionary = list()
        for pid in range(len(train)):
            for i in range(train[pid].shape[0]):
                x = train[pid][i, :, :]
                if self.embedding is None:
                    dictionary.append(x.reshape(-1))
                elif self.embedding == 'scattering':
                    x = torch.from_numpy(x).unsqueeze(0)
                    if torch.cuda.is_available():
                        x = x.cuda()
                    embed_x = self.scattering(x).cpu().detach().numpy()
                    set_trace()
                    dictionary.append(embed_x.reshape(-1))
                       
        dictionary = np.array(dictionary).T
        if normalize_cols:
            return normalize(dictionary, axis=0)
        else:
            return dictionary

    def compute_lp_dictionary(self, eps, lp, block=False, normalize_cols=True):
        idx = list(range(self.N_train))
        bsz = self.N_train
        dictionary = list()
        blocks = dict()
        bx = torch.from_numpy(self.train_X)
        by = torch.from_numpy(self.train_y)
        if not self.use_cnn: 
            bx = bx.flatten(1)
        if torch.cuda.is_available():
            bx = bx.cuda()
            by = by.cuda()

        step = 2000
        dictionary = list()
        for i in range(0, bx.shape[0], step):
            batch_x = bx[i:i+step]
            batch_y = by[i:i+step]
            out = self._lp_attack(lp, eps, batch_x, batch_y, only_delta=True)
            out = out.reshape((out.shape[0], -1))
            dictionary.append(out)
        dictionary = np.concatenate(dictionary)

        by = by.cpu()
        for label in range(self.num_classes):
            blocks[label] = dictionary[torch.nonzero(by == label).squeeze()]
        if block:
            return blocks
        if normalize_cols:
            return normalize(dictionary.T, axis=0)
        else:
            return dictionary.T

    def _lp_attack(self, lp, eps, bx, by, debug=False, only_delta=False, scale=False):
        if eps == 0:
            if only_delta:
                return (bx - bx).detach().numpy()
            else:
                return bx.detach().numpy()
        d = bx.flatten(1).shape[1]

        if torch.cuda.is_available():
            bx = bx.cuda()
            by = by.cuda()
        if scale:
            bx.requires_grad = True
            loss = self.loss_fn(self.net(bx.float()), by)
            self.net.zero_grad()
            loss.backward()
            data_grad = bx.grad.data

            #eps = 0.3
            print("Linf eps: {}".format(eps))
            print("L2 eps: {}".format(eps * d **0.5))
            print("L1 eps: {}".format(eps * d))
            linf = bx + eps * data_grad.sign()
            grad_norm = data_grad.flatten(1).norm(dim=1)
            stacked_norm = torch.stack(d*[grad_norm + 1e-8]).T.reshape(-1, 1, 28, 28)
            l2 = bx + eps * d**0.5 * data_grad / stacked_norm
            adversary = SparseL1DescentAttack(
                self.net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=d*eps,
                nb_iter=150, eps_iter=eps, rand_init=True, clip_min=-np.inf, clip_max=np.inf,
            targeted=False)

            l1 = adversary.perturb(bx, by)
            if only_delta:
                linf = linf - bx
                l2 = l2 - bx
                l1 - l1 - bx 
            linf = linf.cpu().detach().numpy()
            l2 = l2.cpu().detach().numpy()
            l1 = l1.cpu().detach().numpy()
        else:
            #linf = bx + eps * data_grad.sign()
            #l2 = bx + eps * data_grad / torch.stack(d*[data_grad.norm(dim=1) + 1e-8]).T
          
            #self.net.zero_grad()
            #fmodel = foolbox.models.PyTorchModel(self.net, bounds=(0, 1)) 
            #if torch.cuda.is_available():
            #    fmodel = fmodel.gpu()
            #criterion = Misclassification(by) 
            step_size = STEP[lp]
    

            #out = projected_gradient_descent(self.net, bx, eps, step_size, 200, lp, 
            #            #clip_min=0, clip_max=1)
            #out = pgd(self.net, bx, by, torch.nn.CrossEntropyLoss(), k=200, step=0.1, eps=eps, norm=lp)
            #if only_delta:
            #    out = out - bx
            
            #if self.embedding == 'scattering':
            #    out = self.scattering(out)
            #out = out.cpu().detach().numpy()
            #return out

            #bx.requires_grad = True
            self.net.eval()
            self.net.zero_grad()
            #loss = self.loss_fn(self.net(bx.float()), by)
            #loss.backward()
            #data_grad = copy.deepcopy(bx.grad.data.detach())
            #bx.grad.zero_()

            if lp == np.infty: 
                #linf_adv = FGSM()
                #_, linf, _ = linf_adv(fmodel, bx, criterion, epsilons=[eps])
                #linf = linf[0]
                #linf = projected_gradient_descent(self.net, bx, eps, step_size, 200, np.infty, 
                #        clip_min=0, clip_max=1)
                #linf = bx + eps * data_grad.sign()
                #linf = torch.clip(linf, min=0, max=1)
                #set_trace()
                linf_adversary = LinfPGDAttack(
                    self.net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                    nb_iter=100, eps_iter=0.01, rand_init=True, clip_min=0, clip_max=1,
                    targeted=False)
                linf = linf_adversary.perturb(bx, by)
                if only_delta:
                    linf = linf - bx
                linf = linf.cpu().detach().numpy()
                return linf

            elif lp == 2:
                #l2_adv = L2PGD(abs_stepsize=0.1, steps=200)
                l2_adversary = L2PGDAttack(
                    self.net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                    nb_iter=200, eps_iter=0.1, rand_init=True, clip_min=0, clip_max=1,
                    targeted=False)
                #l2 = fast_gradient_method(self.net, bx, eps, 2, clip_min=0, clip_max=1)
                #l2 = projected_gradient_descent(self.net, bx, eps, step_size, 200, 2, 
                #            clip_min=0, clip_max=1)
                #images, labels = samples(fmodel, dataset="mnist", batchsize=20) 
                #set_trace()
                #_, l2, _  = l2_adv(fmodel, bx, criterion, epsilons=[eps])
                #l2 = l2[0]
                l2 = l2_adversary.perturb(bx, by)

                #set_trace()

                #grad_norm = data_grad.flatten(1).norm(dim=1)
                #stacked_norm = torch.stack(d*[grad_norm + 1e-8]).T.reshape(-1, 1, 28, 28)
                #l2 = bx + eps * data_grad / stacked_norm
                if only_delta:
                    l2 = l2 - bx
                l2 = l2.cpu().detach().numpy()
                #acc = self.evaluate(given_examples=(l2, by.cpu().detach().numpy()))
                #print("[L2 Func] Adversarial accuracy: {}".format(acc))
                return l2
           
            elif lp == 1: 
                #l1_adv = L1PGD(rel_stepsize=1.0, steps=200)
                l1_adversary = SparseL1DescentAttack(
                    self.net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                    nb_iter=200, eps_iter=0.8, rand_init=True, clip_min=0, clip_max=1,
                    targeted=False)
                #l1 = projected_gradient_descent(self.net, bx, eps, step_size, 200, 1, 
                #            clip_min=0, clip_max=1)
                #set_trace()
                l1 = l1_adversary.perturb(bx, by)
                #_, l1, _ = l1_adv(fmodel, bx, criterion, epsilons=[eps])
                #l1 = l1[0]
                if only_delta:
                    l1 = l1 - bx
                l1 = l1.cpu().detach().numpy()
                return l1

        if debug:
            return l2, linf
        if lp == 2:
            return l2
        elif lp == np.infty:
            return linf
        elif lp == 1:
            return l1

    def test_lp_attack(self, lp, batch_idx, eps, realizable=False):
        bx = self.test_X[batch_idx, :, :]
        by = self.test_y[batch_idx]
        bx = torch.from_numpy(bx)
        by = torch.from_numpy(by)

        if realizable:
            blocks = self.compute_lp_dictionary(eps=EPS[lp], lp=lp, block=True)

            D = [np.zeros(blocks[i].shape[0]) for i in range(self.num_classes)]
            for i in range(10):
                D[i][0] = 1./3.
                D[i][1] = 1./3.
                D[i][2] = 1./3.
            perturb = np.array([np.dot(blocks[i].T, D[i]) for i in range(self.num_classes)])

            #neigh = NearestNeighbors(n_neighbors=k)
            #neigh.fit(self.train_X.reshape((self.train_X.shape[0], -1)))
            #dist, ind = neigh.kneighbors(bx.reshape((1, -1)))
            #set_trace()

            perturbation = perturb[by, :]
            bx = bx.reshape((bx.shape[0], -1)).numpy()
            adv = bx + perturbation
            return adv
        else:
            bsz, channels, r, c = bx.shape[0], bx.shape[1], bx.shape[2], bx.shape[3]
            if not self.use_cnn:
                bx = bx.flatten(1)
            d = bx.shape[1]

            out = self._lp_attack(lp, eps, bx, by)
        
            if not self.use_cnn:
                out = out.reshape((bsz, channels, r, c))
            return out
            

# Format: data is a list of 38 subjects, each who have m_i x 192 x 168 images.
def parse_yale_data(resize=True):
    PATH = 'CroppedYale'

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
    return data

def split_train_test(data):
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
    return train, test

def serialize_dictionaries(trainer, toolchain):
    attack_dicts = list()
    for attack in toolchain:
        attack_dicts.append(trainer.compute_lp_dictionary(EPS[attack], attack))

    np.random.seed(0)
    test_idx = np.random.choice(list(range(trainer.N_test)), 324)
    Da = np.hstack(attack_dicts)
    Ds = trainer.compute_train_dictionary(trainer.train_full)
    dst = trainer.dataset
    sz = SIZE_MAP[dst]
    if toolchain == [1, 2, np.infty]:
        ident = "all"
    elif toolchain == [2, np.infty]:
        ident = "2_inf"
    elif toolchain == [1, np.infty]:
        ident = "1_inf"
    
    scipy.io.savemat('mats/{}_ce/Ds_sub{}.mat'.format(dst, sz), {'data': Ds})
    #scipy.io.savemat('mats/{}_ce/Da_sub{}_{}.mat'.format(dst, sz, len(toolchain)), {'data': Da})
    scipy.io.savemat('mats/{}_ce/Da_sub{}_{}.mat'.format(dst, sz, ident), {'data': Da})
    scipy.io.savemat('mats/{}_ce/test_y.mat'.format(dst), {'data': trainer.test_y[test_idx]})

    all_attacks = list()
    for lp in toolchain:
        test_adv = trainer.test_lp_attack(lp, test_idx, EPS[lp], realizable=False)
        all_attacks.append(test_adv)

        acc = trainer.evaluate(given_examples=(test_adv, trainer.test_y[test_idx]))
        print("[L{}] Adversarial accuracy: {}".format(lp, acc))
        if lp == np.infty:
            scipy.io.savemat('mats/{}_ce/linf_eps{}.mat'.format(dst, EPS[lp]), {'data': test_adv})
        elif lp == 2:
            scipy.io.savemat('mats/{}_ce/l2_eps{}.mat'.format(dst, EPS[lp]), {'data': test_adv})
        elif lp == 1:
            scipy.io.savemat('mats/{}_ce/l1_eps{}.mat'.format(dst, EPS[lp]), {'data': test_adv})

    #all_attacks = np.array(all_attacks)
    #avg = np.mean(all_attacks, axis=0)
    #scipy.io.savemat('mats/{}_ce/avg_eps{}.mat'.format(dst, EPS[np.infty]), {'data': avg})

def analyze(trainer, lp):
    epss = pickle.load(open('outs/epss_{}.pkl'.format(lp), 'rb'))
    opts = pickle.load(open('outs/opts_{}.pkl'.format(lp), 'rb'))
    lp_map = {2: 0, np.infty: 1}

    epss = np.array(epss)
    opts = np.array(opts)
    
    N = epss.shape[0]
    att_corr = 0
    pid_corr = 0
    for i in range(N):
        eps = epss[i, :, :]
        pid, att = np.unravel_index(eps.argmin(), eps.shape)             
        np.set_printoptions(precision=20)
        print("[{}]: {} - {} vs. {} - {}".format(i, pid, att, trainer.test_y[i], lp_map[lp]))
        if att == lp_map[lp]:
            att_corr += 1
        if pid == trainer.test_y[i]:
            pid_corr += 1
    att_per = att_corr / float(N) * 100.
    pid_per = pid_corr / float(N) * 100.
    print("Attack Percentage correct: {}%".format(att_per))
    print("PID Percentage correct: {}%".format(pid_per))

def plot_heatmap():
    eps = scipy.io.loadmat('mats/linear_mm/err_joint_eps0.1.mat')['err_joint']
    err_class = scipy.io.loadmat('mats/linear_mm/err_class_eps0.1.mat')['err_class']
    err_attack = scipy.io.loadmat('mats/linear_mm/err_attack_eps0.1.mat')['err_attack']
    ax = sns.heatmap(eps, annot=True, fmt="0.02f", annot_kws={"fontsize":8},
                yticklabels=range(1, eps.shape[0]+1), xticklabels=['L2', 'Linf'])
    ax.set_yticklabels(range(1, eps.shape[0]+1), size = 8)
    plt.ylabel('Person ID')
    plt.title("$\| x_{adv} - D_s[i]c_s[i] - D_a[j][i]c_a[j][i]\|_2$")
    plt.show()
    ax = sns.heatmap(err_class, annot=True, fmt="0.02f", annot_kws={"fontsize":8},
                yticklabels=range(1, err_class.shape[0]+1))
    ax.set_yticklabels(range(1, err_class.shape[0]+1), size = 8)
    plt.ylabel('Person ID')
    plt.title("$\| x_{adv} - D_s[i]c_s[i] - D_ac_a\|_2$")
    plt.show()
    ax = sns.heatmap(err_attack, annot=True, fmt="0.02f", annot_kws={"fontsize":8},
                yticklabels=range(1, err_attack.shape[0]+1))
    ax.set_yticklabels(range(1, err_attack.shape[0]+1), size = 8)
    plt.ylabel('Attack ID')
    plt.title("$\| x_{adv} - D_s[i*]c_s[i*] - D_a[i*][j]c_a[i*][j]\|_2$")
    plt.show()
    #plt.savefig('plots/fg_heatmap_success_linf.png')
    #plt.title("$\| x_{adv} - D_s[i]c_s[i] - D_a[j]c_a[j]\|_2$")
    #plt.savefig('plots/coarse_heatmap_success_linf.png')
   
def backtest(trainer, mat):
    pred_cleans = scipy.io.loadmat(mat)['pred_cleans']
    pred_cleans = pred_cleans.reshape((-1, 28, 28))
    test_y = scipy.io.loadmat('mats/mnist_ce/test_y.mat')['data'][0, :100]

    test_acc = trainer.evaluate(given_examples=(pred_cleans, test_y))
    print("Backtest accuracy: {}%".format(test_acc))

def baseline(trainer, test_adv, test_y):
    Ds = trainer.compute_train_dictionary(trainer.train_full)
    s_len = Ds.shape[1]

    thres = 5
    pred = list()
    for i in range(test_adv.shape[0]):
        if i % 10 == 0:
            print(i)

        c = cp.Variable(s_len)
        idx = 0
        objective = 0
        for pid in range(trainer.num_classes):
            l = SIZE_MAP[trainer.dataset]
            objective += norm(c[idx:idx+l], p=2)
            idx += l

        x_adv = test_adv[i, :].reshape(-1)
        minimizer = cp.Minimize(objective)
        constraints = [norm(x_adv - Ds@c, p=2) <= thres]
        prob = cp.Problem(minimizer, constraints)

        #result = prob.solve(verbose=True, solver=cp.MOSEK)
        result = prob.solve(verbose=False)
        opt = np.array(c.value)

        l = SIZE_MAP[trainer.dataset]
        res = list()
        for pid in range(trainer.num_classes):
            res.append(np.linalg.norm(x_adv - Ds[:, l*pid:l*(pid + 1)]@opt[l*pid:l*(pid + 1)]))
        pred.append(np.argmin(res))
    pred = np.array(pred)
    acc = sum(pred == test_y)
    return acc

def eps_plot(trainer):
    epss = [0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
    atts = ['l2', 'l1', 'linf']
    att_map = {'l2': 2, 'linf': np.inf}
    result_map = {'l2': {'signal': [], 'att': [], 'test': [], 'backtest': [], 'baseline': []}, 
                  'linf': {'signal': [], 'att': [], 'test': [], 'backtest': [], 'baseline': []}}

    result_map = {'linf': {'eps': [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], 
                           'signal': [94, 89, 85, 82, 73, 63, 56],
                           'att': [24, 50, 88, 93, 92, 91, 86],
                           'test': [98.99, 93.21, 62.96, 22.53, 5.55, 2.5, 2.5],
                           'backtest': [99, 96, 65, 31, 6, 2.5, 3],
                           'baseline': [92, 84, 82, 73, 72, 66, 54]},
                  'l2': {'eps': [0, 0.3, 0.6, 1, 1.3, 1.6, 2], 
                           'signal': [94, 93, 89, 86, 85, 82, 80],
                           'att': [34, 35, 39, 43, 45, 42, 41],
                           'test': [98.99, 97.53, 92.90, 74.38, 55.86, 50, 44.75],
                           'backtest': [99, 99, 95, 73, 59, 52, 47],
                           'baseline': [92, 89, 85, 84, 81, 79, 76]},
                  'l1': {'eps': [0, 1.5, 3, 5, 6.5, 8, 10], 
                           'signal': [94, 93, 90, 85, 85, 85, 82],
                           'att': [44, 44, 47, 52, 55, 59, 66],
                           'test': [98.99, 98.45, 95.68, 90.12, 79.63, 63.88, 41.97],
                           'backtest': [99, 99, 96, 93, 80, 62, 41],
                           'baseline': [92, 87, 84, 83, 81, 78, 75]}}

    baseline_map = {'linf': {'name': ['M1', 'M2', 'Minf', 'MAX', 'AVG', 'MSD'],
                             'eps_t': [0.287, 0.3, 0.3, 0.3, 0.3, 0.282], 
                             'eps': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3], 
                             'acc': [0, 0.4, 90.3, 51, 65.2, 62.7]},
                    'l2': {'name': ['M1', 'M2', 'Minf', 'MAX', 'AVG', 'MSD'],
                             'eps_t': [2, 1.9, 2, 2, 1.9, 2],
                             'eps': [2, 2, 2, 2, 2, 2],
                             'acc': [38.7, 70.2, 66.8, 64.1, 66.9, 70.2]},
                    'l1': {'name': ['M1', 'M2', 'Minf', 'MAX', 'AVG', 'MSD'],
                             'eps_t': [10, 10, 9.5, 10, 10, 10],
                             'eps': [10, 10, 10, 10, 10, 10],
                             'acc': [74.6, 51.1, 61.8, 61.2, 66.5, 70.4]}}
                
    #np.random.seed(0)
    #test_idx = np.random.choice(list(range(trainer.N_test)), 324)[:100]
    #test_y = scipy.io.loadmat('mats/mnist_ce/test_y.mat')['data'][0, :100]
    """
    for eps in epss:
        for (idx, att) in enumerate(atts):
            att_pred = scipy.io.loadmat('mats/mnist_recon/att_pred_{:0.2f}_{}.mat'.format(eps, att))['att_pred']
            adv_acc = np.sum(att_pred == idx + 1)
            class_pred = scipy.io.loadmat('mats/mnist_recon/class_pred_{:0.2f}_{}.mat'.format(eps, att))['class_pred']
            signal_acc = np.sum(class_pred == test_y)
            pred_cleans = scipy.io.loadmat('mats/mnist_recon/pred_cleans_{:0.2f}_{}.mat'.format(eps, att))['pred_cleans']
            pred_cleans = pred_cleans.reshape((-1, 28, 28))
            backtest_acc = trainer.evaluate(given_examples=(pred_cleans, test_y))

            test_adv = trainer.test_lp_attack(att_map[att], test_idx, eps, realizable=False)

            clean_acc = trainer.evaluate(given_examples=(test_adv, test_y))

            baseline_acc = baseline(trainer, test_adv, test_y)

            result_map[att]['att'].append(adv_acc)
            result_map[att]['signal'].append(signal_acc)
            result_map[att]['backtest'].append(backtest_acc)
            result_map[att]['test'].append(clean_acc)
            result_map[att]['baseline'].append(baseline_acc)
    """

    #result_map = pickle.load(open('mats/mnist_recon/result_map.pkl', 'rb'))

    fig, axes = plt.subplots(1, 3, figsize=(6.4*3, 4.8))
    dual_axes = list()
    for (ax1, att) in zip(axes, ['l1', 'l2', 'linf']):
        #fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        dual_axes.append(ax2)
        ax1.plot(result_map[att]['eps'], result_map[att]['signal'], label='SBSC', marker='*', markersize=7)
        ax2.plot(result_map[att]['eps'], result_map[att]['att'], label='AD', linestyle='--', marker='s', color='black', markerfacecolor='none')
        ax1.plot(result_map[att]['eps'], result_map[att]['test'], label='No Defense', marker='*', markersize=7)
        ax1.plot(result_map[att]['eps'], result_map[att]['backtest'], label='SBSC+CNN', marker='*', markersize=7)
        ax1.plot(result_map[att]['eps'], result_map[att]['baseline'], label='BSC', marker='*', markersize=7)
        bm = baseline_map[att]
        #plt.scatter(bm['eps'], bm['acc'], marker='o', color='black', s=7.5)
        for (name, eps, acc) in zip(bm['name'], bm['eps'], bm['acc']):
            ax1.scatter(eps, acc, label=name, marker='x')
            #plt.annotate(name, (eps, acc+1), fontsize=8)
        #ax1.legend(prop={'size': 9})
        #ax2.legend(prop={'size': 9}, handlelength=3)
        ax1.grid()
        ax1.set_yticks(np.arange(0, 101, 10))
        ax2.set_yticks(np.arange(0, 101, 10))
        ax1.set_xlabel('{} Epsilon'.format(att))
    #fig.set_xlabel('Epsilon of Perturbation')
    #fig.set_ylabel('Signal Classification Accuracy')
    #fig.set_ylabel('Attack Detection Accuracy')
    axes[0].set_ylabel('Signal Classification Accuracy')
    dual_axes[-1].set_ylabel('Attack Detection Accuracy')
    plt.savefig('all_mnist.png')
        #ax1.cla()
        #ax2.cla()
        #fig.clf() 
        #plt.clf()

def irls(trainer, toolchain, lp, eps):
    print("L{} attacks, eps = {}".format(lp, eps))
    attack_dicts = list()
    for attack in toolchain:
        attack_dicts.append(trainer.compute_lp_dictionary(EPS[attack], attack))

    np.random.seed(0)
    test_idx = np.random.choice(list(range(trainer.N_test)), 324)
    test_y = trainer.test_y[test_idx]
    Da = np.hstack(attack_dicts)
    Ds = trainer.compute_train_dictionary(trainer.train_full)
    dst = trainer.dataset
    sz = SIZE_MAP[dst]
    num_attacks = len(toolchain)
    
    test_adv = trainer.test_lp_attack(lp, test_idx, eps, realizable=False)
    #bx = trainer.test_X[test_idx, :, :]
    #set_trace()
    acc = trainer.evaluate(given_examples=(test_adv, test_y))
    print("[L{}] Adversarial accuracy: {}%".format(lp, acc))
    #bacc = baseline(trainer, test_adv[:100, :], test_y[:100])
    #print("Baseline accuracy: {}%".format(bacc))
    #return

    solver = BlockSparseIRLSSolver(Ds, Da, trainer.num_classes, num_attacks, sz, 
            lambda1=5, lambda2=15, del_threshold=0.2)
    class_preds = list()
    attack_preds = list()
    denoised = list()
    for t in range(100):
        if t % 5 == 0:
            print(t)
        x = test_adv[t, :]
        x = x.reshape(-1)
        cs_est, ca_est, Ds_est, Da_est, err_cs, ws, wa, active_classes = solver.solve(x, alg=ALG)

        err_class = list()
        for i in range(solver.sig_bi.num_blocks[0]):
            Ds_blk = solver.sig_bi.get_block(Ds_est, i)
            cs_blk = solver.sig_bi.get_block(cs_est, i)
            err_class.append(np.linalg.norm(x - Ds_blk@cs_blk - Da_est@ca_est))
        err_class = np.array(err_class)
        i_star = np.argmin(err_class)
        class_preds.append(active_classes[i_star])

        err_attack = list()
        for j in range(num_attacks):
            Da_blk = solver.hier_bi.get_block(Da_est, (i_star, j)) 
            ca_blk = solver.hier_bi.get_block(ca_est, (i_star, j)) 
            err_attack.append(np.linalg.norm(x - Ds_est@cs_est - Da_blk@ca_blk))
        err_attack = np.array(err_attack)
        j_star = np.argmin(err_attack)
        attack_preds.append(j_star)
        Da_blk = solver.hier_bi.get_block(Da_est, (i_star, j_star)) 
        ca_blk = solver.hier_bi.get_block(ca_est, (i_star, j_star)) 
        denoised.append(x - Da_blk@ca_blk)
    class_preds = np.array(class_preds)
    attack_preds = np.array(attack_preds)
    denoised = np.array(denoised)
    signal_acc = np.sum(class_preds == test_y[:100])
    attack_acc = np.sum(attack_preds == toolchain.index(lp))
    if DATASET == 'mnist':
        denoised = denoised.reshape((100, 1, 28, 28))
    denoised_acc = trainer.evaluate(given_examples=(denoised, test_y[:100]))
    print("Signal classification accuracy: {}%".format(signal_acc))
    print("Attack detection accuracy: {}%".format(attack_acc))
    print("Denoised accuracy: {}%".format(denoised_acc))

def main():
    np.random.seed(0)
    trainer = Trainer(arch=ARCH, dataset=DATASET, bsz=128, embedding=EMBEDDING)
    #trainer.train(75, 0.05) 
    trainer.net.load_state_dict(torch.load('pretrained_model_ce_{}_{}.pth'.format(ARCH, DATASET)))
    test_acc = trainer.evaluate(test=True)
    print("Loaded pretrained model!. Test accuracy: {}%".format(test_acc))
    irls(trainer, TOOLCHAIN, LP, EPS[LP])
    #serialize_dictionaries(trainer, TOOLCHAIN)

def eps_grid():
    lp = 1
    for eps in [0.0, 1.5, 3.0, 5.0, 6.5, 8.0, 10.0]:
    #for eps in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    #for eps in [0.0, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0]:
        print("-------------EPS = {}---------------".format(eps))
        np.random.seed(0)
        trainer = Trainer(arch=ARCH, dataset=DATASET, bsz=128, embedding=EMBEDDING)
        #trainer.train(75, 0.05) 
        trainer.net.load_state_dict(torch.load('pretrained_model_ce_{}_{}.pth'.format(ARCH, DATASET)))
        test_acc = trainer.evaluate(test=True)
        print("Loaded pretrained model!. Test accuracy: {}%".format(test_acc))
        irls(trainer, TOOLCHAIN, lp, eps)


def _test_linf():
    print("------DEBUG------")
    np.random.seed(0)
    trainer = Trainer(arch=ARCH, dataset=DATASET, bsz=128, embedding=None)
    #trainer.train(2, 0.01) 
    trainer.net.load_state_dict(torch.load('pretrained_model_ce_{}_{}.pth'.format(ARCH, DATASET)))
    test_acc = trainer.evaluate(test=True)
    print("Clean test accuracy: {}%".format(test_acc))

    np.random.seed(0)
    test_idx = np.random.choice(list(range(trainer.N_test)), 324)
    test_y = trainer.test_y[test_idx]
    
    test_adv = trainer.test_lp_attack(2, test_idx, 2., realizable=False)
    acc = trainer.evaluate(given_examples=(test_adv, test_y))
    print("[L{}] Adversarial accuracy: {}%".format(2, acc))
    print("------END DEBUG------")

#_test_linf()
#main()
#eps_grid()
eps_plot(None)
