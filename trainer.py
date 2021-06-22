import numpy as np
import sys
import os
import pickle
import scipy.io
import torch
import torchvision
import torch.nn as nn
import utils
import copy
import baselines

from sklearn.preprocessing import normalize
from neural_net import NN, CNN
from pdb import set_trace
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from advertorch.attacks import SparseL1DescentAttack, L2PGDAttack, LinfPGDAttack, CarliniWagnerL2Attack

class Trainer(object):

    def __init__(self, args, use_maini_cnn=False):
        self.args = args
        self.arch = self.args.arch
        self.use_cnn = 'cnn' in self.arch
        self.bsz = self.args.bsz
        self.dataset = self.args.dataset
        self.eps_map = utils.EPS[self.dataset]
        self.step_map = utils.STEP[self.dataset]
        self.embedding = self.args.embedding
        self.J = self.args.J
        self.L = self.args.L
        if self.dataset == 'yale':
            self.d = 1400
            self.num_classes = 38
            self.in_channels = 1
        elif self.dataset == 'cifar':
            self.input_shape = (32, 32)
            self.d = 32*32*3
            self.num_classes = 10
            #if self.embedding is None:
            self.in_channels = 3
            #else:
            #    self.in_channels = int(1 + self.L*self.J + (self.L**2*self.J*(self.J - 1))/2)
        elif self.dataset == 'mnist':
            self.input_shape= (28, 28)
            self.d = 28*28
            self.num_classes = 10
            #if self.embedding is None:
            self.in_channels = 1
            #else:
            #    self.in_channels = int(1 + self.L*self.J + (self.L**2*self.J*(self.J - 1))/2)
        if self.use_cnn:
            d_arch = '{}_{}'.format(self.arch, self.dataset)
            self.net = CNN(arch=d_arch, embedding=self.embedding, in_channels=self.in_channels,
                num_classes=self.num_classes)
        else:
            self.net = NN(self.d, 256, self.num_classes, linear=False) 
        if use_maini_cnn:
            self.net = baselines.net()
            self.maini_attack = True
        else:
            self.maini_attack = False
        self.scattering = utils.ScatteringTransform(J=self.J, L=self.L, shape=self.input_shape)
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.loss_fn = nn.CrossEntropyLoss()
        #self.loss_fn = nn.MultiMarginLoss()
        self.train_loader, self.test_loader = self.preprocess_data()

    def load_model(self, model_name):
        device_id = 0
        device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(int(device_id))
        model_address = "files/MNIST_Baseline_Models/{}.pt".format(model_name.upper())
        self.net.load_state_dict(torch.load(model_address, map_location=device))
        print("Loaded model {}".format(model_name))

    # Parse data and normalize image to [0, 1]
    def preprocess_data(self):
        transform = transforms.Compose(
                [transforms.ToTensor()])

        if self.dataset == 'yale':
            data, raw_train_full, raw_test_full = utils.parse_yale_data()
            self.train_y = []
            self.test_y = []
            for i in range(len(raw_train_full)):
                self.train_y += [i for _ in range(len(raw_train_full[i]))]
                self.test_y += [i for _ in range(len(raw_test_full[i]))]
            raw_train_X = np.vstack(raw_train_full)
            raw_test_X = np.vstack(raw_test_full)

            self.train_dataset = utils.YaleDataset(raw_train_X, self.train_y, transform=transform)
            self.test_dataset = utils.YaleDataset(raw_test_X, self.test_y, transform=transform)
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
            if len(train_X[y]) >= utils.SIZE_MAP[self.dataset]:
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

    def train(self):
        num_epochs = self.args.num_epochs
        lr = self.args.lr
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

        save_path = 'files/pretrained_model_ce_{}_{}.pth'.format(self.arch, self.dataset)
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
                    dictionary.append(embed_x.reshape(-1))
                       
        dictionary = np.array(dictionary).T
        if normalize_cols:
            return normalize(dictionary, axis=0)
        else:
            return dictionary

    def compute_lp_dictionary(self, eps, lp, block=False, net=None, lp_variant=None):
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
        if net is not None:
            print("USING SOME CNN")
            d_arch = '{}_{}'.format(self.arch, self.dataset)
            net = CNN(arch=d_arch, embedding=self.embedding, in_channels=self.in_channels,
                num_classes=self.num_classes)
            if torch.cuda.is_available():
                net = net.cuda()
            net.load_state_dict(torch.load('files/pretrained_model_ce_{}_{}.pth'.format(self.arch, self.dataset)))
        for i in range(0, bx.shape[0], step):
            batch_x = bx[i:i+step]
            batch_y = by[i:i+step]
            out = self._lp_attack(lp, eps, batch_x, batch_y, only_delta=True, net=net, lp_variant=lp_variant)
            out = out.reshape((out.shape[0], -1))
            dictionary.append(out)
        dictionary = np.concatenate(dictionary)

        by = by.cpu()
        for label in range(self.num_classes):
            blocks[label] = dictionary[torch.nonzero(by == label).squeeze()]
        if block:
            return blocks
        try:
            return normalize(dictionary.T, axis=0)
        except:
            set_trace()

    def _lp_attack(self, lp, eps, bx, by, only_delta=False, net=None, lp_variant=None):
        if eps == 0:
            if only_delta:
                return (bx - bx).detach().numpy()
            else:
                return bx.detach().numpy()
        d = bx.flatten(1).shape[1]
        if torch.cuda.is_available():
            bx = bx.cuda()
            by = by.cuda()
            step_size = self.step_map[lp]

            if net is None:
                net = self.net
            net.eval()
            net.zero_grad()

            if lp == np.infty: 
                adversary = LinfPGDAttack(
                    net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                    nb_iter=100, eps_iter=step_size, rand_init=True, clip_min=0, clip_max=1,
                    targeted=False)
                if self.maini_attack:
                    delta = utils.pgd_linf(net, bx, by, restarts=10, num_iter=100, epsilon=eps)
                    if only_delta:
                        out = delta
                    else:
                        out = bx + delta
                    out = out.cpu().detach().numpy()
                    return out
            elif lp == 2 and lp_variant is None:
                adversary = L2PGDAttack(
                    net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                    nb_iter=200, eps_iter=step_size, rand_init=True, clip_min=0, clip_max=1,
                    targeted=False)
                if self.maini_attack:
                    delta = utils.pgd_l2(net, bx, by, restarts=10, num_iter=200, epsilon=eps)
                    if only_delta:
                        out = delta
                    else:
                        out = bx + delta
                    out = out.cpu().detach().numpy()
                    return out
            elif lp == 1:
                adversary = SparseL1DescentAttack(
                    net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                    nb_iter=100, eps_iter=step_size, rand_init=True, clip_min=0, clip_max=1,
                    targeted=False)
                if self.maini_attack:
                    delta = utils.pgd_l1_topk(net, bx, by, restarts=10, num_iter=100, epsilon=eps)
                    if only_delta:
                        out = delta
                    else:
                        out = bx + delta
                    out = out.cpu().detach().numpy()
                    return out
            elif lp == 2 and lp_variant == 'cw':
                adversary = CarliniWagnerL2Attack(
                    net, num_classes=self.num_classes, learning_rate=step_size, \
                    max_iterations=100)
            out = adversary.perturb(bx, by)
            if only_delta:
                out = out - bx
            out = out.cpu().detach().numpy()
            return out

    def test_lp_attack(self, lp, bx, by, eps, realizable=False, lp_variant=None):
        bx = torch.from_numpy(bx)
        by = torch.from_numpy(by)

        if realizable:
            blocks = self.compute_lp_dictionary(eps=self.eps_map[lp], lp=lp, block=True)

            D = [np.zeros(blocks[i].shape[0]) for i in range(self.num_classes)]
            for i in range(10):
                D[i][0] = 1./3.
                D[i][1] = 1./3.
                D[i][2] = 1./3.
            perturb = np.array([np.dot(blocks[i].T, D[i]) for i in range(self.num_classes)])

            perturbation = perturb[by, :]
            bx = bx.reshape((bx.shape[0], -1)).numpy()
            adv = bx + perturbation
            adv = adv.reshape((adv.shape[0], 1, 28, 28))
            return adv
        else:
            bsz, channels, r, c = bx.shape[0], bx.shape[1], bx.shape[2], bx.shape[3]
            if not self.use_cnn:
                bx = bx.flatten(1)
            d = bx.shape[1]

            out = self._lp_attack(lp, eps, bx, by, lp_variant=lp_variant)
        
            if not self.use_cnn:
                out = out.reshape((bsz, channels, r, c))
            return out

