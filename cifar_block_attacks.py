import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import cvxpy as cp
import sys
import os
import pickle

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from cvxpy.atoms.norm import norm
from pdb import set_trace

class NN(nn.Module):

    def __init__(self, d, m):
        super().__init__()
        self.d = d
        self.m = m
        self.lin1 = nn.Linear(self.d, self.m)
        self.lin2 = nn.Linear(self.m, self.m)
        self.lin3 = nn.Linear(self.m, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        out = self.relu(self.lin1(out))
        out = self.relu(self.lin2(out))
        out = self.lin3(out)
        return out

class CNN(nn.Module):

    def __init__(self, m, num_layers, in_channels=3, num_classes=10):
        super().__init__()
        layers = []
        for i in range(num_layers):
            conv2d = nn.Conv2d(in_channels, m, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU()]
            in_channels = m
        self.features = nn.ModuleList(layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(m*7*7, num_classes)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)  
        return x

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val]
    return weight

def get_loader(dataset, data_augment=False, bsz=512, shuffle=True):
    train_sz = lambda x: round(len(x) * 0.8)
    test_sz = lambda x: round(len(x) * 0.2)
    if dataset == 'cifar10_train':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10('files/', train=True, download=True,
                 transform=transform_train if data_augment else transform_test)
        weights = make_weights_for_balanced_classes(dataset.targets, 10)
        weights = torch.DoubleTensor(weights)                                       
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, int(0.05*len(weights)))
        loader = torch.utils.data.DataLoader(dataset,
              batch_size=bsz,
              worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)),
              sampler=sampler)
    elif dataset == 'cifar10_test':
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR10('files/', train=False, download=True,
                             transform=transform_test),
          batch_size=bsz, shuffle=False,
          worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))
    return loader

class Trainer(object):

    def __init__(self):
        self.net = NN(32*32*3, 256) 
        #self.net = NN(32, 2) 
        self.train_loader = get_loader('cifar10_train')
        self.test_loader = get_loader('cifar10_test')

    def train(self, num_epochs, lr):
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for i in range(num_epochs): 
            for batch_idx, data in enumerate(self.train_loader):
                bx, by = data[0], data[1]
                #bx = bx.reshape((bx.shape[0], -1))
                output = self.net.forward(bx.float())
                loss = loss_fn(output, by)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if i % 1 == 0:
                train_acc = self.evaluate(test=False)
                test_acc = self.evaluate(test=True)
                print("[{}] Loss: {}. Train Accuracy: {}%. Test Accuracy: {}%.".format(i, 
                    loss, train_acc, test_acc))

        torch.save(self.net.state_dict(), 'pretrained_model_cnn.pth')

    def evaluate(self, test=True):
        if test:
            data_loader = self.test_loader
        else:
            data_loader = self.train_loader
        with torch.no_grad():
            loss_fn = nn.CrossEntropyLoss(size_average=False)
            num_correct = 0
            for (batch_X, batch_y) in data_loader:
                #batch_X = batch_X.reshape((batch_X.shape[0], -1))
                if torch.cuda.is_available():
                    batch_X = batch_X.cuda()
                    batch_y = batch_y.cuda() 
                output = self.net.forward(batch_X)
                pred = output.data.argmax(1)
                num_correct += pred.eq(batch_y.data.view_as(pred)).sum().item()
        acc = num_correct / len(data_loader.dataset) * 100.
        return acc

def plot(flattened):
    im = flattened.reshape((1, 3, 32, 32))
    im = np.transpose(im, [0, 3, 2, 1])
    plt.imshow(im[0, :])
    plt.show()

def serialize(data, path, name):
    path = os.path.join(path, name)
    pickle.dump(data, open(path, 'wb'))

class DictionaryAttacker():

    def __init__(self):
        self.net = NN(32*32*3, 256) 
        #self.net = NN(128, 3) 
        self.net.load_state_dict(torch.load('pretrained_model.pth')) 

    # Use FGSM. 
    def compute_train_dictionary(self, eps, lp, block=False):
        blocks = dict()
        train_loader = get_loader('cifar10_train', bsz=50000)
        for bx, by in train_loader:
            bx = bx.reshape((bx.shape[0], -1))
            delta = torch.zeros_like(bx, requires_grad=True)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(self.net(bx + delta), by)
            loss.backward()
            if lp == np.infty:
                attack = (eps * delta.grad.detach().sign()).numpy()
            elif lp == 2:
                attack = (eps * delta.grad.detach() / delta.grad.detach().norm()).numpy()
            else:
                print("Attack not supported!")
                sys.exit(1)

            if block:
                for label in range(10):
                    blocks[label] = attack[(by == label).nonzero().squeeze()]
                return blocks
        return attack

    def attack(self):
        blocks = self.compute_train_dictionary(eps=0.1, lp=np.infty, block=True)
        test_loader = get_loader('cifar10_test', bsz=100)
        #D = [blocks[i].shape[0] for i in range(10)]
        #perturb = [np.dot(blocks[i].T, 1.0/D[i] * np.ones(D[i])) for i in range(10)]
        D = [np.zeros(blocks[i].shape[0]) for i in range(10)]
        for i in range(10):
            D[i][0] = 1
        perturb = [np.dot(blocks[i].T, D[i]) for i in range(10)]
        num_correct = 0
        num_adv_correct = 0
        for bx, by in test_loader:
            bx = bx.reshape((bx.shape[0], -1))
            perturbation = np.array([perturb[y] for y in by])
            adv = bx.numpy() + perturbation

            #serialize(adv, 'cifar_dicts', 'adv.pkl')
            #serialize(bx, 'cifar_dicts', 'clean_x.pkl')
            #sys.exit(0)
            #plot(bx[5, :])
            #plot(adv[5, :])
            output = self.net.forward(bx)
            pred = output.data.argmax(1)
            num_correct += pred.eq(by.data.view_as(pred)).sum().item()
            
            adv_output = self.net.forward(torch.from_numpy(adv).float())
            adv_pred = adv_output.data.argmax(1)
            num_adv_correct += adv_pred.eq(by.data.view_as(adv_pred)).sum().item()
        acc = num_correct / len(test_loader.dataset) * 100.
        adv_acc = num_adv_correct / len(test_loader.dataset) * 100.
        print("Test accuracy: {}%".format(acc))
        print("Adv Test accuracy: {}%".format(adv_acc))

def ricker_function(resolution, center, width):
    """Discrete sub-sampled Ricker (mexican hat) wavelet"""
    x = np.linspace(0, resolution - 1, resolution)
    x = (2 / ((np.sqrt(3 * width) * np.pi ** 1 / 4))) * (
         1 - ((x - center) ** 2 / width ** 2)) * np.exp(
         (-(x - center) ** 2) / (2 * width ** 2))
    return x

def ricker_matrix(width, resolution, n_atoms):
    """Dictionary of Ricker (mexican hat) wavelets"""
    centers = np.linspace(0, resolution - 1, n_atoms)
    D = np.empty((n_atoms, resolution))
    for i, center in enumerate(centers):
        D[i] = ricker_function(resolution, center, width)
    D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
    return D

def main():
    #trainer = Trainer()
    #trainer.train(10, 0.05)

    attacker = DictionaryAttacker()

    attacker.attack()

    l2_dict= attacker.compute_train_dictionary(eps=1, lp=2)
    linf_dict = attacker.compute_train_dictionary(eps=0.1, lp=np.infty)
    if not os.path.exists('cifar_dicts'):
        os.mkdir('cifar_dicts')
    serialize(l2_dict, 'cifar_dicts', 'l2_eps1.pkl')
    serialize(linf_dict, 'cifar_dicts', 'linf_eps0.1.pkl')

def reconstruct():
    # Adversarial Dictionary: D x N
    D = 3072
    l2_dict = pickle.load(open('cifar_dicts/l2_eps1.pkl', 'rb')).T
    linf_dict = pickle.load(open('cifar_dicts/linf_eps0.1.pkl', 'rb')).T
    N = l2_dict.shape[1]

    # Wavelet Dictionary: D x n_atoms
    width = 100
    n_atoms = D // 3

    dic = ricker_matrix(width, D , n_atoms).T

    x_adv = pickle.load(open('cifar_dicts/adv.pkl', 'rb'))[5, :]
    x = pickle.load(open('cifar_dicts/clean_x.pkl', 'rb'))[5, :]

    delta = x_adv - x.numpy()

    thres = 0.1

    print("N = {}, D = {}, n_atoms = {}".format(N, D, n_atoms))

    #c_l2 = cp.Variable(N)
    c_linf = cp.Variable(N)
    #c_s = cp.Variable(n_atoms)

    #objective = cp.Minimize(norm(c_s, p=2) + cp.sum(norm(c_l2, p=2) + norm(c_linf, p=2)))
    #objective = cp.Minimize(cp.sum(norm(c_l2, p=2) + norm(c_linf, p=2)))
    objective = cp.Minimize(norm(linf_dict*c_linf, p=1))
    #constraints = [norm(delta - l2_dict*c_l2 - linf_dict*c_linf, p=2) <= thres]
    constraints = [norm(delta - linf_dict*c_linf, p=2) <= thres]

    prob = cp.Problem(objective, constraints)

    set_trace()

    result = prob.solve(verbose=True, max_iters=3)
    set_trace()
    

#reconstruct()
main()
