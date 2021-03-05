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
from advertorch.attacks import SparseL1DescentAttack, L2PGDAttack
from foolbox.attacks import FGSM, L2PGD, L1PGD
from foolbox.criteria import Misclassification
from foolbox.utils import samples
from irls import BlockSparseIRLSSolver

DATASET = 'mnist'
#EPS = {'mnist': {1: 10.0, \
#       2: 2.0, \
#       np.infty: 0.3}}
EPS = {'mnist': {1: 0.25, \
       2: 0.3, \
       np.infty: 0.3}}
EPS = EPS[DATASET]
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

    def __init__(self, use_cnn=True, linear=False, dataset='yale'):
        self.use_cnn = use_cnn
        self.dataset = dataset
        if self.dataset == 'yale':
            self.d = 1400
            self.num_classes = 38
        elif self.dataset == 'cifar':
            self.d = 32*32*3
            self.num_classes = 10
        elif self.dataset == 'mnist':
            self.d = 28*28
            self.num_classes = 10
        if use_cnn:
            self.net = CNN(m=32, num_layers=3, in_channels=1, 
                num_classes=self.num_classes, linear=linear)
        else:
            self.net = NN(self.d, 256, self.num_classes, linear=linear) 
        self.loss_fn = nn.CrossEntropyLoss()
        #self.loss_fn = nn.MultiMarginLoss()
        self.train_loader, self.test_loader = self.preprocess_data()

    # Normalize image to [0, 1] and then standardize.
    def preprocess_data(self):
        transform = transforms.Compose(
                [transforms.ToTensor()])
                 #transforms.Normalize((MEAN_MAP[self.dataset]), (STD_MAP[self.dataset]))])

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

        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32,
            shuffle=True, worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=100,
            shuffle=True, worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))
        return train_loader, test_loader

    def train(self, num_epochs, lr, bsz=32):
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        for epoch in range(num_epochs): 
            for batch_idx, data in enumerate(self.train_loader):
                bx = data[0]
                by = data[1]
                if not self.use_cnn:
                    bx = bx.flatten(1)

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

        torch.save(self.net.state_dict(), \
            'pretrained_model_ce_{}.pth'.format(self.dataset))

    def evaluate(self, test=True, given_examples=None):
        if test:
            data_loader = self.test_loader
        else:
            data_loader = self.train_loader
        with torch.no_grad():
            loss_fn = nn.CrossEntropyLoss(reduction='sum')
            num_correct = 0
            if given_examples is not None:
                data_loader = [given_examples]
            for (bx, by) in data_loader:
                if given_examples is not None:
                    bx = torch.from_numpy(bx)
                    by = torch.from_numpy(by)
                bx = bx.squeeze()
                if not self.use_cnn:
                    bx = bx.flatten(1)
                output = self.net.forward(bx.float())
                pred = output.data.argmax(1)
                num_correct += pred.eq(by.data.view_as(pred)).sum().item()
        if given_examples is not None:
            acc = num_correct / len(by) * 100.
        else:
            acc = num_correct / len(data_loader.dataset) * 100.
            
        return acc

    def compute_train_dictionary(self, normalize_cols=True, embedding=None):
        train = self.train_full
        dictionary = list()
        for pid in range(len(train)):
            for i in range(train[pid].shape[0]):
                x = train[pid][i, :, :]
                dictionary.append(x.reshape(-1))
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

        out = self._lp_attack(lp, eps, bx, by, only_delta=True)
        dictionary = out
        dictionary = dictionary.reshape((dictionary.shape[0], -1))

        for label in range(self.num_classes):
            blocks[label] = dictionary[torch.nonzero(by == label).squeeze()]
        if block:
            return blocks
        if normalize_cols:
            return normalize(dictionary.T, axis=0)
        else:
            return dictionary.T

    def _lp_attack(self, lp, eps, bx, by, debug=False, only_delta=False, scale=True):
        d = bx.shape[1]
        bx.requires_grad = True
        loss = self.loss_fn(self.net(bx.float()), by)
        self.net.zero_grad()
        loss.backward()
        data_grad = bx.grad.data

        if scale:
            linf = bx + eps * data_grad.sign()
            l2 = bx + eps * d**0.5 * data_grad / torch.stack(d*[data_grad.norm(dim=1) + 1e-8]).T
            adversary = SparseL1DescentAttack(
                self.net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=d*eps,
                nb_iter=150, eps_iter=eps, rand_init=True, clip_min=-np.inf, clip_max=np.inf,
            targeted=False)

            l1 = adversary.perturb(bx, by)
            if only_delta:
                linf = linf - bx
                l2 = l2 - bx
                l1 - l1 - bx 
            linf = linf.detach().numpy()
            l2 = l2.detach().numpy()
            l1 = l1.detach().numpy()
        else:
            #linf = bx + eps * data_grad.sign()
            #l2 = bx + eps * data_grad / torch.stack(d*[data_grad.norm(dim=1) + 1e-8]).T
          
            self.net.zero_grad()
            self.net.eval()
            fmodel = foolbox.models.PyTorchModel(self.net, bounds=(0, 1)) 
            criterion = Misclassification(by) 
            if lp == np.infty: 
                linf_adv = FGSM()
                _, linf, _ = linf_adv(fmodel, bx, criterion, epsilons=[eps])
                linf = linf[0]
                #linf = bx + eps * data_grad.sign()
                #set_trace()
                if only_delta:
                    linf = linf - bx
                linf = linf.detach().numpy()
                return linf

            elif lp == 2:
                l2_adv = L2PGD(rel_stepsize=0.1, steps=200)
                l2_adversary = L2PGDAttack(
                    self.net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                    nb_iter=200, eps_iter=0.1, rand_init=True, clip_min=0, clip_max=1,
                    targeted=False)
                #images, labels = samples(fmodel, dataset="mnist", batchsize=20) 
                #set_trace()
                _, l2, _  = l2_adv(fmodel, bx, criterion, epsilons=[eps])
                l2 = l2[0]
                #l2 = l2_adversary.perturb(bx, by)
                if only_delta:
                    l2 = l2 - bx
                l2 = l2.detach().numpy()
                return l2
           
            elif lp == 1: 
                l1_adv = L1PGD(rel_stepsize=1.0, steps=200)
                l1_adversary = SparseL1DescentAttack(
                    self.net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                    nb_iter=200, eps_iter=10.0, rand_init=True, clip_min=0, clip_max=1,
                    targeted=False)
                #set_trace()
                #l1 = l1_adversary.perturb(bx, by)
                _, l1, _ = l1_adv(fmodel, bx, criterion, epsilons=[eps])
                l1 = l1[0]
                if only_delta:
                    l1 = l1 - bx
                l1 = l1.detach().numpy()
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
        test_adv = trainer.test_lp_attack(lp, test_idx, EPS[attack], realizable=False)
        all_attacks.append(test_adv)

        acc = trainer.evaluate(given_examples=(test_adv, trainer.test_y[test_idx]))
        print("[L{}] Adversarial accuracy: {}".format(lp, acc))
        if lp == np.infty:
            scipy.io.savemat('mats/{}_ce/linf_eps{}.mat'.format(dst, EPS[attack]), {'data': test_adv})
        elif lp == 2:
            scipy.io.savemat('mats/{}_ce/l2_eps{}.mat'.format(dst, EPS[attack]), {'data': test_adv})
        elif lp == 1:
            scipy.io.savemat('mats/{}_ce/l1_eps{}.mat'.format(dst, EPS[attack]), {'data': test_adv})

    all_attacks = np.array(all_attacks)
    avg = np.mean(all_attacks, axis=0)
    scipy.io.savemat('mats/{}_ce/avg_eps{}.mat'.format(dst, EPS[np.infty]), {'data': avg})

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
    Ds = compute_dictionary(trainer.train_full)
    s_len = Ds.shape[1]

    thres = 10
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
    atts = ['l2', 'linf']
    att_map = {'l2': 2, 'linf': np.inf}
    result_map = {'l2': {'signal': [], 'att': [], 'test': [], 'backtest': [], 'baseline': []}, 
                  'linf': {'signal': [], 'att': [], 'test': [], 'backtest': [], 'baseline': []}}

    np.random.seed(0)
    test_idx = np.random.choice(list(range(trainer.N_test)), 324)[:100]
    test_y = scipy.io.loadmat('mats/mnist_ce/test_y.mat')['data'][0, :100]
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

    result_map = pickle.load(open('mats/mnist_recon/result_map.pkl', 'rb'))

    for att in atts:
        plt.plot(epss, result_map[att]['signal'], label='Signal Classification', marker='*')
        plt.plot(epss, result_map[att]['att'], label='Attack Detection', marker='*')
        plt.plot(epss, result_map[att]['test'], label='NN Adversarial', marker='*')
        plt.plot(epss, result_map[att]['backtest'], label='NN Denoising', marker='*')
        plt.plot(epss, result_map[att]['baseline'], label='Baseline', marker='*')
        plt.legend()
        plt.grid()
        plt.xlabel('Epsilon of Perturbation')
        plt.ylabel('Accuracy')
        plt.yticks(np.arange(0, 101, 10))
        plt.title('Test Accuracies for 100 data points perturbed with {} attack'.format(att))
        plt.show()
    set_trace()    

def irls(trainer, toolchain, lp):
    print("L{} attacks, eps = {}".format(lp, EPS[lp]))
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
    
    test_adv = trainer.test_lp_attack(lp, test_idx, EPS[attack], realizable=False)
    acc = trainer.evaluate(given_examples=(test_adv, test_y))
    print("[L{}] Adversarial accuracy: {}%".format(lp, acc))

    solver = BlockSparseIRLSSolver(Ds, Da, trainer.num_classes, num_attacks, sz)
    class_preds = list()
    attack_preds = list()
    denoised = list()
    for t in range(20):
        if t % 5 == 0:
            print(t)
        x = test_adv[t, :, :]
        x = x.reshape(-1)
        cs_est, ca_est, Ds_est, Da_est, err_cs, ws, wa, active_classes = solver.solve(x)

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
        attack_preds.append(np.argmin(err_attack))
        denoised.append(x - Da_est@ca_est)
    class_preds = np.array(class_preds)
    attack_preds = np.array(attack_preds)
    denoised = np.array(denoised)
    set_trace()
        

def main():
    np.random.seed(0)
    trainer = Trainer(use_cnn=False, dataset=DATASET)
    #trainer.train(20, 1e-2) 
    trainer.net.load_state_dict(torch.load('pretrained_model_ce_{}.pth'.format(DATASET)))
    test_acc = trainer.evaluate(test=True)
    print("Loaded pretrained model!. Test accuracy: {}%".format(test_acc))
    toolchain = [2, np.infty]
    irls(trainer, toolchain, 2)
    #serialize_dictionaries(trainer, toolchain)

main()
