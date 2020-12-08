import numpy as np
import cvxpy as cp
import sys
import os
import pickle
import scipy.io
import torch
import seaborn as sns
import torch.nn as nn
import re
import matplotlib.pyplot as plt

from PIL import Image
from cvxpy.atoms.norm import norm
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from neural_net import NN, CNN
from pdb import set_trace

SIGNAL = 'signal'
ATTACK = 'attack'
EPS = 10
SIGNAL_BIDXS = list()
ATTACK_BIDXS = list()

class Trainer(object):

    def __init__(self):
        self.net = NN(1400, 256, 38) 
        self.data = parse_data(normalize=True)
        train_full, test_full = split_train_test(self.data)
        train_y = []
        test_y = []
        for i in range(len(train_full)):
            train_y += [i for _ in range(len(train_full[i]))]
            test_y += [i for _ in range(len(test_full[i]))]
        self.train_X = np.vstack(train_full)
        self.test_X = np.vstack(test_full)
        self.train_y = np.array(train_y)
        self.test_y = np.array(test_y)
        self.N_train = self.train_X.shape[0]
        self.N_test = self.test_X.shape[0]

    def train(self, num_epochs, lr, bsz=32):
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(num_epochs): 
            idx = list(range(self.N_train))
            np.random.shuffle(idx)
            for i in range(self.N_train // bsz):
                batch_idx = idx[i*bsz:(i+1)*bsz]
                bx = self.train_X[batch_idx, :, :]
                by = self.train_y[batch_idx]
                bx = bx.reshape((bx.shape[0], -1))

                bx = torch.from_numpy(bx)
                by = torch.from_numpy(by)

                output = self.net.forward(bx.float())
                loss = loss_fn(output, by)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if epoch % 1 == 0:
                train_acc = self.evaluate(test=False)
                test_acc = self.evaluate(test=True)
                print("[{}] Loss: {}. Train Accuracy: {}%. Test Accuracy: {}%.".format(epoch, 
                    loss, train_acc, test_acc))

        torch.save(self.net.state_dict(), 'pretrained_model.pth')

    def evaluate(self, test=True):
        if test:
            X, y, sz = self.test_X, self.test_y, self.N_test
        else:
            X, y, sz = self.train_X, self.train_y, self.N_train
        bsz = 32
        with torch.no_grad():
            loss_fn = nn.CrossEntropyLoss(size_average=False)
            num_correct = 0
            idx = list(range(sz))
            np.random.shuffle(idx)
            for i in range(sz // bsz):
                batch_idx = idx[i*bsz:(i+1)*bsz]
                bx = X[batch_idx, :, :]
                by = y[batch_idx]
                bx = bx.reshape((bx.shape[0], -1))
                bx = torch.from_numpy(bx)
                by = torch.from_numpy(by)

                output = self.net.forward(bx.float())
                pred = output.data.argmax(1)
                num_correct += pred.eq(by.data.view_as(pred)).sum().item()
        acc = num_correct / X.shape[0] * 100.
        return acc

    def compute_lp_dictionary(self, eps, lp, block=False):
        idx = list(range(self.N_train))
        bsz = self.N_train
        dictionary = list()
        blocks = dict()
        for i in range(self.N_train // bsz):
            batch_idx = idx[i*bsz:(i+1)*bsz]
            bx = self.train_X[batch_idx, :, :]
            by = self.train_y[batch_idx]
            bx = bx.reshape((bx.shape[0], -1))
            bx = torch.from_numpy(bx).float()
            by = torch.from_numpy(by)

            delta = torch.zeros_like(bx, requires_grad=True)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(self.net(bx + delta), by)
            loss.backward()
            if lp == np.infty:
                bx_attack = (eps * delta.grad.detach().sign()).numpy()
            elif lp == 2:
                bx_attack = (eps * delta.grad.detach() / delta.grad.detach().norm()).numpy()
            else:
                print("Attack not supported!")
                sys.exit(1)
            dictionary = bx_attack
            for label in range(38):
                blocks[label] = bx_attack[(by == label).nonzero().squeeze()]
            if block:
                return blocks
        return dictionary.T

    # k-nearest neighbors
    def sample_attack(self, lp, ex_idx, k):
        blocks = self.compute_lp_dictionary(eps=EPS, lp=lp, block=True)

        D = [np.zeros(blocks[i].shape[0]) for i in range(38)]
        for i in range(10):
            D[i][0] = 1./3.
            D[i][1] = 1./3.
            D[i][2] = 1./3.
        perturb = [np.dot(blocks[i].T, D[i]) for i in range(38)]

        bx = self.test_X[ex_idx, :, :]
        by = self.test_y[ex_idx]
        bx = bx.reshape(-1)

        #neigh = NearestNeighbors(n_neighbors=k)
        #neigh.fit(self.train_X.reshape((self.train_X.shape[0], -1)))
        #dist, ind = neigh.kneighbors(bx.reshape((1, -1)))
        #set_trace()

        perturbation = np.array(perturb[by])
        adv = bx + perturbation
        return adv

def attack(im, attack_im, offset):
    x, y = attack_im.shape
    off_x, off_y = offset
    im[off_x:off_x+x, off_y:off_y+y] = attack_im
    return im 

def compute_dictionary(train, attack_im=None, offset=None):
    dictionary = list()
    for pid in range(len(train)):
        for i in range(train[pid].shape[0]):
            if attack_im is not None:
                x_adv = attack(train[pid][i, :, :], attack_im, offset)
                #plt.imshow(x_adv, cmap='gray')
                #plt.show()
                dictionary.append(x_adv.reshape(-1))
            else:
                x = train[pid][i, :, :]
                dictionary.append(x.reshape(-1))
    dictionary = np.array(dictionary).T
    return dictionary

# Format: data is a list of 38 subjects, each who have m_i x 192 x 168 images.
def parse_data(resize=True, normalize=False):
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
            if normalize:
                im = (im - 0.5) / 0.5
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
        m = 44 # ~70%
        train[pid] = data[pid][:m, :, :]
        test[pid] = data[pid][m:, :, :]
    return train, test

def to_mat():
    data = parse_data(resize=False)
    # Pad with zeros to make 64 images per person exactly.
    for pid in range(len(data)):
        if data[pid].shape[0] < 64:
            x, y = data[pid][0, :, :].shape
            data[pid] = np.concatenate([data[pid], np.zeros((64-data[pid].shape[0], x, y))])
    data = np.array(data)
    data = np.transpose(data, (2, 3, 1, 0))
    scipy.io.savemat('YaleBCrop.mat', {'I': data})
    set_trace()

def get_block(x, bidx, mode):
    global SIGNAL_BIDXS, ATTACK_BIDXS
    if mode == SIGNAL:
        if len(x.shape) == 2:
            return x[:, SIGNAL_BIDXS[bidx]:SIGNAL_BIDXS[bidx+1]]
        else:
            return x[SIGNAL_BIDXS[bidx]:SIGNAL_BIDXS[bidx+1]]
    elif mode == ATTACK:
        if len(x.shape) == 2:
            return x[:, ATTACK_BIDXS[bidx]:ATTACK_BIDXS[bidx+1]]
        else:
            return x[ATTACK_BIDXS[bidx]:ATTACK_BIDXS[bidx+1]]

def lp_reconstruct(trainer, ex_idx, lp, dict_in_obj=True, finegrained=False):
    print("-------------------{}-{}---------------------".format(ex_idx, lp))
    l2_dict = trainer.compute_lp_dictionary(EPS, 2)
    linf_dict = trainer.compute_lp_dictionary(EPS, np.infty)
    attack_dict = np.hstack([l2_dict, linf_dict])
    attack_dict = normalize(attack_dict, axis=0)
    x_adv = trainer.sample_attack(lp, ex_idx, k=5)

    data = parse_data()
    train, test = split_train_test(data)
    global SIGNAL_BIDXS
    SIGNAL_BIDXS = [0]
    for i in range(len(train)):
        SIGNAL_BIDXS.append(SIGNAL_BIDXS[-1] + train[i].shape[0])
    signal_dict = compute_dictionary(train)
    signal_dict = normalize(signal_dict, axis=0)

    concat_dict = np.hstack([signal_dict, attack_dict])

    s_len, l2_len, linf_len = signal_dict.shape[1], l2_dict.shape[1], linf_dict.shape[1]

    attack_lens = [l2_len, linf_len]
    # Attack comes after signal in concat_dict
    global ATTACK_BIDXS
    if finegrained:
        ATTACK_BIDXS = [s_len + i for i in SIGNAL_BIDXS]
        ATTACK_BIDXS += [s_len+l2_len + i for i in SIGNAL_BIDXS]
    else: 
        ATTACK_BIDXS = [s_len, s_len+l2_len, s_len+l2_len+linf_len]

    thres = 0.1

    c = cp.Variable(s_len + sum(attack_lens))

    idx = 0
    objective = 0
    for pid in range(len(train)):
        l = train[pid].shape[0]
        if dict_in_obj:
            objective += norm(concat_dict[:, idx:idx+l]@c[idx:idx+l], p=2)
        else:
            objective += norm(c[idx:idx+l], p=2)
        idx += l

    for a in range(len(attack_lens)):
        a_len = attack_lens[a]
        if dict_in_obj:
            objective += norm(concat_dict[:, idx:idx+a_len]@c[idx:idx+a_len], p=2)
        else:
            objective += norm(c[idx:idx+a_len], p=2)
        idx += a_len
    minimizer = cp.Minimize(objective)
    constraints = [norm(x_adv - concat_dict@c, p=2) <= thres]

    prob = cp.Problem(minimizer, constraints)

    result = prob.solve(verbose=True, solver=cp.MOSEK)

    opt = np.array(c.value)

    eps = np.zeros((len(SIGNAL_BIDXS) - 1, len(ATTACK_BIDXS) - 1))
    min_val = np.infty
    argmin_val = None
    for i in range(len(SIGNAL_BIDXS) - 1):
        for j in range(len(ATTACK_BIDXS) - 1):
            try:
                assert s_len == l2_len and s_len == linf_len
                attack_d = get_block(concat_dict, j, ATTACK)
                attack_person_d = get_block(attack_d, i, SIGNAL)
                attack_c = get_block(opt, j, ATTACK)
                attack_person_c = get_block(attack_c, i, SIGNAL)
                res = x_adv - \
                      get_block(concat_dict, i, SIGNAL) @ get_block(opt, i, SIGNAL) - \
                      attack_person_d @ attack_person_c 
                      #get_block(concat_dict, j, ATTACK) @ get_block(opt, j, ATTACK)
            except:
                print("ERROR!")
                set_trace()
            eps[i][j] = np.linalg.norm(res, 2)
            if eps[i][j] < min_val:
                min_val = eps[i][j]
                argmin_val = i, j
    print(eps)
    print("min_val at person {}, attack {}. value: {}".format(argmin_val[0], argmin_val[1], min_val))
    c_a = opt[s_len:]
    #plt.plot(c_a)
    #plt.show()

    ax = sns.heatmap(eps, annot=False, fmt="0.0f",
                xticklabels=range(1, eps.shape[1]+1), yticklabels=range(1, eps.shape[0]+1))
    #plt.show()

    perturb = 1./3 * linf_dict[:, 0] + 1./3 *linf_dict[:, 1] + 1./3*linf_dict[:, 2]
    signal_res = x_adv - \
        get_block(concat_dict, 0, SIGNAL) @ get_block(opt, 0, SIGNAL) - perturb
    res = x_adv - \
        get_block(concat_dict, 0, SIGNAL) @ get_block(opt, 0, SIGNAL) - \
        get_block(concat_dict, 1, ATTACK) @ get_block(opt, 1, ATTACK)
    set_trace()

def reconstruct():
    # Adversarial Dictionary: D x N
    D = 1400
    data = parse_data()
    train, test = split_train_test(data)

    dog = np.array(Image.open('dog.png').resize((15, 15), resample=Image.LANCZOS).convert('L'))
    mario = np.array(Image.open('mario.png').resize((15, 15), resample=Image.LANCZOS).convert('L'))
    x = test[0][0, :, :]
    x_adv = attack(x, dog, (0, 0)).reshape(-1)

    #dog_dict = compute_dictionary(train, dog, (0, 0))
    #mario_dict = compute_dictionary(train, mario, (15, 15))

    dog_dict = attack(np.zeros(x.shape), dog, (0, 0)).reshape((-1, 1))
    mario_dict = attack(np.zeros(x.shape), mario, (0, 0)).reshape((-1, 1))

    adv_dict = np.hstack([dog_dict, mario_dict])

    # Signal Dictionary: D x n_atoms
    signal_dict = compute_dictionary(train)

    concat_dict = np.hstack([signal_dict, dog_dict, mario_dict])

    s_len, dog_len, mario_len = signal_dict.shape[1], dog_dict.shape[1], mario_dict.shape[1]

    attack_lens = [dog_len, mario_len]

    thres = 0.1

    #c_l2 = cp.Variable(N)
    c = cp.Variable(s_len + dog_len + mario_len)
    #c_s = cp.Variable(n_atoms)

    #objective = cp.Minimize(norm(c_s, p=2) + cp.sum(norm(c_l2, p=2) + norm(c_linf, p=2)))
    #objective = cp.Minimize(cp.sum(norm(c_l2, p=2) + norm(c_linf, p=2)))

    idx = 0
    objective = 0
    for pid in range(len(train)):
        l = train[pid].shape[0]
        objective += norm(concat_dict[:, idx:idx+l]@c[idx:idx+l], p=2)
        idx += l

    x = x.reshape(-1)

    #for a in range(len(attack_lens)):
    #    a_len = attack_lens[a]
    #    objective += norm(concat_dict[:, idx:idx+a_len]@c[idx:idx+a_len], p=2)
    minimizer = cp.Minimize(0.1*objective + norm(x - signal_dict@c[:s_len], p=2)) 
    #constraints = [norm(x_adv - concat_dict@c, p=2) <= thres]
    #constraints = [norm(x - signal_dict@c[:s_len], p=2) <= thres]

    #prob = cp.Problem(minimizer, constraints)
    prob = cp.Problem(minimizer)

    result = prob.solve(verbose=True, max_iters=10)
    set_trace()
    

np.random.seed(0)
trainer = Trainer()
trainer.train(10, 1e-4) 
#lp_reconstruct(trainer, 0, 2, dict_in_obj=False)
lp_reconstruct(trainer, 0, np.infty, dict_in_obj=False)
#lp_reconstruct(trainer, 10, 2)
#lp_reconstruct(trainer, 10, np.infty)
#lp_reconstruct(trainer, 20, 2)
#lp_reconstruct(trainer, 20, np.infty)
