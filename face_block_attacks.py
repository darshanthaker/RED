import numpy as np
import cvxpy as cp
import sys
import os
import pickle
import scipy.io
import torch
import seaborn as sns; #sns.set(rc={'text.usetex' : True}); sns.set_style("ticks")
import torch.nn as nn
import re
import matplotlib.pyplot as plt
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

SIGNAL = 'signal'
ATTACK = 'attack'
ATTACK_F = 'attack_f'
EPS = 15
#MEAN = 0.2728
#STD = 0.2453
MEAN = 69.5591
STD = 62.5569
SIGNAL_BIDXS = list()
ATTACK_BIDXS = list()
ATTACK_BIDXS_F = list()

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

            x = utils.to_tensor_custom(x)

            if self.transform:
                x = self.transform(x)
            return x, y

class Trainer(object):

    def __init__(self):
        #self.net = NN(1400, 256, 38) 
        self.net = CNN(m=32, num_layers=3, in_channels=1, num_classes=38)
        self.train_loader, self.test_loader = self.preprocess_data()

    # Normalize image to [0, 1] and then standardize.
    def preprocess_data(self):
        data = parse_data()
        raw_train_full, raw_test_full = split_train_test(data)
        self.train_y = []
        self.test_y = []
        for i in range(len(raw_train_full)):
            self.train_y += [i for _ in range(len(raw_train_full[i]))]
            self.test_y += [i for _ in range(len(raw_test_full[i]))]
        raw_train_X = np.vstack(raw_train_full)
        raw_test_X = np.vstack(raw_test_full)
        self.train_y = np.array(self.train_y)
        self.test_y = np.array(self.test_y)
        self.N_train = raw_train_X.shape[0]
        self.N_test = raw_test_X.shape[0]

        transform = transforms.Compose(
                [#transforms.ToTensor(),
                 transforms.Normalize((MEAN), (STD))])
        self.train_dataset = YaleDataset(raw_train_X, self.train_y, transform=transform)
        self.test_dataset = YaleDataset(raw_test_X, self.test_y, transform=transform)

        train_X = [list() for i in range(38)]
        test_X = [list() for i in range(38)]
        for i in range(self.N_train):
            x, y = self.train_dataset[i]
            train_X[y].append(x.numpy())
        for i in range(self.N_test):
            x, y = self.test_dataset[i]
            test_X[y].append(x.numpy())
        train_X = [np.array(val) for val in train_X]
        test_X = [np.array(val) for val in test_X]

        self.train_full = train_X
        self.test_full = test_X
        self.train_X = np.vstack(train_X)
        self.test_X = np.vstack(test_X)

        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32,
            shuffle=True, worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=100,
            shuffle=True, worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))
        return train_loader, test_loader

    def train(self, num_epochs, lr, bsz=32):
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(num_epochs): 
            for batch_idx, data in enumerate(self.train_loader):
                bx = data[0]
                by = data[1]

                self.net.train()
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
            data_loader = self.test_loader
        else:
            data_loader = self.train_loader
        with torch.no_grad():
            loss_fn = nn.CrossEntropyLoss(reduction='sum')
            num_correct = 0
            for (bx, by) in data_loader:
                output = self.net.forward(bx.float())
                pred = output.data.argmax(1)
                num_correct += pred.eq(by.data.view_as(pred)).sum().item()
        acc = num_correct / len(data_loader.dataset) * 100.
        return acc

    def compute_lp_dictionary(self, eps, lp, block=False, normalize_cols=True):
        idx = list(range(self.N_train))
        bsz = self.N_train
        dictionary = list()
        blocks = dict()
        bx = torch.from_numpy(self.train_X)
        by = torch.from_numpy(self.train_y)
        delta = torch.zeros_like(bx, requires_grad=True)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(self.net((bx + delta).float()), by)
        loss.backward()
        if lp == np.infty:
            bx_attack = (eps * delta.grad.detach().sign()).numpy()
        elif lp == 2:
            bx_attack = (eps * delta.grad.detach() / delta.grad.detach().norm()).numpy()
        else:
            print("Attack not supported!")
            sys.exit(1)
        dictionary = bx_attack 
        dictionary = dictionary.reshape((dictionary.shape[0], -1))
        for label in range(38):
            blocks[label] = dictionary[torch.nonzero(by == label).squeeze()]
        if block:
            return blocks
        if normalize_cols:
            return normalize(dictionary.T, axis=0)
        else:
            return dictionary.T

    def test_lp_attack(self, lp, batch_idx, eps, realizable=False):
        bx = self.test_X[batch_idx, :, :]
        by = self.test_y[batch_idx]
        bx = torch.from_numpy(bx)
        by = torch.from_numpy(by)

        if realizable:
            blocks = self.compute_lp_dictionary(eps=EPS, lp=lp, block=True)

            D = [np.zeros(blocks[i].shape[0]) for i in range(38)]
            for i in range(10):
                D[i][0] = 1./3.
                D[i][1] = 1./3.
                D[i][2] = 1./3.
            perturb = np.array([np.dot(blocks[i].T, D[i]) for i in range(38)])

            #neigh = NearestNeighbors(n_neighbors=k)
            #neigh.fit(self.train_X.reshape((self.train_X.shape[0], -1)))
            #dist, ind = neigh.kneighbors(bx.reshape((1, -1)))
            #set_trace()

            perturbation = perturb[by, :]
            bx = bx.reshape((bx.shape[0], -1)).numpy()
            adv = bx + perturbation
            return adv
        else:
            delta = torch.zeros_like(bx, requires_grad=True)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(self.net((bx + delta).float()), by)
            loss.backward()
            if lp == np.infty:
                bx_attack = (eps * delta.grad.detach().sign()).numpy()
            elif lp == 2:
                bx_attack = (eps * delta.grad.detach() / delta.grad.detach().norm()).numpy()
            return bx_attack

def compute_dictionary(train, normalize_cols=True):
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

# Format: data is a list of 38 subjects, each who have m_i x 192 x 168 images.
def parse_data(resize=True):
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

def get_block(x, bidx, mode):
    global SIGNAL_BIDXS, ATTACK_BIDXS, ATTACK_BIDXS_F
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
    elif mode == ATTACK_F:
        assert len(bidx) == 2
        pid, aid = bidx[0], bidx[1]
        final_id = pid*(aid+1)
        if len(x.shape) == 2:
            return x[:, ATTACK_BIDXS_F[final_id]:ATTACK_BIDXS_F[final_id+1]]
        else:
            return x[ATTACK_BIDXS_F[final_id]:ATTACK_BIDXS_F[final_id+1]]

def get_objective(concat_dict, c, train, attack_lens, obj_type, dict_in_obj=False):
    idx = 0
    objective = 0
    if obj_type == 'bssc':
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
    elif obj_type == 'hierarchical':
        for pid in range(len(train)):
            aid_blocks = list()
            for aid in range(len(attack_lens)): 
                aid_block = get_block(c, (pid, aid), ATTACK_F)
                aid_blocks.append(aid_block)
                objective += norm(aid_block, p=2)
            pid_block = get_block(c, pid, SIGNAL)
            i_term1 = hstack([pid_block] + aid_blocks)
            #objective += norm(i_term1, p=2)
    return objective

def full_lp_reconstruct(trainer, lp, dict_in_obj=True, finegrained=False):
    #test_idx = list(range(trainer.N_test))
    test_idx = list(range(100))
    test_adv = trainer.test_lp_attack(lp, test_idx, EPS, realizable=True)

    opts = list()
    epss = list()
    for i in range(test_adv.shape[0]):
        print("Example {}".format(i))
        x_adv = test_adv[i, :].reshape(-1)
        opt, eps = lp_reconstruct(trainer, x_adv, lp, 
            dict_in_obj=dict_in_obj, finegrained=finegrained)
        opts.append(opt)
        epss.append(eps)
        set_trace()
    pickle.dump(opts, open('outs/opts_{}.pkl'.format(lp), 'wb'))
    pickle.dump(epss, open('outs/epss_{}.pkl'.format(lp), 'wb'))

def lp_reconstruct(trainer, x_adv, lp, dict_in_obj=True, finegrained=False):
    #print("-------------------{}-{}---------------------".format(ex_idx, lp))
    l2_dict = trainer.compute_lp_dictionary(EPS, 2)
    linf_dict = trainer.compute_lp_dictionary(EPS, np.infty)
    attack_dict = np.hstack([l2_dict, linf_dict])

    train = trainer.train_full
    test = trainer.test_full
    global SIGNAL_BIDXS
    SIGNAL_BIDXS = [0]
    for i in range(len(train)):
        SIGNAL_BIDXS.append(SIGNAL_BIDXS[-1] + train[i].shape[0])
    signal_dict = compute_dictionary(train)

    concat_dict = np.hstack([signal_dict, attack_dict])

    s_len, l2_len, linf_len = signal_dict.shape[1], l2_dict.shape[1], linf_dict.shape[1]

    attack_lens = [l2_len, linf_len]
    # Attack comes after signal in concat_dict
    global ATTACK_BIDXS, ATTACK_BIDXS_F
    ATTACK_BIDXS_F = [s_len + i for i in SIGNAL_BIDXS]
    ATTACK_BIDXS_F += [s_len+l2_len + i for i in SIGNAL_BIDXS]
    ATTACK_BIDXS = [s_len, s_len+l2_len, s_len+l2_len+linf_len]

    thres = 0.1

    c = cp.Variable(s_len + sum(attack_lens))

    objective = get_objective(concat_dict, c, train, attack_lens, 'bssc')
    #objective = get_objective(concat_dict, c, train, attack_lens, 'hierarchical')
    
    minimizer = cp.Minimize(objective)
    constraints = [norm(x_adv - concat_dict@c, p=2) <= thres]

    prob = cp.Problem(minimizer, constraints)

    result = prob.solve(verbose=False, solver=cp.MOSEK)

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
    return opt, eps

    """
    ax = sns.heatmap(eps, annot=True, fmt="0.0f", annot_kws={"fontsize":8},
                yticklabels=range(1, eps.shape[0]+1), xticklabels=['L2', 'Linf'])
    ax.set_yticklabels(range(1, eps.shape[0]+1), size = 8)
    plt.ylabel('Person ID')
    plt.title("$\| x_{adv} - D_s[i]c_s[i] - D_a[j][i]c_a[j][i]\|_2$")
    plt.savefig('plots/fg_heatmap_success_linf.png')
    #plt.title("$\| x_{adv} - D_s[i]c_s[i] - D_a[j]c_a[j]\|_2$")
    #plt.savefig('plots/coarse_heatmap_success_linf.png')
    plt.show()

    perturb = 1./3 * linf_dict[:, 0] + 1./3 *linf_dict[:, 1] + 1./3*linf_dict[:, 2]
    signal_res = x_adv - \
        get_block(concat_dict, 0, SIGNAL) @ get_block(opt, 0, SIGNAL) - perturb
    res = x_adv - \
        get_block(concat_dict, 0, SIGNAL) @ get_block(opt, 0, SIGNAL) - \
        get_block(concat_dict, 1, ATTACK) @ get_block(opt, 1, ATTACK)
    set_trace()
    """

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

    l2_dict = trainer.compute_lp_dictionary(EPS, 2)
    linf_dict = trainer.compute_lp_dictionary(EPS, np.infty)

    l2_b = orth(l2_dict)
    linf_b = orth(linf_dict)
    l2_b2 = orth(clipped_l2)
    linf_b2 = orth(clipped_linf)

    _, c1, _ = svds(l2_b.T @ linf_b, k=1)
    _, c2, _ = svds(l2_b2.T @ linf_b2, k=1)

    print("Normal shape: L2: {}, Linf: {}".format(l2_dict.shape[1], linf_dict.shape[1]))
    print("Clipped shape: L2: {}, Linf: {}".format(clipped_l2.shape[1], clipped_linf.shape[1]))
    print("Rank of l2: {}. Rank of linf: {}".format(l2_b.shape[1], linf_b.shape[1]))
    print("Rank of l2 clipped: {}. Rank of linf clipped: {}".format(l2_b2.shape[1], linf_b2.shape[1]))
    print("Coherence all classes: {}. Coherence diff classes: {}".format(c1[0], c2[0]))

np.random.seed(0)
trainer = Trainer()
#trainer.train(20, 1e-2) 
trainer.net.load_state_dict(torch.load('pretrained_model.pth'))
test_acc = trainer.evaluate(test=True)
print("Loaded pretrained model!. Test accuracy: {}%".format(test_acc))
#lp_reconstruct(trainer, 0, 2, dict_in_obj=False)
#lp_reconstruct(trainer, 0, np.infty, dict_in_obj=False)
#lp_reconstruct(trainer, 10, 2, dict_in_obj=False)
#lp_reconstruct(trainer, 10, np.infty, dict_in_obj=False)
#lp_reconstruct(trainer, 20, 2)
#lp_reconstruct(trainer, 20, np.infty)

full_lp_reconstruct(trainer, 2, dict_in_obj=False)
#full_lp_reconstruct(trainer, np.infty, dict_in_obj=False)
#analyze(trainer, 2)
#analyze(trainer, np.infty)
