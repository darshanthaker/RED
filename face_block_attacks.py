import numpy as np
import cvxpy as cp
import sys
import os
import pickle
import re
import matplotlib.pyplot as plt

from PIL import Image
from cvxpy.atoms.norm import norm
from pdb import set_trace

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
def parse_data():
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
            im = np.array(Image.open(im_path).resize((35, 40)))
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
        m = int(0.7 * N)
        train[pid] = data[pid][:m, :, :]
        test[pid] = data[pid][m:, :, :]
    return train, test

def reconstruct():
    # Adversarial Dictionary: D x N
    D = 1400
    data = parse_data()
    train, test = split_train_test(data)

    dog = np.array(Image.open('dog.png').resize((15, 15)).convert('L'))
    mario = np.array(Image.open('mario.png').resize((15, 15)).convert('L'))
    x = test[0][0, :, :]
    x_adv = attack(x, dog, (0, 0)).reshape(-1)

    #dog_dict = compute_dictionary(train, dog, (0, 0))
    #mario_dict = compute_dictionary(train, mario, (15, 15))

    dog_dict = attack(np.zeros(x.shape), dog, (0, 0)).reshape((-1, 1))
    mario_dict = attack(np.zeros(x.shape), mario, (0, 0)).reshape((-1, 1))

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
    minimizer = cp.Minimize(objective)
    #constraints = [norm(x_adv - concat_dict@c, p=2) <= thres]
    constraints = [norm(x - signal_dict@c[:s_len], p=2) <= thres]

    prob = cp.Problem(minimizer, constraints)

    result = prob.solve(verbose=True, max_iters=10)
    set_trace()
    

np.random.seed(0)
reconstruct()
