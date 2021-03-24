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
import argparse

from cvxpy.atoms.norm import norm
from cvxpy.atoms.affine.hstack import hstack
from neural_net import NN, CNN
from pdb import set_trace
from scipy.linalg import orth
from scipy.sparse.linalg import svds
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from advertorch.attacks import SparseL1DescentAttack, L2PGDAttack, LinfPGDAttack
from baselines import evaluate_baseline 
from kymatio.torch import Scattering2D
from trainer import Trainer
from irls import BlockSparseIRLSSolver

def serialize_dictionaries(trainer, args):
    toolchain = args.toolchain
    attack_dicts = list()
    eps_map = utils.EPS[args.dataset]
    for attack in toolchain:
        attack_dicts.append(trainer.compute_lp_dictionary(eps_map[attack], attack))

    np.random.seed(0)
    test_idx = np.random.choice(list(range(trainer.N_test)), 324)
    Da = np.hstack(attack_dicts)
    Ds = trainer.compute_train_dictionary(trainer.train_full)
    dst = trainer.dataset
    sz = utils.SIZE_MAP[dst]
    if toolchain == [1, 2, np.infty]:
        ident = "all"
    elif toolchain == [2, np.infty]:
        ident = "2_inf"
    elif toolchain == [1, np.infty]:
        ident = "1_inf"
    
    scipy.io.savemat('mats/{}_ce/Ds_sub{}.mat'.format(dst, sz), {'data': Ds})
    scipy.io.savemat('mats/{}_ce/Da_sub{}_{}.mat'.format(dst, sz, ident), {'data': Da})
    scipy.io.savemat('mats/{}_ce/test_y.mat'.format(dst), {'data': trainer.test_y[test_idx]})

    all_attacks = list()
    for lp in toolchain:
        test_adv = trainer.test_lp_attack(lp, test_idx, eps_map[lp], realizable=False)
        all_attacks.append(test_adv)

        acc = trainer.evaluate(given_examples=(test_adv, trainer.test_y[test_idx]))
        print("[L{}, eps = {}] Adversarial accuracy: {}".format(lp, eps_map[lp], acc))
        if lp == np.infty:
            scipy.io.savemat('mats/{}_ce/linf_eps{}.mat'.format(dst, eps_map[lp]), {'data': test_adv})
        elif lp == 2:
            scipy.io.savemat('mats/{}_ce/l2_eps{}.mat'.format(dst, eps_map[lp]), {'data': test_adv})
        elif lp == 1:
            scipy.io.savemat('mats/{}_ce/l1_eps{}.mat'.format(dst, eps_map[lp]), {'data': test_adv})

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
            l = utils.SIZE_MAP[trainer.dataset]
            objective += norm(c[idx:idx+l], p=2)
            idx += l

        x_adv = test_adv[i, :].reshape(-1)
        minimizer = cp.Minimize(objective)
        constraints = [norm(x_adv - Ds@c, p=2) <= thres]
        prob = cp.Problem(minimizer, constraints)

        #result = prob.solve(verbose=True, solver=cp.MOSEK)
        result = prob.solve(verbose=False)
        opt = np.array(c.value)

        l = utils.SIZE_MAP[trainer.dataset]
        res = list()
        for pid in range(trainer.num_classes):
            res.append(np.linalg.norm(x_adv - Ds[:, l*pid:l*(pid + 1)]@opt[l*pid:l*(pid + 1)]))
        pred.append(np.argmin(res))
    pred = np.array(pred)
    acc = sum(pred == test_y)
    return acc

def sbsc(trainer, args, eps, test_lp):
    toolchain = args.toolchain
    eps_map = utils.EPS[args.dataset]
    #eps = eps_map[args.test_lp]

    attack_dicts = list()
    for attack in toolchain:
        attack_dicts.append(trainer.compute_lp_dictionary(eps_map[attack], attack))

    np.random.seed(0)
    test_idx = list(range(100))
    #test_idx = np.random.choice(list(range(trainer.N_test)), 324)
    #test_idx = list(range(trainer.N_test))
    test_y = trainer.test_y[test_idx]
    Da = np.hstack(attack_dicts)
    Ds = trainer.compute_train_dictionary(trainer.train_full)
    dst = trainer.dataset
    sz = utils.SIZE_MAP[dst]
    num_attacks = len(toolchain)
    
    test_adv = trainer.test_lp_attack(test_lp, test_idx, eps, realizable=False)
    set_trace()
    #bx = trainer.test_X[test_idx, :, :]
    #set_trace()
    #acc = trainer.evaluate(given_examples=(test_adv, test_y))
    #print("[L{}, eps={}] Adversarial accuracy: {}%".format(test_lp, eps, acc))

    models = ['L1', 'L2', 'Linf', 'Max', 'Avg', 'Msd', 'Vanilla']
    for model in models:
        acc = evaluate_baseline(model, (test_adv[:, :], test_y[:]))
        print("{}: Defense Accuracy: {}%".format(model, acc))
    set_trace()
    #bacc = baseline(trainer, test_adv[:100, :], test_y[:100])
    #print("Baseline accuracy: {}%".format(bacc))
    #return

    solver = BlockSparseIRLSSolver(Ds, Da, trainer.num_classes, num_attacks, sz, 
            lambda1=args.lambda1, lambda2=args.lambda2, del_threshold=args.del_threshold)
    class_preds = list()
    attack_preds = list()
    denoised = list()
    for t in range(100):
        if t % 5 == 0:
            print(t)
        x = test_adv[t, :]
        x = x.reshape(-1)
        cs_est, ca_est, Ds_est, Da_est, err_cs, ws, wa, active_classes = solver.solve(x, alg=args.regularizer)

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
        #Da_blk = solver.hier_bi.get_block(Da_est, (i_star, j_star)) 
        #ca_blk = solver.hier_bi.get_block(ca_est, (i_star, j_star)) 
        Ds_blk = solver.sig_bi.get_block(Ds_est, i_star)
        cs_blk = solver.sig_bi.get_block(cs_est, i_star)
        denoised.append(Ds_blk@cs_blk)
        #denoised.append(x - Da_blk@ca_blk)
    class_preds = np.array(class_preds)
    attack_preds = np.array(attack_preds)
    denoised = np.array(denoised)
    signal_acc = np.sum(class_preds == test_y[:100])
    attack_acc = np.sum(attack_preds == toolchain.index(test_lp))
    if args.dataset == 'mnist':
        denoised = denoised.reshape((100, 1, 28, 28))
    denoised_acc = trainer.evaluate(given_examples=(denoised, test_y[:100]))

    print("Signal classification accuracy: {}%".format(signal_acc))
    print("Attack detection accuracy: {}%".format(attack_acc))
    print("Denoised accuracy: {}%".format(denoised_acc))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='SBSC')
    parser = utils.get_parser(parser)
    args = parser.parse_args()

    main(args) 

## YaleB
## LAMBDA1: 0.2
## LAMBDA2: 0.1
