import numpy as np
import cvxpy as cp
import sys
import os
import pickle
import time
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
import multiprocessing

from cvxpy.atoms.norm import norm
from cvxpy.atoms.affine.hstack import hstack
from neural_net import NN, CNN
from pdb import set_trace
from scipy.linalg import orth
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from advertorch.attacks import SparseL1DescentAttack, L2PGDAttack, LinfPGDAttack
from baselines import evaluate_baseline 
from kymatio.torch import Scattering2D
from trainer import Trainer
from irls import BlockSparseIRLSSolver
from active_set import BlockSparseActiveSetSolver

def sbsc(trainer, args, test_adv):
    toolchain = args.toolchain
    eps_map = utils.EPS[args.dataset]

    attack_dicts = list()
    for attack in toolchain:
        attack_dicts.append(trainer.compute_lp_dictionary(eps_map[attack], attack))

    Ds = trainer.compute_train_dictionary(trainer.train_full)
    raw_Da = np.hstack(attack_dicts)
    
    dst = trainer.dataset
    sz = utils.SIZE_MAP[dst]
    num_attacks = len(toolchain)
    
    err_attacks = list()
    num_examples = 10
    solvers = list()
    xs = list()
    for t in range(num_examples):
        if t % 5 == 0:
            print(t)
        #raw_x = test_x[t, :]
        corrupted_x = test_adv[t, :]
        #ty = torch.from_numpy(np.array(test_y[t:t+1]))
        #print("ty = {}, lp = {}".format(ty, test_lp))
        if args.embedding == 'scattering':
            cheat_x = torch.from_numpy(test_x[t, :]).unsqueeze(0)
            tcorrupted_x = torch.from_numpy(corrupted_x).unsqueeze(0)
            if torch.cuda.is_available():
                cheat_x = cheat_x.cuda()
                tcorrupted_x = tcorrupted_x.cuda()
            traw_x = torch.from_numpy(raw_x[None, :])
            x_Da = trainer._lp_attack(test_lp, eps, traw_x, ty, only_delta=True)
            x_Da = x_Da.reshape(-1)
            #raw_Da[:, 2000*toolchain.index(test_lp) + 200*ty + 50] = x_Da
            #Ds[:, 200*ty+50] = trainer.scattering(traw_x).cpu().detach().numpy().reshape(-1)
            start = time.time()
            try:
                grad = pickle.load(open('files/jacobians_{}/grad_{}_{}.pkl'.format(args.dataset, test_lp, t), 'rb'))
            except:
                grad = torch.autograd.functional.jacobian(trainer.scattering.scattering, tcorrupted_x, vectorize=True)
                grad = grad.reshape(-1, trainer.d)
                pickle.dump(grad, open('files/jacobians_{}/grad_{}_{}.pkl'.format(args.dataset, test_lp, t), 'wb'))
            try:
                cheat_grad = pickle.load(open('files/jacobians_{}/cheat_grad_{}.pkl'.format(args.dataset, t), 'rb'))
            except:
                cheat_grad = torch.autograd.functional.jacobian(trainer.scattering.scattering, cheat_x, vectorize=True)
                cheat_grad = cheat_grad.reshape(-1, trainer.d)
                pickle.dump(cheat_grad, open('files/jacobians_{}/cheat_grad_{}.pkl'.format(args.dataset, t), 'wb'))
            end = time.time()
            print("jacobian took {} seconds".format(end - start))

            if args.use_cheat_grad:
                print("USING CHEAT GRAD!")
                Da = cheat_grad.cpu() @ raw_Da
            else:
                Da = grad.cpu() @ raw_Da
            Da = Da.numpy()
            Ds = normalize(Ds, axis=0)
            Da = normalize(Da, axis=0)
            x = trainer.scattering(tcorrupted_x).cpu().detach().numpy().reshape(-1)
        else:
            Da = raw_Da
            Da = normalize(Da, axis=0)
            x = corrupted_x.reshape(-1)

        if args.solver == 'irls':
            solver = BlockSparseIRLSSolver(Ds, Da, trainer.num_classes, num_attacks, sz, 
                    lambda1=args.lambda1, lambda2=args.lambda2, del_threshold=args.del_threshold)
        else:
            solver = BlockSparseActiveSetSolver(Ds, Da, trainer.num_classes, num_attacks, sz, 
                    lambda1=args.lambda1, lambda2=args.lambda2)
        solvers.append(solver)
        xs.append(x)
    results = list()
    if args.solver == 'irls':
        #cs_est, ca_est, Ds_est, Da_est, err_cs, ws, wa, active_classes = solver.solve(x, alg=args.regularizer)
        #print("Active classes: {}".format(active_classes))
        # IRLS deprecated for now.
        assert False
    elif args.solver == 'en_full':
        futs = [p.apply_async(solvers[i].solve, args=([xs[i]]), kwds={'alg': 'en_full'}) for i in range(num_examples)]
        results = [fut.get() for fut in futs]
    elif args.solver == 'active_refined':
        for i in range(num_examples):
            if i % 5 == 0:
                print(i)
            results.append(solvers[i].solve(xs[i], alg='Ds+Da'))
    for res in results:
        cs_est, ca_est, Ds_est, Da_est, class_pred, attack_pred, dn, err_attack = res
        class_preds.append(class_pred)
    return class_preds

