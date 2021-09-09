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
from skimage.transform import swirl
from torch.utils.data import DataLoader, Dataset
from advertorch.attacks import SparseL1DescentAttack, L2PGDAttack, LinfPGDAttack
from baselines import evaluate_baseline 
from kymatio.torch import Scattering2D
from trainer import Trainer
from irls import BlockSparseIRLSSolver
from active_set import BlockSparseActiveSetSolver
from multiprocessing import Pool

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
    acc = sum(pred == test_y) / test_y.shape[0] * 100.
    return acc

def epoch_adversarial(loader, lr_schedule, model, epoch_i, attack, criterion = nn.CrossEntropyLoss(), 
                        opt=None, device = "cuda:1", stop = False, stats = False, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0

    for i, batch in enumerate(loader):
        X,y = batch[0].to(device), batch[1].to(device)
        # if attack == pgd_all_old:
        #     delta = attack(model, X, y, device = device, **kwargs)
        #     delta = delta[0]
        # else:
        if stats:
            delta = attack(model, X, y, device = device, batchid=i, epoch_i = epoch_i , **kwargs)
        else:
            delta = attack(model, X, y, device = device, **kwargs)
        adv = X + delta

        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_loss += loss.item()*y.size(0)
        train_acc += (yp.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        if stop:
            break
        
    return train_loss / train_n, train_acc / train_n, adv

def sbsc(trainer, args, eps, test_lp, lp_variant, use_cnn_for_dict=False, test_adv=None):
    toolchain = args.toolchain
    eps_map = utils.EPS[args.dataset]
    #eps = eps_map[args.test_lp]

    attack_dicts = list()
    for attack in toolchain:
        if use_cnn_for_dict:
            attack_dicts.append(trainer.compute_lp_dictionary(eps_map[attack], attack, net='cnn'))
        else:
            attack_dicts.append(trainer.compute_lp_dictionary(eps_map[attack], attack))

    np.random.seed(0)
    #test_idx = list(range(100))
    test_idx = np.random.choice(list(range(trainer.N_test)), 100)
    #test_idx = list(range(trainer.N_test))
    test_x = trainer.test_X[test_idx, :, :]
    test_y = trainer.test_y[test_idx]
    Ds = trainer.compute_train_dictionary(trainer.train_full)
    raw_Da = np.hstack(attack_dicts)
    #pickle.dump(Ds, open('files/Ds.pkl', 'wb'))
    #pickle.dump(raw_Da, open('files/raw_Da.pkl', 'wb'))
    #Ds = pickle.load(open('files/Ds.pkl', 'rb'))
    #raw_Da = pickle.load(open('files/raw_Da.pkl', 'rb'))
    
    dst = trainer.dataset
    sz = utils.SIZE_MAP[dst]
    num_attacks = len(toolchain)
    
    if test_adv is None:
        test_adv = trainer.test_lp_attack(test_lp, test_x, test_y, eps, realizable=False, lp_variant=lp_variant)

    #real_test_adv = trainer.test_lp_attack(test_lp, test_x, test_y, 0.3, realizable=False, lp_variant=lp_variant)

    ####### MAINI CODE
    #trainer.load_model('Msd')
    #test_loader = DataLoader(trainer.test_dataset, batch_size = 100, shuffle=False)
    #device_id = 0
    #device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
    #torch.cuda.set_device(int(device_id))
    #total_loss, maini_acc, maini_adv = epoch_adversarial(test_loader, None, trainer.net, 0, utils.pgd_linf, device = device, stop = True, restarts = 10, num_iter = 100, epsilon=eps)

    
    #maini = next(iter(test_loader))
    #maini_x = maini[0].cuda()
    #maini_y = maini[1].cuda()
    #test_adv = trainer.test_lp_attack(test_lp, maini_x, maini_y, eps, realizable=False)
    #delta = utils.pgd_linf(trainer.net, maini_x, maini_y, restarts=10, num_iter=100, epsilon=eps)
    #red_adv =  trainer._lp_attack(test_lp, eps, maini_x, maini_y)
    #maini_adv = maini_adv.cpu().detach().numpy()
    #maini_y = maini_y.cpu().detach().numpy()
    ########
    #bx = trainer.test_X[test_idx, :, :]
    #set_trace()
    acc = trainer.evaluate(given_examples=(test_adv, test_y))
    print("[L{}, variant={}, eps={}] Adversarial accuracy: {}%".format(test_lp, lp_variant, eps, acc))

    #models = ['L1', 'L2', 'Linf', 'Max', 'Avg', 'Msd', 'Vanilla']
    #for model in models:
    #    acc = evaluate_baseline(model, (test_adv, maini_y))
    #    maini_acc = evaluate_baseline(model, (maini_adv, maini_y))
    #    print("{}: Defense Accuracy: {}%".format(model, acc))
    #    print("{}: Maini Defense Accuracy: {}%".format(model, maini_acc))
    #set_trace()
    #bacc = baseline(trainer, test_adv[:100, :], test_y[:100])
    #print("Baseline accuracy: {}%".format(bacc))
    #return

    class_preds = list()
    attack_preds = list()
    denoised = list()
    mismatch = 0
    num_examples = 10
    solvers = list()
    xs = list()
    for t in range(num_examples):
        if t % 5 == 0:
            print(t)
        raw_x = test_x[t, :]
        corrupted_x = test_adv[t, :]
        ty = torch.from_numpy(np.array(test_y[t:t+1]))
        print("ty = {}, lp = {}".format(ty, test_lp))
        if args.embedding == 'scattering':
            cheat_x = torch.from_numpy(test_x[t, :]).unsqueeze(0)
            tcorrupted_x = torch.from_numpy(corrupted_x).unsqueeze(0)
            if torch.cuda.is_available():
                cheat_x = cheat_x.cuda()
                tcorrupted_x = tcorrupted_x.cuda()
            traw_x = torch.from_numpy(raw_x[None, :])

            if args.realizable:
                x_Da = trainer._lp_attack(test_lp, eps, traw_x, ty, only_delta=True)
                x_Da = x_Da.reshape(-1)
                raw_Da[:, 2000*toolchain.index(test_lp) + 200*ty + 50] = x_Da
                Ds[:, 200*ty+50] = trainer.scattering(traw_x).cpu().detach().numpy().reshape(-1)

            start = time.time()
            try:
                grad = pickle.load(open('files/jacobians_{}/grad_{}_{}_{}.pkl'.format(args.dataset, test_lp, eps, t), 'rb'))
            except:
                grad = torch.autograd.functional.jacobian(trainer.scattering.scattering, tcorrupted_x, vectorize=True)
                grad = grad.reshape(-1, trainer.d)
                pickle.dump(grad, open('files/jacobians_{}/grad_{}_{}_{}.pkl'.format(args.dataset, test_lp, eps, t), 'wb'))
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
        elif args.embedding == 'warp':
            #swirled = swirl(image, rotation=0, strength=5, radius=120)    
            tcorrupted_x = torch.from_numpy(corrupted_x).unsqueeze(0)
            grad = torch.autograd.functional.jacobian(swirl, tcorrupted_x, vectorize=True)
            set_trace()
        else:
            Da = raw_Da
            Ds = normalize(Ds, axis=0)
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
    #print("Parallelizing over {} CPUs".format(multiprocessing.cpu_count() - 4))
    #p = Pool(processes=multiprocessing.cpu_count() - 4)
    results = list()
    #print("USING DS ALGORITHM")
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

            """
            corrupted_x = real_test_adv[t, :]
            tcorrupted_x = torch.from_numpy(corrupted_x).unsqueeze(0)
            if torch.cuda.is_available():
                tcorrupted_x = tcorrupted_x.cuda()
            x = trainer.scattering(tcorrupted_x).cpu().detach().numpy().reshape(-1)
            cs_est, Ds_est, class_pred, dn = results[i]
            grad = pickle.load(open('files/jacobians_{}/grad_{}_{}.pkl'.format(args.dataset, test_lp, i), 'rb'))
            set_trace()
            delta = trainer._lp_attack(test_lp, 0.3, torch.from_numpy(test_x[0:1, :]), torch.from_numpy(test_y[0:1]), only_delta=True)
            delta = delta.reshape(-1)
            lhs = x
            rhs = Ds_est @ cs_est + grad @ delta
            set_trace()
            """
        #futs = [p.apply_async(solvers[i].solve, args=([xs[i]])) for i in range(num_examples)]
        #results = [fut.get() for fut in futs]
    for res in results:
        cs_est, ca_est, Ds_est, Da_est, class_pred, attack_pred, dn, err_attack = res
        #cs_est, Ds_est, class_pred, dn = res
        #attack_pred = -1
        class_preds.append(class_pred)
        attack_preds.append(attack_pred)
        denoised.append(dn)

    class_preds = np.array(class_preds)
    attack_preds = np.array(attack_preds)
    print("Class preds: {}. Ground Truth: {}".format(class_preds, test_y[:num_examples]))
    print("Attack preds: {}. Ground Truth: {}".format(attack_preds, toolchain.index(test_lp)))
    denoised = np.array(denoised)
    signal_acc = np.sum(class_preds == test_y[:num_examples]) / float(num_examples) * 100.
    attack_acc = np.sum(attack_preds == toolchain.index(test_lp)) / float(num_examples) * 100.
    #if args.dataset == 'mnist':
    #    denoised = denoised.reshape((num_examples, 1, 28, 28))
    #denoised_acc = trainer.evaluate(given_examples=(denoised, test_y[:100]))

    print("Signal classification accuracy: {}%".format(signal_acc))
    print("Attack detection accuracy: {}%".format(attack_acc))
    #print("Mismatch: {}%".format(mismatch / float(num_examples) * 100.))
    #print("Denoised accuracy: {}%".format(denoised_acc))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='SBSC')
    parser = utils.get_parser(parser)
    args = parser.parse_args()

    main(args) 

## YaleB
## LAMBDA1: 0.2
## LAMBDA2: 0.1
