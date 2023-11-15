import matplotlib
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
import json
import spams

from cvxpy.atoms.norm import norm
from cvxpy.atoms.affine.hstack import hstack
from neural_net import NN, CNN
from pdb import set_trace
from scipy.linalg import orth
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.linear_model import OrthogonalMatchingPursuit
from torchvision import transforms
from skimage.transform import swirl
from torch.utils.data import DataLoader, Dataset
from advertorch.attacks import SparseL1DescentAttack, L2PGDAttack, LinfPGDAttack
from baselines import evaluate_baseline 
from kymatio.torch import Scattering2D
from trainer import Trainer
from irls import BlockSparseIRLSSolver
from active_set import BlockSparseActiveSetSolver
from prox_solver import ProxSolver
from prox_gan_solver import ProxGanSolver
from clip_encoder import CLIPEncoder
from PIL import Image
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

def clip_omp(trainer, train_x, train_y):
    clip_enc = CLIPEncoder()

    f = open('files/cifar10_CBM.txt', 'rb')
    concepts = list()
    for line in f:
        concepts.append(line.rstrip().decode())

    Ds_blocks = list()
    for idx in range(0, len(concepts), 100):
        end_idx = min(idx + 100, len(concepts))
        Ds_blocks.append(clip_enc.encode_text(concepts[idx:end_idx]).cpu().detach().numpy())
    Ds = np.concatenate(Ds_blocks).T
    Ds = normalize(Ds, axis=0)

    im = Image.fromarray((train_x[-1, :].transpose(1, 2, 0) * 255).astype(np.uint8))
    x = clip_enc.encode_image(im).squeeze().cpu().detach().numpy()

    class_coefs = list()
    concept_dict = dict()
    for lab in range(10):
        raw_X = train_x[np.where(train_y == lab)]
        enc_X = list()
        for i in range(raw_X.shape[0]):
            im = Image.fromarray((raw_X[i, :].transpose(1, 2, 0) * 255).astype(np.uint8))
            x = clip_enc.encode_image(im).squeeze().cpu().detach().numpy()
            enc_X.append(x)
        X = np.asfortranarray(np.vstack(enc_X).T, dtype=np.float32)
        Ds = np.asfortranarray(Ds, dtype=np.float32)
        coef = spams.omp(X, Ds, L=20, eps=0.1).toarray()
        class_coefs.append(np.mean(coef, axis=1))
        print("Finished {}".format(lab))
    for (lab, coef) in enumerate(class_coefs):
        nz_idx = coef.nonzero()[0]
        vals = sorted(zip(nz_idx, coef[nz_idx]), key=lambda pair: -pair[1])
        vals = list(filter(lambda x: x[1] >= 0, vals))
        filtered_concepts = [concepts[idx] for (idx, _) in vals][:20]
        concept_dict[utils.LABEL_MAP['cifar'][lab]] = filtered_concepts
        print("{}: {}".format(utils.LABEL_MAP['cifar'][lab], filtered_concepts))
    json_object = json.dumps(concept_dict, indent=4)
    with open ('files/cifar10_omp_concepts.json', 'w') as of:
        of.write(json_object)


def clip_ood(trainer, train_x, train_y, test_adv, test_y):
    clip_enc = CLIPEncoder()
    concepts = json.load(open('files/cifar10_omp_concepts.json', 'rb'))
    Ds_blocks = list()
    concept_lst = list()
    block_idxs = [0]
    for label_name in utils.LABEL_MAP['cifar']:
        tmp = list()
        concept_lst += concepts[label_name]
        for idx in range(0, len(concepts[label_name]), 100):
            end_idx = min(idx + 100, len(concepts[label_name]))
            tmp.append(clip_enc.encode_text(concepts[label_name][idx:end_idx]).cpu().detach().numpy())
        
        Ds_blocks.append(np.concatenate(tmp))
        block_idxs.append(block_idxs[-1] + Ds_blocks[-1].shape[0])
    Ds = np.concatenate(Ds_blocks).T
    Ds = normalize(Ds, axis=0)

    s_len = Ds.shape[1]

    class_coefs = list()
    concept_dict = dict()
    for lab in range(0):
        raw_X = train_x[np.where(train_y == lab)]
        enc_X = list()
        for i in range(raw_X.shape[0]):
            im = Image.fromarray((raw_X[i, :].transpose(1, 2, 0) * 255).astype(np.uint8))
            x = clip_enc.encode_image(im).squeeze().cpu().detach().numpy()
            enc_X.append(x)
        X = np.asfortranarray(np.vstack(enc_X).T, dtype=np.float32)
        Ds = np.asfortranarray(Ds, dtype=np.float32)
        coef = spams.omp(X, Ds, L=20, eps=0.1).toarray()
        class_coefs.append(np.mean(coef, axis=1))
        print("Finished {}".format(lab))

    enc_X = list()
    ims = list()
    for i in range(100):
        im = Image.fromarray((test_adv[i, :].transpose(1, 2, 0) * 255).astype(np.uint8))
        ims.append(im)
        x = clip_enc.encode_image(im).squeeze().cpu().detach().numpy()
        enc_X.append(x)
    X = np.asfortranarray(np.vstack(enc_X).T, dtype=np.float32)
    Ds = np.asfortranarray(Ds, dtype=np.float32)
    coef = spams.omp(X, Ds, L=10, eps=0.1).toarray()
    
    coef_concepts = list()
    for i in range(100):
        opt = coef[:, i]
        idx = np.where(opt >= 0.001)[0]
        coef_concepts.append([concept_lst[j] for j in idx])

    # Amplify reddish-brown coat in the bird, which is label 2.
    mod_coef = np.copy(coef[:, 0])
    mod_coef[15] = 5
    mod_X_clip = Ds @ mod_coef
    recon_X_clip = Ds @ coef[:, 0]

    #raw_X = train_x[np.where(train_y == 2)]
    raw_X = train_x
    best_mod_sim = -np.float("inf")
    best_sim = -np.float("inf")
    nn_mod_im = None
    nn_im = None
    for i in range(raw_X.shape[0]):
        im = Image.fromarray((raw_X[i, :].transpose(1, 2, 0) * 255).astype(np.uint8))
        x = clip_enc.encode_image(im).squeeze().cpu().detach().numpy()
        mod_sim = (x @ mod_X_clip) / (np.linalg.norm(x) * np.linalg.norm(mod_X_clip))
        sim = (x @ recon_X_clip) / (np.linalg.norm(x) * np.linalg.norm(recon_X_clip))
        if mod_sim > best_mod_sim:
            best_mod_sim = mod_sim
            nn_mod_im = im 
        if sim > best_sim:
            best_sim = sim
            nn_im = im

    im = Image.fromarray((test_adv[0, :].transpose(1, 2, 0) * 255).astype(np.uint8))
    x = clip_enc.encode_image(im).squeeze().cpu().detach().numpy()
    sim = (x @ recon_X_clip) / (np.linalg.norm(x) * np.linalg.norm(recon_X_clip))
    set_trace()

def clip_bsc(trainer, test_adv, test_y):
    clip_enc = CLIPEncoder() 

    #concepts = json.load(open('files/gpt3_cifar10_important.json', 'rb'))
    concepts = json.load(open('files/cifar10_omp_concepts.json', 'rb'))
    #concepts = json.load(open('files/cifar10_labo.json', 'rb'))
    Ds_blocks = list()
    concept_lst = list()
    block_idxs = [0]
    for label_name in utils.LABEL_MAP['cifar']:
        tmp = list()
        concept_lst += concepts[label_name]
        for idx in range(0, len(concepts[label_name]), 100):
            end_idx = min(idx + 100, len(concepts[label_name]))
            tmp.append(clip_enc.encode_text(concepts[label_name][idx:end_idx]).cpu().detach().numpy())
        
        Ds_blocks.append(np.concatenate(tmp))
        block_idxs.append(block_idxs[-1] + Ds_blocks[-1].shape[0])
    Ds = np.concatenate(Ds_blocks).T

    s_len = Ds.shape[1]

    thres = 10
    pred = list()
    test_y = test_y[:10]
    for i in range(10):
        if i % 10 == 0:
            print(i)

        im = Image.fromarray((test_adv[i, :].transpose(1, 2, 0) * 255).astype(np.uint8))
        x_adv = clip_enc.encode_image(im).squeeze().cpu().detach().numpy()

        c = cp.Variable(s_len)
        idx = 0
        objective = 0
        lam = 0.7
        for pid in range(trainer.num_classes):
            l = Ds_blocks[pid].shape[0]
            objective += lam * norm(c[idx:idx+l], p=2)
            idx += l

        objective += norm(x_adv - Ds@c, p=2)

        minimizer = cp.Minimize(objective)
        #constraints = [norm(x_adv - Ds@c, p=2) <= thres]
        prob = cp.Problem(minimizer, None)

        #result = prob.solve(verbose=True, solver=cp.MOSEK)
        result = prob.solve(verbose=True)
        opt = np.array(c.value)

        idx = np.where(opt >= 0.001)[0]
        print("Concepts: {}".format([concept_lst[j] for j in idx]))
        plt.stem(range(len(opt)), opt)
        plt.show()
        set_trace()

        res = list()
        for pid in range(trainer.num_classes):
            l = Ds_blocks[pid].shape[0]
            try:
                res.append(np.linalg.norm(x_adv - Ds[:, block_idxs[pid]:block_idxs[pid + 1]]@opt[block_idxs[pid]:block_idxs[(pid + 1)]]))
            except:
                set_trace()
        pred.append(np.argmin(res))
        if pred[-1] != test_y[i]:
            set_trace()
    pred = np.array(pred)
    acc = sum(pred == test_y) / test_y.shape[0] * 100.
    print("Accuracy: {}".format(acc))
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

def sbsc(trainer, args, eps, test_lp, lp_variant, use_cnn_for_dict=False, test_adv=None, use_pca=False, use_gan_Ds=False):
    toolchain = args.toolchain
    eps_map = utils.EPS[args.dataset]
    #eps = 0

    #print("UNATTACKED EXAMPLES TAKE NOTE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    """
    attack_dicts = list()
    for attack in toolchain:
        if use_cnn_for_dict:
            attack_dicts.append(trainer.compute_lp_dictionary(eps_map[attack], attack, net='cnn'))
        else:
            attack_dicts.append(trainer.compute_lp_dictionary(eps_map[attack], attack))
    """

    np.random.seed(0)
    #raw_Ds = trainer.compute_train_dictionary()
    #if trainer.embedding == 'warp':
    #    Ds = trainer.warp(raw_Ds)
    #elif trainer.embedding == 'scattering':
    #    Ds = raw_Ds
    #else:
    #    Ds = raw_Ds
    #raw_Da = np.hstack(attack_dicts)
    #pickle.dump(attack_dicts[0], open('files/Da_l1_{}_20.pkl'.format(args.dataset), 'wb'))
    #pickle.dump(attack_dicts[1], open('files/Da_l2_{}_20.pkl'.format(args.dataset), 'wb'))
    #@pickle.dump(attack_dicts[2], open('files/Da_linf_{}_20.pkl'.format(args.dataset), 'wb'))
    #Da = np.hstack(attack_dicts)
    #pickle.dump(Da, open('files/Da_{}_20.pkl'.format(args.dataset), 'wb'))
    #if use_gan_Ds:
    #    Ds = pickle.load(open('files/Ds_{}_inf.pkl'.format(args.dataset), 'rb'))
    Ds = pickle.load(open('files/raw_Ds_cifar_inf.pkl', 'rb'))
    attack_dicts = list()
    if args.dataset == 'tiny_imagenet':
        attack_dicts.append(pickle.load(open('files/Da_l1_{}_20.pkl'.format(args.dataset), 'rb')))
        attack_dicts.append(pickle.load(open('files/Da_l2_{}_20.pkl'.format(args.dataset), 'rb')))
        attack_dicts.append(pickle.load(open('files/Da_linf_{}_20.pkl'.format(args.dataset), 'rb')))
        Da = np.hstack(attack_dicts)
    else:
        Da = pickle.load(open('files/Da_{}.pkl'.format(args.dataset), 'rb'))
        #Da = pickle.load(open('files/Da_{}_400.pkl'.format(args.dataset), 'rb'))
    #Da = pickle.load(open('files/Da_{}.pkl'.format(args.dataset), 'rb'))
    dst = trainer.dataset
    sz = utils.SIZE_MAP[dst]
    """
    from block_indexer import BlockIndexer
    hier_bi = BlockIndexer(200, (10, 3))
    new_hier_bi = BlockIndexer(10, (10, 3))
    Da_sub = np.zeros((Da.shape[0], 10*10*3))
    for i in range(hier_bi.num_blocks[0]):
        for j in range(hier_bi.num_blocks[1]):
            Da_ij = hier_bi.get_block(Da, (i, j))
            Da_sub = new_hier_bi.set_block(Da_sub, (i, j), Da_ij[:, :10])
    #corr = Da_sub.T @ Da_sub
    #corr = Da.T @ Da
    #corr = np.abs(corr)
    #corr = np.clip(corr, 0, 1) # For numerical issues.
    #angles = np.arccos(corr)
    #ax = sns.heatmap(angles)
    #ax.set_xticklabels(range(1, 300, 50), size = 8)
    #ax.set_yticklabels(range(1, 300, 50), size = 8)
    #plt.ylabel('Person ID')
    #plt.title("$\| x_{adv} - D_s[i]c_s[i] - D_ac_a\|_2$")
    #plt.show()
    #set_trace()
    """

    if args.make_realizable:
        test_x = list()
        test_y = list()
        num_s = 0
        for i in range(100):
            cs_star = np.zeros(Ds.shape[1], dtype=np.float32)
            #num_s = np.random.randint(25)
            cs_star[num_s:num_s+1] = 1.0
            num_s += 1
            test_x.append(raw_Ds @ cs_star)
            test_y.append(0)
        test_x = np.vstack(test_x)
        test_y = np.array(test_y)
    else:
        test_idx = np.random.choice(list(range(trainer.N_test)), 100)
        test_x = trainer.test_X[test_idx, :]
        test_y = trainer.test_y[test_idx]
        #print("REALIZABLE CASE!!")
        #test_idx = np.random.choice(list(range(trainer.N_train)), 100)
        #test_x = trainer.train_X[test_idx, :]
        #test_y = trainer.train_y[test_idx]

    dst = trainer.dataset
    sz = utils.SIZE_MAP[dst]
    num_attacks = len(toolchain)
    
    #test_adv = pickle.load(open('files/test_adv_{}_{}.pkl'.format(args.dataset, test_lp), 'rb'))[:100]
    if test_adv is None:
        test_adv = trainer.test_lp_attack(test_lp, test_x, test_y, eps, realizable=False, lp_variant=lp_variant)
        #delta = trainer.test_lp_attack(test_lp, test_x, test_y, eps, realizable=False, lp_variant=lp_variant, only_delta=True)

    pickle.dump(test_adv, open('files/test_adv_{}_{}.pkl'.format(args.dataset, test_lp), 'wb'))
    pickle.dump(test_x, open('files/test_x_{}.pkl'.format(args.dataset), 'wb'))
    pickle.dump(test_y, open('files/test_y_{}.pkl'.format(args.dataset), 'wb'))

    acc = trainer.evaluate(given_examples=(test_adv, test_y), topk=True)
    print("[L{}, variant={}, eps={}] Adversarial accuracy: {}%".format(test_lp, lp_variant, eps, acc))

    #acc = clip_bsc(trainer, test_adv, test_y)
    #acc = clip_ood(trainer, trainer.train_X, trainer.train_y, test_adv, test_y)
    #acc = clip_omp(trainer, trainer.train_X, trainer.train_y)
    #set_trace()

    class_preds = list()
    attack_preds = list()
    denoised = list()
    mismatch = 0
    num_examples = 100
    solvers = list()
    xs = list()
    for t in range(num_examples):
        if t % 5 == 0:
            print(t)
        raw_x = test_x[t, :]
        corrupted_x = test_adv[t, :]
        ty = torch.from_numpy(np.array(test_y[t:t+1]))
        print("ty = {}, lp = {}".format(ty, test_lp))
        #print("not normalizing Ds!!!")
        Ds = normalize(Ds, axis=0)
        Da = normalize(Da, axis=0)
        x = corrupted_x.reshape(-1)
        #pickle.dump(raw_x, open('decoder_outs_adv_linf_cifar/{}/raw_x.pkl'.format(t), 'wb'))
        #continue
        #print("USING RAW X INSTEAD OF CORRUPTED!")
        #x = raw_x.reshape(-1)

        if args.solver == 'irls':
            solver = BlockSparseIRLSSolver(Ds, Da, trainer.num_classes, num_attacks, sz, 
                    lambda1=args.lambda1, lambda2=args.lambda2, del_threshold=args.del_threshold)
        elif args.solver == 'active_refined':
            solver = BlockSparseActiveSetSolver(Ds, Da, trainer.decoder, trainer.num_classes, num_attacks, sz, 
                    lambda1=args.lambda1, lambda2=args.lambda2)
        elif args.solver == 'prox':
            if args.dataset == 'cifar':
                input_shape = (3, 32, 32)
            else:
                input_shape = trainer.input_shape
            solver = ProxSolver(Ds, Da, trainer.decoder, trainer.num_classes, num_attacks, 
                    sz, input_shape, lambda1=args.lambda1, lambda2=args.lambda2)
        elif args.solver == 'prox_gan':
            if args.dataset == 'cifar':
                input_shape = (3, 32, 32)
            elif args.dataset == 'tiny_imagenet':
                input_shape = (3, 64, 64)
            else:
                input_shape = (1, 28, 28)
            #solver = ProxGanSolver(Da, trainer.decoder, trainer.num_classes, num_attacks,
                    #sz, input_shape)
            solver = None
        solvers.append(solver)

        if args.dataset == 'mnist':
            xs.append(2*x - 1)
        else:
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
        #futs = [p.apply_async(solvers[i].solve, args=([xs[i]])) for i in range(num_examples)]
        #results = [fut.get() for fut in futs]
    elif args.solver == 'prox':
        if args.embedding is None:
            eta_s = 1.0 / np.linalg.norm(Ds @ Ds.T, ord=2)
            eta_a = 1.0 / np.linalg.norm(Da @ Da.T, ord=2)
        else:
            eta_s = None
            eta_a = None
        print("Lambda_s: {}. Lambda_a: {}".format(args.lambda1, args.lambda2))
        for i in range(num_examples):
            if i % 5 == 0:
                print(i)
            if test_lp == float("inf"):
                str_lp = "inf"
            else:
                str_lp = str(int(test_lp))
            if not os.path.exists('decoder_outs_adv_l{}/{}'.format(str_lp, i)):
                os.mkdir('decoder_outs_adv_l{}/{}'.format(str_lp, i))
            else:
                os.system('rm -rf decoder_outs_adv_l{}/{}'.format(str_lp, i))
                #os.rmdir('decoder_outs_adv_l{}/{}'.format(str_lp, i))
                os.mkdir('decoder_outs_adv_l{}/{}'.format(str_lp, i))
            #results.append(solvers[i].solve(xs[i], eta_s=eta_s, eta_a=eta_a, cheat_y=test_y[i], dir_name='decoder_outs_cheat/{}'.format(i)))
            results.append(solvers[i].solve(xs[i], eta_s=eta_s, eta_a=eta_a, dir_name='decoder_outs_adv_l{}/{}'.format(str_lp, i)))
        #futs = [p.apply_async(solvers[i].solve(xs[i], eta_s=eta_s, eta_a=eta_a)) for i in range(num_examples)]
        #results = [fut.get() for fut in futs]
            print("GROUND TRUTH LABEL: {}".format(test_y[i]))
    elif args.solver == 'prox_gan':
        for i in range(num_examples):
            if i % 5 == 0:
                print(i)
            if test_lp == float("inf"):
                str_lp = "inf"
            else:
                str_lp = str(int(test_lp))
            if not os.path.exists('decoder_outs_adv_l{}_{}/{}'.format(str_lp, args.dataset, i)):
                os.mkdir('decoder_outs_adv_l{}_{}/{}'.format(str_lp, args.dataset, i))
            else:
                os.system('rm -rf decoder_outs_adv_l{}_{}/{}'.format(str_lp, args.dataset, i))
                os.mkdir('decoder_outs_adv_l{}_{}/{}'.format(str_lp, args.dataset, i))
            solver = ProxGanSolver(Da, trainer.decoder, trainer.num_classes, num_attacks,
                    sz, input_shape)
            results.append(solver.solve(xs[i], dir_name='decoder_outs_adv_l{}_{}/{}'.format(str_lp, args.dataset, i), y=test_y[i]))
            print("GROUND TRUTH LABEL: {}".format(test_y[i]))

    for (res_idx, res) in enumerate(results):
        if args.solver == 'active_refined':
            cs_est, ca_est, Ds_est, Da_est, class_pred, attack_pred, dn, err_attack = res
        elif args.solver == 'prox':
            cs_est, ca_est, class_pred, attack_pred, dn = res
        elif args.solver == 'prox_gan':
            cs_est, attack_pred, dn = res
            
        if args.solver != 'prox_gan':
            class_preds.append(class_pred)
        attack_preds.append(attack_pred)
        denoised.append(dn)

    if args.solver != 'prox_gan':
        class_preds = np.array(class_preds)
        print("Class preds: {}. Ground Truth: {}".format(class_preds, test_y[:num_examples]))
        signal_acc = np.sum(class_preds == test_y[:num_examples]) / float(num_examples) * 100.
        print("Signal classification accuracy: {}%".format(signal_acc))

    attack_preds = np.array(attack_preds)
    print("Attack preds: {}. Ground Truth: {}".format(attack_preds, toolchain.index(test_lp)))
    denoised = np.array(denoised)
    attack_acc = np.sum(attack_preds == toolchain.index(test_lp)) / float(num_examples) * 100.
    #if args.dataset == 'mnist':
    #    denoised = denoised.reshape((num_examples, 1, 28, 28))
    denoised_acc = trainer.evaluate(given_examples=(denoised, test_y[:num_examples]), topk=True)

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
