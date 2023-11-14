import numpy as np
import matplotlib.pyplot as plt
import argparse
import utils
import sys
import os
import pickle
import torch
import torchvision
import torch.nn as nn
import copy
import sbsc

torch.set_num_threads(1)

from pdb import set_trace
from train_embedding import EmbeddingTrainer
from trainer import Trainer
from train_embedding import EmbeddingTrainer
from irls import BlockSparseIRLSSolver

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

def eps_plot():
    epss = [0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
    atts = ['l2', 'l1', 'linf']
    att_map = {'l2': 2, 'linf': np.inf}
    result_map = {'l2': {'signal': [], 'att': [], 'test': [], 'backtest': [], 'baseline': []}, 
                  'l1': {'signal': [], 'att': [], 'test': [], 'backtest': [], 'baseline': []}, 
                  'linf': {'signal': [], 'att': [], 'test': [], 'backtest': [], 'baseline': []}}

    result_map['linf']['eps'] = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    result_map['linf']['signal'] = pickle.load(open('active_outs/signal_accs_inf_mnist.pkl', 'rb'))
    result_map['linf']['att'] = pickle.load(open('active_outs/attack_accs_inf_mnist.pkl', 'rb'))
    result_map['linf']['test'] = pickle.load(open('active_outs/adv_accs_inf_mnist.pkl', 'rb'))
    result_map['linf']['backtest'] = pickle.load(open('active_outs/denoised_accs_inf_mnist.pkl', 'rb'))
    result_map['linf']['baseline'] = [92, 84, 82, 73, 72, 66, 54]
    result_map['l1']['eps'] = [0, 1.5, 3, 5, 6.5, 8, 10]
    result_map['l1']['signal'] = pickle.load(open('active_outs/signal_accs_1.0_mnist.pkl', 'rb'))
    result_map['l1']['att'] = pickle.load(open('active_outs/attack_accs_1.0_mnist.pkl', 'rb'))
    result_map['l1']['test'] = pickle.load(open('active_outs/adv_accs_1.0_mnist.pkl', 'rb'))
    result_map['l1']['backtest'] = pickle.load(open('active_outs/denoised_accs_1.0_mnist.pkl', 'rb'))
    result_map['l1']['baseline'] = [92, 87, 84, 83, 81, 78, 75]
    result_map['l2']['eps'] =[0, 0.3, 0.6, 1, 1.3, 1.6, 2]
    result_map['l2']['signal'] = pickle.load(open('active_outs/signal_accs_2.0_mnist.pkl', 'rb'))
    result_map['l2']['att'] = pickle.load(open('active_outs/attack_accs_2.0_mnist.pkl', 'rb'))
    result_map['l2']['test'] = pickle.load(open('active_outs/adv_accs_2.0_mnist.pkl', 'rb'))
    result_map['l2']['backtest'] = pickle.load(open('active_outs/denoised_accs_2.0_mnist.pkl', 'rb'))
    result_map['l2']['baseline'] = [92, 89, 85, 84, 81, 79, 76]
    set_trace()

    ## MNIST
    """
    result_map = {'linf': {'eps': [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], 
                           'signal': [94, 89, 85, 82, 73, 63, 56],
                           'att': [24, 50, 88, 93, 92, 91, 86],
                           'test': [98.99, 93.21, 62.96, 22.53, 5.55, 2.5, 2.5],
                           'backtest': [94, 89, 85, 81, 72, 59, 53],
                           'baseline': [92, 84, 82, 73, 72, 66, 54]},
                  'l2': {'eps': [0, 0.3, 0.6, 1, 1.3, 1.6, 2], 
                           'signal': [94, 93, 89, 86, 87, 85, 81],
                           'att': [34, 35, 39, 43, 45, 42, 41],
                           'test': [98.99, 97.53, 92.90, 74.38, 55.86, 50, 44.75],
                           'backtest': [94, 93, 89, 86, 87, 85, 81],
                           'baseline': [92, 89, 85, 84, 81, 79, 76]},
                  'l1': {'eps': [0, 1.5, 3, 5, 6.5, 8, 10], 
                           'signal': [94, 93, 90, 85, 85, 85, 82],
                           'att': [44, 44, 47, 52, 55, 59, 66],
                           'test': [98.99, 98.45, 95.68, 90.12, 79.63, 63.88, 41.97],
                           'backtest': [94, 93, 90, 85, 85, 85, 82],
                           'baseline': [92, 87, 84, 83, 81, 78, 75]}}

    result_map = {'linf': {'eps': [0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1], 
                           'signal': [95, 94, 93, 92, 87, 86, 81],
                           'att': [40, 77, 95, 98, 99, 99],
                           'test': [97.53, 63.88, 16.05, 0.61, 0.61, 0],
                           'backtest': [96, 66, 16, 0, 0, 0],
                           'baseline': []},
                  'l2': {'eps': [0.0, 0.75, 1.5, 2.5, 3.25, 4.0, 5.0], 
                           'signal': [95, 92, 86, 79, 61, 49, 32],
                           'att': [32, 20, 20, 29, 22, 24, 17],
                           'test': [97.53, 4.01, 0.61, 0.61, 0.3, 0.3, 0],
                           'backtest': [96, 4, 0, 0, 0, 0, 0],
                           'baseline': []},
                  'l1': {'eps': [0.0, 2.25, 4.5, 7.5, 9.75, 12, 15], 
                           'signal': [95, 94, 93, 93, 91, 91, 87],
                           'att': [4, 62, 88, 93, 95, 98, 87],
                           'test': [97.53, 83.024, 50.30, 14.50, 1.85, 0.61],
                           'backtest': [30, 82, 48, 16, 6, 3, 0],
                           'baseline': []}}
    """


    baseline_map = {'linf': {'name': ['M1', 'M2', 'Minf', 'MAX', 'AVG', 'MSD'],
                             'eps_t': [0.287, 0.3, 0.3, 0.3, 0.3, 0.282], 
                             'eps': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3], 
                             'acc': [0, 0.4, 90.3, 51, 65.2, 62.7]},
                    'l2': {'name': ['M1', 'M2', 'Minf', 'MAX', 'AVG', 'MSD'],
                             'eps_t': [2, 1.9, 2, 2, 1.9, 2],
                             'eps': [2, 2, 2, 2, 2, 2],
                             'acc': [38.7, 70.2, 66.8, 64.1, 66.9, 70.2]},
                    'l1': {'name': ['M1', 'M2', 'Minf', 'MAX', 'AVG', 'MSD'],
                             'eps_t': [10, 10, 9.5, 10, 10, 10],
                             'eps': [10, 10, 10, 10, 10, 10],
                             'acc': [74.6, 51.1, 61.8, 61.2, 66.5, 70.4]}}
                
    #np.random.seed(0)
    #test_idx = np.random.choice(list(range(trainer.N_test)), 324)[:100]
    #test_y = scipy.io.loadmat('mats/mnist_ce/test_y.mat')['data'][0, :100]
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

    #result_map = pickle.load(open('mats/mnist_recon/result_map.pkl', 'rb'))

    fig, axes = plt.subplots(1, 3, figsize=(6.4*3, 4.8))
    dual_axes = list()
    #attacks = ['linf']
    attacks = ['l1', 'l2', 'linf']
    for (ax1, att) in zip(axes, attacks):
        #fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        dual_axes.append(ax2)
        ax1.plot(result_map[att]['eps'], result_map[att]['signal'], label='SBSC', marker='*', markersize=7)
        ax2.plot(result_map[att]['eps'], result_map[att]['att'], label='SBSAD', linestyle='--', marker='s', color='black', markerfacecolor='none')
        ax1.plot(result_map[att]['eps'], result_map[att]['test'], label='No Defense', marker='*', markersize=7)
        ax1.plot(result_map[att]['eps'], result_map[att]['backtest'], label='SBSC+CNN', marker='*', markersize=7)
        ax1.plot(result_map[att]['eps'], result_map[att]['baseline'], label='BSC', marker='*', markersize=7)
        #bm = baseline_map[att]
        #for (name, eps, acc) in zip(bm['name'], bm['eps'], bm['acc']):
        #    ax1.scatter(eps, acc, label=name, marker='x')
        #ax1.legend(prop={'size': 9})
        #ax2.legend(prop={'size': 9}, handlelength=3)
        ax1.grid()
        ax1.set_yticks(np.arange(0, 101, 10))
        ax2.set_yticks(np.arange(0, 101, 10))
        ax1.set_xlabel('{} Epsilon'.format(att))
    #fig.set_xlabel('Epsilon of Perturbation')
    #fig.set_ylabel('Signal Classification Accuracy')
    #fig.set_ylabel('Attack Detection Accuracy')
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left')
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    axes[0].set_ylabel('Signal Classification Accuracy')
    dual_axes[-1].set_ylabel('Attack Detection Accuracy')
    #dual_axes[-1].legend()
    plt.savefig('all_mnist_active.png')
    #plt.legend()
    #plt.show()
        #ax1.cla()
        #ax2.cla()
        #fig.clf() 
        #plt.clf()

def eps_grid(args):
    lp = args.test_lp
    if args.dataset == 'yale':
        if lp == 1:
            epss = [0.0, 2.25, 4.5, 7.5, 9.75, 12.0, 15.0]
        elif lp == 2:
            epss = [0.0, 0.75, 1.5, 2.5, 3.25, 4.0, 5.0]
        elif lp == np.infty:
            epss = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
    elif args.dataset == 'mnist':
        if lp == 1: 
            epss = [0.0, 1.5, 3.0, 5.0, 6.5, 8.0, 10.0]
        elif lp == 2: 
            epss =[0.0, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0]
        elif lp == np.infty:
            epss = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for eps in epss:
        print("-------------EPS = {}---------------".format(eps))
        np.random.seed(0)
        trainer = Trainer(args)
        #trainer.train(75, 0.05) 
        trainer.net.load_state_dict(torch.load('files/pretrained_model_ce_{}_{}.pth'.format(args.arch, args.dataset), map_location=torch.device('cpu')))
        test_acc = trainer.evaluate(test=True)
        print("Loaded pretrained model!. Test accuracy: {}%".format(test_acc))
        lp_variant = args.lp_variant
        sbsc.sbsc(trainer, args, eps, lp, lp_variant)

def sbsc_test(args):
    np.random.seed(0)
    use_pca = False
    use_gan_Ds = False
    trainer = Trainer(args, use_maini_cnn=False, use_pca=use_pca)
    trainer.net.eval()
    #trainer.train() 
    #trainer.train_encoder()
    #set_trace()
    #trainer.train_decoder(use_pca=use_pca)
    #trainer.test_decoder()
    #set_trace()
    if args.arch == 'wresnet': 
        trainer.net.load_state_dict(torch.load('files/pretrained_model_ce_{}_{}.pth'.format(args.arch, args.dataset), map_location=torch.device('cpu'))['state_dict'])
    elif args.arch != 'densenet':
        trainer.net.load_state_dict(torch.load('files/pretrained_model_ce_{}_{}.pth'.format(args.arch, args.dataset), map_location=torch.device('cpu')))
    if args.embedding == 'scattering':
        if use_pca:
            trainer.decoder.load_state_dict(torch.load('files/decoder_scattering_pca_{}.pth'.format(args.dataset), map_location=torch.device('cpu')))
        else:
            trainer.decoder.load_state_dict(torch.load('files/decoder_scattering_{}.pth'.format(args.dataset), map_location=torch.device('cpu')))
    elif args.embedding == 'studiogan':
        trainer.decoder.load_state_dict(torch.load('files/netG_{}.pth'.format(args.dataset), map_location=torch.device('cpu'))['state_dict'])
    elif args.embedding == 'gan' or args.embedding == 'wgan':
        trainer.decoder.load_state_dict(torch.load('files/netG_{}.pth'.format(args.dataset), map_location=torch.device('cpu')))
    elif args.embedding == 'stylegan_xl':
        pass
    else:
        print("No decoder loaded!!")
    #test_acc = trainer.evaluate(test=True, topk=True)
    #print("Loaded pretrained model and decoder!. Test accuracy: {}%".format(test_acc))

    eps_map = utils.EPS[args.dataset]
    eps = eps_map[args.test_lp]
    test_lp = args.test_lp
    lp_variant = args.lp_variant
    print("-------------EPS = {}---------------".format(eps))
    sbsc.sbsc(trainer, args, eps, test_lp, lp_variant, use_pca=use_pca, use_gan_Ds=use_gan_Ds)
    #sbsc.serialize_dictionaries(trainer, args)

def sbsc_test_zero_eps(args):
    np.random.seed(0)
    trainer = Trainer(args, use_maini_cnn=False)
    #trainer.train() 
    #set_trace()
    trainer.net.load_state_dict(torch.load('files/pretrained_model_ce_{}_{}.pth'.format(args.arch, args.dataset), map_location=torch.device('cpu')))
    test_acc = trainer.evaluate(test=True)
    print("Loaded pretrained model!. Test accuracy: {}%".format(test_acc))

    eps = 0
    test_lp = args.test_lp
    lp_variant = args.lp_variant
    print("-------------EPS = {}---------------".format(eps))
    sbsc.sbsc(trainer, args, eps, test_lp, lp_variant)

def sbsc_maini_test(args):
    np.random.seed(0)
    trainer = Trainer(args, use_maini_cnn=True)
    #trainer.train(75, 0.05) 
    model_name = 'msd'
    device_id = 0
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(int(device_id))
    model_address = "files/MNIST_Baseline_Models/{}.pt".format(model_name.upper())
    trainer.net.load_state_dict(torch.load(model_address, map_location=device))
    test_acc = trainer.evaluate(test=True)
    print("Loaded pretrained model Maini {}!. Test accuracy: {}%".format(model_name, test_acc))

    eps_map = utils.EPS[args.dataset]
    eps = eps_map[args.test_lp]
    test_lp = args.test_lp
    sbsc.sbsc(trainer, args, eps, test_lp, use_cnn_for_dict=True)

# element-wise max of all attacks in toolchain
def adaptive_attack(args):
    np.random.seed(0)
    trainer = Trainer(args, use_maini_cnn=False)

    trainer.net.load_state_dict(torch.load('files/pretrained_model_ce_{}_{}.pth'.format(args.arch, args.dataset), map_location=torch.device('cpu')))
    test_acc = trainer.evaluate(test=True)
    print("Loaded pretrained model!. Test accuracy: {}%".format(test_acc))

    test_idx = np.random.choice(list(range(trainer.N_test)), 100)
    test_x = trainer.test_X[test_idx, :, :]
    test_y = trainer.test_y[test_idx]

    eps_map = utils.EPS[args.dataset]
    attacks = list()
    for lp in args.toolchain:
        lp = trainer.test_lp_attack(lp, test_x, test_y, eps_map[lp], realizable=False, lp_variant=None)
        attacks.append(lp)
    test_adv = np.maximum.reduce(attacks)  

    sbsc.sbsc(trainer, args, 0, -1, None, test_adv=test_adv)
    set_trace()

def embedding_train(args):
    Ds = pickle.load(open('files/Ds_{}_unnorm.pkl'.format(args.dataset), 'rb'))
    embed_trainer = EmbeddingTrainer(args, Ds)
    embed_trainer.train()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Experiments')
    parser = utils.get_parser(parser)
    args = parser.parse_args()

    #sbsc_maini_test(args)
    #sbsc_test_zero_eps(args)
    sbsc_test(args)
    #adaptive_attack(args)
    #eps_grid(args)
    #eps_plot()
    #embedding_train(args)
