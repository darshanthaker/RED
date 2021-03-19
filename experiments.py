import numpy as np
import matplotlib.pyplot as plt

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

def eps_plot(trainer):
    epss = [0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
    atts = ['l2', 'l1', 'linf']
    att_map = {'l2': 2, 'linf': np.inf}
    result_map = {'l2': {'signal': [], 'att': [], 'test': [], 'backtest': [], 'baseline': []}, 
                  'linf': {'signal': [], 'att': [], 'test': [], 'backtest': [], 'baseline': []}}

    result_map = {'linf': {'eps': [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], 
                           'signal': [94, 89, 85, 82, 73, 63, 56],
                           'att': [24, 50, 88, 93, 92, 91, 86],
                           'test': [98.99, 93.21, 62.96, 22.53, 5.55, 2.5, 2.5],
                           'backtest': [99, 96, 65, 31, 6, 2.5, 3],
                           'baseline': [92, 84, 82, 73, 72, 66, 54]},
                  'l2': {'eps': [0, 0.3, 0.6, 1, 1.3, 1.6, 2], 
                           'signal': [94, 93, 89, 86, 85, 82, 80],
                           'att': [34, 35, 39, 43, 45, 42, 41],
                           'test': [98.99, 97.53, 92.90, 74.38, 55.86, 50, 44.75],
                           'backtest': [99, 99, 95, 73, 59, 52, 47],
                           'baseline': [92, 89, 85, 84, 81, 79, 76]},
                  'l1': {'eps': [0, 1.5, 3, 5, 6.5, 8, 10], 
                           'signal': [94, 93, 90, 85, 85, 85, 82],
                           'att': [44, 44, 47, 52, 55, 59, 66],
                           'test': [98.99, 98.45, 95.68, 90.12, 79.63, 63.88, 41.97],
                           'backtest': [99, 99, 96, 93, 80, 62, 41],
                           'baseline': [92, 87, 84, 83, 81, 78, 75]}}

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
    for (ax1, att) in zip(axes, ['l1', 'l2', 'linf']):
        #fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        dual_axes.append(ax2)
        ax1.plot(result_map[att]['eps'], result_map[att]['signal'], label='SBSC', marker='*', markersize=7)
        ax2.plot(result_map[att]['eps'], result_map[att]['att'], label='AD', linestyle='--', marker='s', color='black', markerfacecolor='none')
        ax1.plot(result_map[att]['eps'], result_map[att]['test'], label='No Defense', marker='*', markersize=7)
        ax1.plot(result_map[att]['eps'], result_map[att]['backtest'], label='SBSC+CNN', marker='*', markersize=7)
        ax1.plot(result_map[att]['eps'], result_map[att]['baseline'], label='BSC', marker='*', markersize=7)
        bm = baseline_map[att]
        #plt.scatter(bm['eps'], bm['acc'], marker='o', color='black', s=7.5)
        for (name, eps, acc) in zip(bm['name'], bm['eps'], bm['acc']):
            ax1.scatter(eps, acc, label=name, marker='x')
            #plt.annotate(name, (eps, acc+1), fontsize=8)
        #ax1.legend(prop={'size': 9})
        #ax2.legend(prop={'size': 9}, handlelength=3)
        ax1.grid()
        ax1.set_yticks(np.arange(0, 101, 10))
        ax2.set_yticks(np.arange(0, 101, 10))
        ax1.set_xlabel('{} Epsilon'.format(att))
    #fig.set_xlabel('Epsilon of Perturbation')
    #fig.set_ylabel('Signal Classification Accuracy')
    #fig.set_ylabel('Attack Detection Accuracy')
    axes[0].set_ylabel('Signal Classification Accuracy')
    dual_axes[-1].set_ylabel('Attack Detection Accuracy')
    plt.savefig('all_mnist.png')
        #ax1.cla()
        #ax2.cla()
        #fig.clf() 
        #plt.clf()

def eps_grid():
    lp = 1
    for eps in [0.0, 1.5, 3.0, 5.0, 6.5, 8.0, 10.0]:
    #for eps in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    #for eps in [0.0, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0]:
        print("-------------EPS = {}---------------".format(eps))
        np.random.seed(0)
        trainer = Trainer(arch=ARCH, dataset=DATASET, bsz=128, embedding=EMBEDDING)
        #trainer.train(75, 0.05) 
        trainer.net.load_state_dict(torch.load('files/pretrained_model_ce_{}_{}.pth'.format(ARCH, DATASET)))
        test_acc = trainer.evaluate(test=True)
        print("Loaded pretrained model!. Test accuracy: {}%".format(test_acc))
        irls(trainer, TOOLCHAIN, lp, eps)



