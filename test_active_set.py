import numpy as np

from pdb import set_trace
from block_indexer import BlockIndexer
from active_set import BlockSparseActiveSetSolver

def main():
    num_classes = 10
    num_attacks = 3 
    bsz = 50
    d = 100
    m = bsz*num_classes
    n = bsz*num_classes*num_attacks
    sig_bi = BlockIndexer(bsz, [num_classes])
    hier_bi = BlockIndexer(bsz, [num_classes, num_attacks])
    correct_class = 0
    correct_att = 0
    num_class = 0
    num_att = 0

    for istar in range(num_classes):
        for jstar in range(num_attacks):
            cs_star = np.zeros(m) 
            ca_star = np.zeros(n) 
            Ds = np.random.normal(size=(d, m))
            Ds = Ds.reshape((d, m))
            Da = np.random.normal(size=(d,n))
            Da = Da.reshape((d, n))
            cs_star_i = np.random.normal(size=bsz)
            ca_star_ij = np.random.normal(size=bsz)
           
            cs_star = sig_bi.set_block(cs_star, istar, cs_star_i)
            ca_star = hier_bi.set_block(ca_star, (istar, jstar), ca_star_ij)

            x = Ds@cs_star + Da@ca_star 

            solver = BlockSparseActiveSetSolver(Ds, Da, num_classes, num_attacks, bsz)

            cs_est, ca_est, sig_active_set, att_active_set = solver.solve(x)

            class_preds = list()
            attack_preds = list()
            err_class = list()
            for i in sig_active_set:
                Ds_blk = solver.sig_bi.get_block(Ds, i)
                cs_blk = solver.sig_bi.get_block(cs_est, i)
                # TODO: This shouldn't be full Da.
                err_class.append(np.linalg.norm(x - Ds_blk@cs_blk - Da@ca_est))
            err_class = np.array(err_class)
            i_pred = np.argmin(err_class)
            class_preds.append(sig_active_set[i_pred])

            err_attack = list()
            for j in att_active_set:
                Da_blk = solver.hier_bi.get_block(Da, (i_pred, j)) 
                ca_blk = solver.hier_bi.get_block(ca_est, (i_pred, j)) 
                err_attack.append(np.linalg.norm(x - Ds@cs_est - Da_blk@ca_blk))
            err_attack = np.array(err_attack)
            j_pred = np.argmin(err_attack)
            attack_preds.append(att_active_set[j_pred])

            print("Predicted signal class: {} vs true {}. Predicted attack class: {} vs true {}".format(class_preds[0], istar, attack_preds[0], jstar))
            if class_preds[0] == istar:
                correct_class += 1
            if attack_preds[0] == jstar:
                correct_att += 1
            num_class += 1
            num_att += 1
    print("Signal classification accuracy: {}%".format(correct_class / float(num_class) * 100.))
    print("Attack classification accuracy: {}%".format(correct_att / float(num_att) * 100.))

main()
