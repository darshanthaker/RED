import numpy as np
import copy

from pdb import set_trace
from utils import BlockIndexer

class BlockSparseIRLSSolver(object):

    def __init__(self, Ds, Da, num_classes, num_attacks, block_size, \
                 max_iter=300, lambda1=3, lambda2=15, lambda_reg=0.01):
        self.Ds = Ds
        self.Da = Da
        self.num_classes = num_classes
        self.num_attacks = num_attacks
        self.block_size = block_size
        self.max_iter = max_iter
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_reg = lambda_reg

        assert self.Ds.shape[1] == num_classes * block_size
        assert self.Da.shape[1] == num_classes * num_attacks * block_size

        self.hier_bi = BlockIndexer(block_size, [num_classes, num_attacks])
        self.sig_bi = BlockIndexer(block_size, [num_classes])
        self.att_bi = BlockIndexer(block_size*num_classes, [num_attacks])

    def prune(self, Ds_est, Da_est, cs_est, ca_est, del_threshold=4):
        eps = 1e-5
        i = 0
        ws = np.zeros(cs_est.shape)
        wa = np.zeros(ca_est.shape)
        active_classes = np.array(list(range(self.num_classes)))
        norm_ca = np.array([[0 for j in range(self.num_attacks)] for i in range(self.num_classes)])

        while i < len(active_classes) and len(active_classes) > 1:
            for j in range(self.num_attacks):
                norm_ca[i, j] = np.linalg.norm(self.hier_bi.get_block(ca_est, (i, j)))

            del_s = (np.linalg.norm(self.sig_bi.get_block(cs_est, i))**2 +  \
                                self.lambda2*np.sum(norm_ca[i, :]))**0.5
            if del_s < del_threshold:
                print("PRUNING")
                ws = self.sig_bi.delete_block(ws, i)
                Ds_est = self.sig_bi.delete_block(Ds_est, i)
                cs_est = self.sig_bi.delete_block(cs_est, i)
                self.sig_bi.num_blocks[0] = self.sig_bi.num_blocks[0] - 1
                self.sig_bi.sanity_check([Ds_est, cs_est, ws])
                np.delete(active_classes, i)

                for j in range(self.num_attacks):
                    wa = self.hier_bi.delete_block(wa, (i, j))
                    Da_est = self.hier_bi.delete_block(Da_est, (i, j))
                    ca_est = self.hier_bi.delete_block(ca_est, (i, j))
                self.hier_bi.num_blocks[0] = self.hier_bi.num_blocks[0] - 1
                self.hier_bi.sanity_check([Da_est, ca_est, wa])
            else:
                ws = self.sig_bi.set_block(ws, i, self.lambda1/del_s)

                for j in range(self.num_attacks):
                    del_1 = (np.linalg.norm(self.sig_bi.get_block(cs_est, i))**2 + \
                                np.sum(norm_ca[i, :]))**0.5
                    del_2 = norm_ca[i, j]
                    del_1 = max(del_1, eps)
                    del_2 = max(del_2, eps)
                    wa = self.hier_bi.set_block(wa, (i, j), 
                        (self.lambda2*self.lambda1)/(del_1*del_2) + self.lambda_reg)
                i += 1
            print(active_classes)

        assert len(active_classes) == self.hier_bi.num_blocks[0]
        assert len(active_classes) == self.sig_bi.num_blocks[0]
        return Ds_est, Da_est, cs_est, ca_est, ws, wa, active_classes

    def solve(self, x):
        epsilon = 1e-5
        Ds_est = copy.deepcopy(self.Ds)
        Da_est = copy.deepcopy(self.Da)
        m = self.Ds.shape[1]
        n = self.Da.shape[1]

        cs_est = np.random.normal(size=m)
        ca_est = np.zeros(n)

        err_cs = np.zeros(self.max_iter)
        err_ca = np.zeros(self.max_iter)
        converged = False
        t = 0
        thres = 1e-5

        err_cs = list()
        while not converged and t < self.max_iter:
            cs_old = cs_est
            Ds_old = Ds_est
            ca_old = ca_est
            Da_old = Da_est
            Ds_est, Da_est, cs_est, ca_est, ws, wa, active_classes = self.prune(Ds_est, Da_est, cs_est, ca_est)
            Ws = np.diag(ws)
            Wa = np.diag(wa) 
            
            Da_est_gram = Da_est.T @ Da_est
            Ds_est_gram = Ds_est.T @ Ds_est
            ca = np.linalg.solve(Da_est_gram + Wa, Da_est.T @ (x - Ds_est@cs_est))
            cs = np.linalg.solve(Ds_est_gram + Ws, Ds_est.T @ (x - Da_est@ca_est))
            err_cs.append(np.linalg.norm(Ds_est@cs_est - Ds_old@cs_old) / np.linalg.norm(Ds_old@cs_old))
             
            if err_cs[-1] <= thres or len(active_classes) == 1:
                converged = True
            t += 1
        err_cs = np.array(err_cs)
        return cs_est, ca_est, Ds_est, Da_est, err_cs, ws, wa, active_classes
