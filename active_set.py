import numpy as np
import cvxpy as cp
import copy
import utils

from cvxpy.atoms.norm import norm
from pdb import set_trace
from block_indexer import BlockIndexer

class BlockSparseActiveSetSolver(object):

    def __init__(self, Ds, Da, num_classes, num_attacks, block_size, \
                 max_iter=50, lambda1=0.1, lambda2=0.1):
        self.Ds = Ds
        self.Da = Da
        self.num_classes = num_classes
        self.num_attacks = num_attacks
        self.block_size = block_size
        self.max_iter = max_iter
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        assert self.Ds.shape[1] == num_classes * block_size
        assert self.Da.shape[1] == num_classes * num_attacks * block_size

    def _solve_Ds(self, x):
        Ds_est = copy.deepcopy(self.Ds)
        m = self.Ds.shape[1]

        cs_est = np.zeros(m)
        converged = False
        t = 0

        self.sig_bi = BlockIndexer(self.block_size, [self.num_classes])

        prev_sig_idx = None
        while not converged and t < self.max_iter:
            max_sig_val = None
            max_sig_val_norm = -float("inf")
            max_sig_idx = prev_sig_idx
            #print("----------------{}-----------------".format(t))
            for i in range(self.num_classes):
                Ds_i = self.sig_bi.get_block(Ds_est, i)
                cs_i = self.sig_bi.get_block(cs_est, i)
                val = Ds_i.T @ (x - Ds_i@cs_i)
                val_norm = np.linalg.norm(val)
                if val_norm > max_sig_val_norm:
                    max_sig_val = val
                    max_sig_val_norm = val_norm
                    max_sig_idx = i 
            #cs_est_i = 1.0 / (1 - self.lambda1) * utils.blocksoft_thres(max_sig_val, self.lambda1)
            cs_est_i = np.linalg.inv(Ds_i.T @ Ds_i) @ (Ds_i.T @ x)
            cs_est = self.sig_bi.set_block(cs_est, max_sig_idx, cs_est_i)
            if prev_sig_idx == max_sig_idx:
                converged = True
            prev_sig_idx = max_sig_idx
            t += 1
        print("Number of iterations: {}".format(t))
        return cs_est, max_sig_idx

    def _solve_full(self, x):
        return self._elastic_net_solver_full(x, self.Ds, self.Da)

    def _solve_Ds_Da(self, x):
        Ds_est = copy.deepcopy(self.Ds)
        Da_est = copy.deepcopy(self.Da)
        m = self.Ds.shape[1]
        n = self.Da.shape[1]

        cs_est = np.zeros(m)
        ca_est = np.zeros(n)
        
        #cs_est = np.random.normal(size=m)
        #ca_est = np.random.normal(size=n)

        converged = False
        t = 0

        self.hier_bi = BlockIndexer(self.block_size, [self.num_classes, self.num_attacks])
        self.sig_bi = BlockIndexer(self.block_size, [self.num_classes])

        prev_sig_idx = 0
        prev_att_idx = (0, 0)
        sig_active_set = set()
        att_active_set = set()
        while not converged and t < self.max_iter:
            max_sig_val = None
            max_sig_val_norm = -float("inf")
            max_sig_idx = prev_sig_idx
            max_att_val = None
            max_att_val_norm = -float("inf")
            max_att_idx = prev_att_idx
            #print("----------------{}-----------------".format(t))
            #print("-----------cs---------")
            for i in range(self.num_classes):
                Ds_i = self.sig_bi.get_block(Ds_est, i)
                cs_i = self.sig_bi.get_block(cs_est, i)
                Da_ij = self.hier_bi.get_block(Da_est, max_att_idx)
                ca_ij = self.hier_bi.get_block(ca_est, max_att_idx)
                val = Ds_i.T @ (x - Ds_i @ cs_i - Da_ij @ ca_ij)
                val_norm = np.linalg.norm(val)
                #print("{}: norm: {}".format(i, val_norm))
                if val_norm > max_sig_val_norm:
                    max_sig_val = val
                    max_sig_val_norm = val_norm
                    max_sig_idx = i 

            #print("-----------ca---------")
            #cs_est_i = 1.0 / (1 - self.lambda1) * utils.blocksoft_thres(max_sig_val, self.lambda1)
            #cs_est = self.sig_bi.set_block(cs_est, max_sig_idx, cs_est_i)
            #cs_est_i, _ = self._elastic_net_solver(x, Ds_est, Da_est, max_sig_idx, max_att_idx[1])
            #cs_est = self.sig_bi.set_block(cs_est, max_sig_idx, cs_est_i)

            #for i in range(self.num_classes):
            for j in range(self.num_attacks):
                Ds_i = self.sig_bi.get_block(Ds_est, max_sig_idx)
                cs_i = self.sig_bi.get_block(cs_est, max_sig_idx)
                Da_ij = self.hier_bi.get_block(Da_est, (max_sig_idx, j))
                ca_ij = self.hier_bi.get_block(ca_est, (max_sig_idx, j))
                #Ds_i = self.sig_bi.get_block(Ds_est, i)
                #cs_i = self.sig_bi.get_block(cs_est, i)
                #Da_ij = self.hier_bi.get_block(Da_est, (i, j))
                #ca_ij = self.hier_bi.get_block(ca_est, (i, j))
                val = Da_ij.T @ (x - Ds_i @ cs_i - Da_ij @ ca_ij)
                #val = Da_ij.T @ x
                val_norm = np.linalg.norm(val)
                #print("{}: norm: {}".format(j, val_norm))
                if val_norm > max_att_val_norm:
                    max_att_val = val
                    max_att_val_norm = val_norm
                    max_att_idx = (max_sig_idx, j)
                    #max_att_idx = (i, j)

            
            #cs_est_i = np.linalg.inv(Ds_i.T @ Ds_i) @ (Ds_i.T @ x)
            #cs_est_i = 1.0 / (1 - self.lambda1) * utils.blocksoft_thres(max_sig_val, self.lambda1)
            #cs_est = self.sig_bi.set_block(cs_est, max_sig_idx, cs_est_i)
            #ca_est_ij = 1.0 / (1 - self.lambda2) * utils.blocksoft_thres(max_att_val, self.lambda2)
            #ca_est = self.hier_bi.set_block(ca_est, max_att_idx, ca_est_ij)
            cs_est_i, ca_est_ij = self._elastic_net_solver(x, Ds_est, Da_est, max_sig_idx, max_att_idx[1])
            cs_est = np.zeros(m)
            ca_est = np.zeros(n)
            cs_est = self.sig_bi.set_block(cs_est, max_sig_idx, cs_est_i)
            ca_est = self.hier_bi.set_block(ca_est, max_att_idx, ca_est_ij)
            #print("Max sig idx: {}. Max att idx: {}".format(max_sig_idx, max_att_idx))
            #set_trace()
            if max_sig_idx in sig_active_set and max_att_idx[1] in att_active_set:
                converged = True
            #if prev_sig_idx == max_sig_idx and prev_att_idx == max_att_idx:
            #    converged = True
            #converged = True
            sig_active_set.add(max_sig_idx)
            att_active_set.add(max_att_idx[1])
            print("Sig active set: {}. Att active set: {}".format(sig_active_set, att_active_set))
            prev_sig_idx = max_sig_idx
            prev_att_idx = max_att_idx
            t += 1
        print("Number of iterations: {}".format(t))
        return cs_est, ca_est, sig_active_set, att_active_set

    def _elastic_net_solver(self, x, Ds, Da, i, j):
        Ds_i = self.sig_bi.get_block(Ds, i)
        Da_ij = self.hier_bi.get_block(Da, (i, j))
       
        s_len = Ds_i.shape[1]         
        a_len = Da_ij.shape[1]

        cs_i = cp.Variable(s_len)
        ca_ij = cp.Variable(a_len) 
        objective = norm(x - Ds_i@cs_i - Da_ij@ca_ij, p=2)**2

        objective += self.lambda1 * norm(cs_i, p=2)
        objective += (1.0 - self.lambda1) / 2. * norm(cs_i, p=2)**2
        
        objective += self.lambda2 * norm(ca_ij, p=2)
        objective += (1.0 - self.lambda2) / 2. * norm(ca_ij, p=2)**2

        minimizer = cp.Minimize(objective)
        prob = cp.Problem(minimizer)
        result = prob.solve(verbose=False)
        cs_i_opt = np.array(cs_i.value)
        ca_ij_opt = np.array(ca_ij.value)
        return cs_i_opt, ca_ij_opt

    def _elastic_net_solver_full(self, x, Ds, Da):
        s_len = Ds.shape[1]
        a_len = Da.shape[1]

        self.hier_bi = BlockIndexer(self.block_size, [self.num_classes, self.num_attacks])
        self.sig_bi = BlockIndexer(self.block_size, [self.num_classes])

        cs = cp.Variable(s_len)
        ca = cp.Variable(a_len) 
        objective = norm(x - Ds@cs - Da@ca, p=2)**2

        for i in range(self.num_classes):
            cs_i = self.sig_bi.get_block(cs, i)
            objective += self.lambda1 * norm(cs_i, p=2)
            objective += (1.0 - self.lambda1) / 2. * norm(cs_i, p=2)**2
       
        for i in range(self.num_classes):
            for j in range(self.num_attacks): 
                ca_ij = self.hier_bi.get_block(ca, (i, j))
                objective += self.lambda2 * norm(ca_ij, p=2)
                objective += (1.0 - self.lambda2) / 2. * norm(ca_ij, p=2)**2

        minimizer = cp.Minimize(objective)
        prob = cp.Problem(minimizer)
        result = prob.solve(verbose=False)
        cs_opt = np.array(cs.value)
        ca_opt = np.array(ca.value)
        return cs_opt, ca_opt, Ds, Da

    def solve(self, x, alg='Ds+Da'):
        if alg == 'Ds':
            return self._solve_Ds(x)
        elif alg == 'Ds+Da':
            return self._solve_Ds_Da(x)
        elif alg == 'en_full':
            return self._solve_full(x)
