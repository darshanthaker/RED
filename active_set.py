import numpy as np
import cvxpy as cp
import copy
import utils

from cvxpy.atoms.norm import norm
from pdb import set_trace
from block_indexer import BlockIndexer
from ordered_set import OrderedSet

class BlockSparseActiveSetSolver(object):

    def __init__(self, Ds, Da, decoder, num_classes, num_attacks, block_size, \
                 max_iter=4, lambda1=0.1, lambda2=0.1):
        self.Ds = Ds
        self.Da = Da
        self.decoder = decoder
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
        prev_sig_active_set = OrderedSet()
        sig_active_set = OrderedSet()
        self.sig_bi = BlockIndexer(self.block_size, [self.num_classes])

        cs_est = np.zeros(m)
        
        converged = False
        t = 0
        while not converged and t < self.max_iter:
            Ds_a = self.get_active_blocks(Ds_est, self.sig_bi, sig_active_set)
            cs_a = self.get_active_blocks(cs_est, self.sig_bi, sig_active_set)
            sig_norms = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                Ds_i = self.sig_bi.get_block(Ds_est, i)
                val = Ds_i.T @ (x - Ds_a @ cs_a)
                sig_norms[i] = np.linalg.norm(val)

            sorted_sig_norms = np.sort(sig_norms)[::-1]
            gamma_s = (sorted_sig_norms[0] + sorted_sig_norms[1]) / 2
            max_sig_idx  = np.argmax(sig_norms)
            sig_active_set.add(max_sig_idx)

            if sig_active_set.issubset(prev_sig_active_set):
                converged = True

            print("Sig active set: {}".format(sig_active_set))

            Ds_a = self.get_active_blocks(Ds_est, self.sig_bi, sig_active_set)
            s_len = Ds_a.shape[1]
            cs_a = cp.Variable(s_len)
            objective = norm(x - Ds_a@cs_a, p=2)**2

            sig_bi = BlockIndexer(self.block_size, [len(sig_active_set)])
            for i in range(len(sig_active_set)):
                cs_i = sig_bi.get_block(cs_a, i)
                objective += gamma_s * self.lambda1 * norm(cs_i, p=2)
                objective += gamma_s * (1.0 - self.lambda1) / 2. * norm(cs_i, p=2)**2
        
            minimizer = cp.Minimize(objective)
            prob = cp.Problem(minimizer )
            result = prob.solve(verbose=False, max_iters=75, solver=cp.SCS)
            cs_est_blocks = np.array(cs_a.value)

            active_sig_bi = BlockIndexer(self.block_size, [len(sig_active_set)])
            for (sig_idx, i) in enumerate(sig_active_set):
                try:
                    cs_est_i = active_sig_bi.get_block(cs_est_blocks, sig_idx)
                except:
                    cs_est_i = np.zeros(self.block_size)
                    print("SOLVER FAILED")
                cs_est = self.sig_bi.set_block(cs_est, i, cs_est_i)

            prev_sig_active_set = copy.deepcopy(sig_active_set)
            t += 1
        print("Number of iterations: {}".format(t))
        return cs_est, sig_active_set

    def _solve_full(self, x):
        return self._elastic_net_solver_full(x, self.Ds, self.Da)
   
    def get_active_blocks(self, x, bi, active_set, hier_active_set=None):
        if len(active_set) == 0:
            if hier_active_set is not None: 
                return np.zeros(bi.get_block(x, (0, 0)).shape)
            else:
                return np.zeros(bi.get_block(x, 0).shape) 
        if hier_active_set is not None and len(hier_active_set) == 0:
            return np.zeros(bi.get_block(x, (0, 0)).shape)
        if hier_active_set is not None: 
            x_blocks = [bi.get_block(x, (i, j)) for i in active_set for j in hier_active_set]
        else:
            x_blocks = [bi.get_block(x, i) for i in active_set]
        return np.hstack(x_blocks)

    def _solve_Ds_Da_alt(self, x):
        Ds_est = copy.deepcopy(self.Ds)
        Da_est = copy.deepcopy(self.Da)
        m = self.Ds.shape[1]
        n = self.Da.shape[1]

        cs_est = np.zeros(m)
        ca_est = np.zeros(n)
        
        converged = False
        t = 0

        self.hier_bi = BlockIndexer(self.block_size, [self.num_classes, self.num_attacks])
        self.sig_bi = BlockIndexer(self.block_size, [self.num_classes])

        prev_sig_active_set = OrderedSet()
        prev_att_active_set = OrderedSet()
        sig_active_set = OrderedSet()
        att_active_set = OrderedSet()

        gamma_s = 10
        gamma_a = 10
        while not converged and t < self.max_iter:
            Ds_a = self.get_active_blocks(Ds_est, self.sig_bi, sig_active_set)
            cs_a = self.get_active_blocks(cs_est, self.sig_bi, sig_active_set)
            Da_a = self.get_active_blocks(Da_est, self.hier_bi, sig_active_set, hier_active_set=att_active_set)
            ca_a = self.get_active_blocks(ca_est, self.hier_bi, sig_active_set, hier_active_set=att_active_set)
            #print("----------------{}-----------------".format(t))
            #print("-----------cs---------")
            sig_norms = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                Ds_i = self.sig_bi.get_block(Ds_est, i)
                val = Ds_i.T @ (x - Ds_a @ cs_a - Da_a @ ca_a)
                sig_norms[i] = np.linalg.norm(val)
            sig_norms /= gamma_s

            sig_active_idxs = np.where(sig_norms >= self.lambda1)[0]
            for sig_idx in sig_active_idxs:
                sig_active_set.add(sig_idx)

            att_norms = np.zeros((self.num_classes, self.num_attacks))
            for i in range(self.num_classes):
                for j in range(self.num_attacks):
                    Da_ij = self.hier_bi.get_block(Da_est, (i, j))
                    val = Da_ij.T @ (x - Ds_a @ cs_a - Da_a @ ca_a)
                    att_norms[i][j] = np.linalg.norm(val)
            att_norms /= gamma_a

            att_active_idxs = np.where(att_norms >= self.lambda2)[1]
            for att_idx in att_active_idxs:
                att_active_set.add(att_idx)

            #if max_sig_idx in sig_active_set and max_att_idx[1] in att_active_set:
            if sig_active_set.issubset(prev_sig_active_set) and att_active_set.issubset(prev_att_active_set):
                converged = True
            print("Sig active set: {}. Att active set: {}".format(sig_active_set, att_active_set))

            cs_est_blocks, ca_est_blocks = self._elastic_net_solver(x, Ds_est, Da_est, gamma_s, gamma_a, sig_active_set, att_active_set, reg_sig=True, reg_att=True)
            active_sig_bi = BlockIndexer(self.block_size, [len(sig_active_set)])
            active_hier_bi = BlockIndexer(self.block_size, [len(sig_active_set), len(att_active_set)])
            for (sig_idx, i) in enumerate(sig_active_set):
                cs_est_i = active_sig_bi.get_block(cs_est_blocks, sig_idx)
                cs_est = self.sig_bi.set_block(cs_est, i, cs_est_i)
                for (att_idx, j) in enumerate(att_active_set):
                    ca_est_ij = active_hier_bi.get_block(ca_est_blocks, (sig_idx, att_idx))
                    ca_est = self.hier_bi.set_block(ca_est, (i, j), ca_est_ij)
            #set_trace()
            #converged = True
            prev_sig_active_set = copy.deepcopy(sig_active_set)
            prev_att_active_set = copy.deepcopy(att_active_set)
            t += 1
        print("Number of iterations: {}".format(t))
        return cs_est, ca_est, sig_active_set, att_active_set

    def get_decoder_Ds_cs(self, Ds, cs_est):
        torch_inp = torch.from_numpy(np.asarray(Ds @ cs_est, dtype=np.float32))
        torch_inp = torch_inp.reshape((1, 81, 7, 7))
        #torch_inp = torch_inp.reshape((1, 1, 28, 28))
        if torch.cuda.is_available():
            torch_inp = torch_inp.cuda()
        decoder_out = self.decoder(torch_inp).cpu().detach().numpy().reshape(-1)
        return torch_inp, decoder_out

    def compute_optimality(self, x, Ds_est, Da_est, cs_est, Ds_a, cs_a, Da_a, ca_a):
        sig_norms = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            Ds_i = self.sig_bi.get_block(Ds_est, i)
            cs_i = self.sig_bi.get_block(cs_est, i)
            torch_inp, _ = self.get_decoder_Ds_cs(Ds_i, cs_i)
            _, decoder_out = self.get_decoder_Ds_cs(Ds_a, cs_a)
            decoder_grad = torch.autograd.functional.jacobian(self.decoder, torch_inp, strict=True)
            decoder_grad = decoder_grad.cpu().detach().numpy().reshape((d, embed_d))
            val = Ds_i.T @ decoder_grad.T @ (x - decoder_out - Da_a @ ca_a)
            sig_norms[i] = np.linalg.norm(val)

        att_norms = np.zeros((self.num_classes, self.num_attacks))
        for i in range(self.num_classes):
            for j in range(self.num_attacks):
                Da_ij = self.hier_bi.get_block(Da_est, (i, j))
                _, decoder_out = self.get_decoder_Ds_cs(Ds_a, cs_a)
                val = Da_ij.T @ (x - decoder_out - Da_a @ ca_a)
                att_norms[i][j] = np.linalg.norm(val)
        return sig_norms, att_norms

    def _solve_Ds_Da(self, x):
        Ds_est = copy.deepcopy(self.Ds)
        Da_est = copy.deepcopy(self.Da)
        m = self.Ds.shape[1]
        n = self.Da.shape[1]
        prev_sig_active_set = OrderedSet()
        prev_att_active_set = OrderedSet()
        sig_active_set = OrderedSet()
        att_active_set = OrderedSet()
        self.hier_bi = BlockIndexer(self.block_size, [self.num_classes, self.num_attacks])
        self.sig_bi = BlockIndexer(self.block_size, [self.num_classes])

        cs_est = np.zeros(m)
        ca_est = np.zeros(n)

        converged = False
        t = 0
        while not converged and t < self.max_iter:
            Ds_a = self.get_active_blocks(Ds_est, self.sig_bi, sig_active_set)
            cs_a = self.get_active_blocks(cs_est, self.sig_bi, sig_active_set)
            Da_a = self.get_active_blocks(Da_est, self.hier_bi, sig_active_set, hier_active_set=att_active_set)
            ca_a = self.get_active_blocks(ca_est, self.hier_bi, sig_active_set, hier_active_set=att_active_set)
            sig_norms, att_norms = self.compute_optimality(x, Ds_est, Da_est, cs_est, Ds_a, cs_a, Da_a, ca_a)

            sorted_sig_norms = np.sort(sig_norms)[::-1]
            sorted_att_norms = np.sort(att_norms.reshape(-1))[::-1]
            gamma_s = (sorted_sig_norms[0] + sorted_sig_norms[1]) / 2
            gamma_a = (sorted_att_norms[0] + sorted_att_norms[1]) / 2
            #gamma_s = sorted_sig_norms[0]
            #gamma_a = sorted_att_norms[0]
            print("gamma_s: {}. gamma_a: {}".format(gamma_s, gamma_a))
            max_sig_idx  = np.argmax(sig_norms)
            max_att_idx = np.unravel_index(np.argmax(att_norms), att_norms.shape)[1]
            sig_active_set.add(max_sig_idx)
            att_active_set.add(max_att_idx)

            if sig_active_set.issubset(prev_sig_active_set) and att_active_set.issubset(prev_att_active_set):
                converged = True
                #break

            print("Sig active set: {}. Att active set: {}".format(sig_active_set, att_active_set))

            cs_est_blocks, ca_est_blocks = self._elastic_net_solver(x, Ds_est, Da_est, gamma_s, gamma_a, sig_active_set, att_active_set)
            active_sig_bi = BlockIndexer(self.block_size, [len(sig_active_set)])
            active_hier_bi = BlockIndexer(self.block_size, [len(sig_active_set), len(att_active_set)])
            for (sig_idx, i) in enumerate(sig_active_set):
                #if i != max_sig_idx:
                #    continue
                try:
                    cs_est_i = active_sig_bi.get_block(cs_est_blocks, sig_idx)
                except:
                    cs_est_i = np.zeros(self.block_size)
                    print("SOLVER FAILED")
                cs_est = self.sig_bi.set_block(cs_est, i, cs_est_i)
                for (att_idx, j) in enumerate(att_active_set):
                    #if i != max_sig_idx and j != max_att_idx:
                    #    continue
                    ca_est_ij = active_hier_bi.get_block(ca_est_blocks, (sig_idx, att_idx))
                    ca_est = self.hier_bi.set_block(ca_est, (i, j), ca_est_ij)

            #print("Max sig idx: {}. Max att idx: {}".format(max_sig_idx, max_att_idx))
            #set_trace()
            #converged = True
            prev_sig_active_set = copy.deepcopy(sig_active_set)
            prev_att_active_set = copy.deepcopy(att_active_set)
            t += 1
        print("Number of iterations: {}".format(t))
        return cs_est, ca_est, sig_active_set, att_active_set

    @utils.timing
    def _elastic_net_solver(self, x, Ds_est, Da_est, gamma_s, gamma_a, sig_active, att_active, reg_sig=True, reg_att=True):
        Ds_a = self.get_active_blocks(Ds_est, self.sig_bi, sig_active)
        Da_a = self.get_active_blocks(Da_est, self.hier_bi, sig_active, hier_active_set=att_active)

        s_len = Ds_a.shape[1]
        a_len = Da_a.shape[1]

        cs_a = cp.Variable(s_len)
        ca_a = cp.Variable(a_len) 
        objective = norm(x - Ds_a@cs_a - Da_a@ca_a, p=2)**2

        hier_bi = BlockIndexer(self.block_size, [len(sig_active), len(att_active)])
        sig_bi = BlockIndexer(self.block_size, [len(sig_active)])
        for i in range(len(sig_active)):
            if reg_sig:
                cs_i = sig_bi.get_block(cs_a, i)
                objective += gamma_s * self.lambda1 * norm(cs_i, p=2)
                objective += gamma_s * (1.0 - self.lambda1) / 2. * norm(cs_i, p=2)**2
            for j in range(len(att_active)):
                if reg_att:
                    ca_ij = hier_bi.get_block(ca_a, (i, j))
                    objective += gamma_a * self.lambda2 * norm(ca_ij, p=2)
                    objective += gamma_a * (1.0 - self.lambda2) / 2. * norm(ca_ij, p=2)**2
        
        minimizer = cp.Minimize(objective)
        prob = cp.Problem(minimizer)
        result = prob.solve(verbose=False, max_iters=50, solver=cp.SCS)
        cs_a_opt = np.array(cs_a.value)
        ca_a_opt = np.array(ca_a.value)
        return cs_a_opt, ca_a_opt

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

    #@utils.timing
    def solve(self, x, alg='Ds+Da'):
        if alg == 'Ds':
            cs_est, sig_active_set = self._solve_Ds(x)
            Ds_est = self.Ds
            Da_est = self.Da
            sig_active_set = list(sig_active_set)
            err_class = list()
            for i in sig_active_set:
                Ds_blk = self.sig_bi.get_block(Ds_est, i)
                cs_blk = self.sig_bi.get_block(cs_est, i)
                err_class.append(np.linalg.norm(x - Ds_blk@cs_blk))
            err_class = np.array(err_class)
            i_star = np.argmin(err_class)
            Ds_blk = self.sig_bi.get_block(Ds_est, i_star)
            cs_blk = self.sig_bi.get_block(cs_est, i_star)
            denoised = Ds_blk@cs_blk
            class_pred = sig_active_set[i_star]
            return cs_est, Ds_est, class_pred, denoised
        elif alg == 'Ds+Da':
            cs_est, ca_est, sig_active_set, att_active_set = self._solve_Ds_Da(x)
            Ds_est = self.Ds
            Da_est = self.Da
        elif alg == 'en_full':
            cs_est, ca_est, Ds_est, Da_est = self._solve_full(x)
            sig_active_set = range(self.sig_bi.num_blocks[0])
            att_active_est = range(self.num_attacks)

        sig_active_set = list(sig_active_set)
        att_active_set = list(att_active_set)
        print("Sig active set: {}. Att active set: {}".format(sig_active_set, att_active_set))
        err_class = list()
        for i in sig_active_set:
            Ds_blk = self.sig_bi.get_block(Ds_est, i)
            cs_blk = self.sig_bi.get_block(cs_est, i)
            # TODO: This shouldn't be full Da.
            err_class.append(np.linalg.norm(x - Ds_blk@cs_blk - Da_est@ca_est))
        err_class = np.array(err_class)
        i_star = np.argmin(err_class)
        class_pred = sig_active_set[i_star]

        err_attack = list()
        for j in att_active_set:
            Da_blk = self.hier_bi.get_block(Da_est, (class_pred, j)) 
            ca_blk = self.hier_bi.get_block(ca_est, (class_pred, j)) 
            err_attack.append(np.linalg.norm(x - Ds_est@cs_est - Da_blk@ca_blk))
        err_attack = np.array(err_attack)
        j_star = np.argmin(err_attack)
        attack_pred = att_active_set[j_star]
        Ds_blk = self.sig_bi.get_block(Ds_est, class_pred)
        cs_blk = self.sig_bi.get_block(cs_est, class_pred)
        denoised = Ds_blk@cs_blk
        return cs_est, ca_est, Ds_est, Da_est, class_pred, attack_pred, denoised, err_attack
