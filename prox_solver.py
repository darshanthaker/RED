import numpy as np
import torch
import copy
import pickle

from pdb import set_trace
from block_indexer import BlockIndexer

class ProxSolver(object):

    def __init__(self, Ds, Da, decoder, num_classes, num_attacks, block_size, max_iter=300, lambda1=3, lambda2=15):
        self.Ds = Ds
        self.Da = Da
        self.num_classes = num_classes
        self.num_attacks = num_attacks
        self.block_size = block_size
        self.max_iter = max_iter
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.decoder = decoder

        self.hier_bi = BlockIndexer(self.block_size, [self.num_classes, self.num_attacks])
        self.sig_bi = BlockIndexer(self.block_size, [self.num_classes])

        assert self.Ds.shape[1] == num_classes * block_size
        assert self.Da.shape[1] == num_classes * num_attacks * block_size
    
    def prox_l2(self, vec, lam):
        vec_norm = np.linalg.norm(vec)
        if vec_norm > lam:
            out = vec - lam* vec / vec_norm
        else:
            out = np.zeros(vec.shape)
        return out

    def prox_l1_2(self, vec, lam, bi):
        out = np.zeros(vec.shape)
        if bi.hierarchical:
            for i in range(bi.num_blocks[0]):
                for j in range(bi.num_blocks[1]):
                    block = bi.get_block(vec, (i, j))
                    block = self.prox_l2(block, lam)
                    out = bi.set_block(out, (i, j), block)
        else:
            for i in range(bi.num_blocks[0]):
                block = bi.get_block(vec, i)
                block = self.prox_l2(block, lam)
                out = bi.set_block(out, i, block)
        return out

    def get_decoder_Ds_cs(self, Ds, cs_est):
        torch_inp = torch.from_numpy(np.asarray(Ds @ cs_est, dtype=np.float32))
        torch_inp = torch_inp.reshape((1, 81, 7, 7))
        #torch_inp = torch_inp.reshape((1, 1, 28, 28))
        if torch.cuda.is_available():
            torch_inp = torch_inp.cuda()
        decoder_out = self.decoder(torch_inp).cpu().detach().numpy().reshape(-1)
        return torch_inp, decoder_out

    def compute_loss(self, x, cs_est, ca_est):
        _, decoder_out = self.get_decoder_Ds_cs(self.Ds, cs_est) 
        #fitting = 0.5*np.linalg.norm(x - decoder_out - self.Da @ ca_est)**2
        fitting = 0.5*np.linalg.norm(x - decoder_out)**2
        cs_norm = sum([np.linalg.norm(self.sig_bi.get_block(cs_est, i)) \
                for i in range(self.num_classes)])
        ca_norm = sum([np.linalg.norm(self.hier_bi.get_block(cs_est, (i, j))) \
                for i in range(self.num_classes) for j in range(self.num_attacks)])
        #loss = fitting + cs_norm + ca_norm
        loss = fitting + cs_norm
        #loss = fitting
        return loss

    def btls(self, grad, x, c,  beta, lam, bi):
        t = 1
        assert beta > 0 and beta < 1
        while True:
            prox = self.prox_l1_2(c, lam, bi)
            general_grad = (c - prox) / t
            _, decoder_out = self.get_decoder_Ds_cs(self.Ds, c) 
            fitting = 0.5*np.linalg.norm(x - decoder_out)**2
            suff_decrease = fitting - t*(grad.T @ general_grad) + \
                    t/2 * np.linalg.norm(general_grad)**2
            update = self.prox_l1_2(c - t*grad, lam, bi)
            _, decoder_out = self.get_decoder_Ds_cs(self.Ds, update) 
            fitting_update = 0.5*np.linalg.norm(x - decoder_out)**2
            if fitting_update <= suff_decrease:
                break
            else:
                t *= beta
        return t

    def solve(self, x):
        embed_d = self.Ds.shape[0]
        d = x.shape[0]
        n = self.Ds.shape[1]
        cs_est = np.zeros(self.Ds.shape[1])
        ca_est = np.zeros(self.Da.shape[1])
        #cs_est = np.random.randn(self.Ds.shape[1])
        #ca_est = np.random.randn(self.Da.shape[1])
        #eta_s = 0.003
        beta_s = 0.9
        eta_a = 0.0005
        T = 100

        for t in range(T):
            if t % 1 == 0:
                loss = self.compute_loss(x, cs_est, ca_est)
                print("[{}] Loss: {}".format(t, loss))
            torch_inp, decoder_out = self.get_decoder_Ds_cs(self.Ds, cs_est)
            decoder_grad = torch.autograd.functional.jacobian(self.decoder, torch_inp, strict=True)
            decoder_grad = decoder_grad.cpu().detach().numpy().reshape((d, embed_d))
            grad_cs_fitting = - self.Ds.T @ decoder_grad.T @ (x - decoder_out)
            #grad_cs_fitting = - self.Ds.T @ decoder_grad.T @ (x - decoder_out - self.Da @ ca_est)
            eta_s = self.btls(grad_cs_fitting, x, cs_est, beta_s, self.lambda1, self.sig_bi)
            print('eta_s: {}'.format(eta_s))
            cs_est = cs_est - eta_s * grad_cs_fitting
            cs_est = self.prox_l1_2(cs_est, self.lambda1, self.sig_bi)

            #torch_inp, decoder_out = self.get_decoder_Ds_cs(self.Ds, cs_est)
            #grad_ca_fitting = - self.Da.T @ (x - decoder_out - self.Da @ ca_est)
            #ca_est = ca_est - eta_a * grad_ca_fitting
            #ca_est = self.prox_l1_2(ca_est, self.lambda2, self.hier_bi)

        err_class = list()
        for i in range(self.num_classes):
            Ds_blk = self.sig_bi.get_block(self.Ds, i)
            cs_blk = self.sig_bi.get_block(cs_est, i)
            _, decoder_out = self.get_decoder_Ds_cs(Ds_blk, cs_blk)
            #err_class.append(np.linalg.norm(x - decoder_out - self.Da @ ca_est))
            err_class.append(np.linalg.norm(x - decoder_out))
        err_class = np.array(err_class)
        class_pred = np.argmin(err_class)

        _, decoder_out = self.get_decoder_Ds_cs(self.Ds, cs_est)
        err_attack = list()
        for j in range(self.num_attacks):
            Da_blk = self.hier_bi.get_block(self.Da, (class_pred, j)) 
            ca_blk = self.hier_bi.get_block(ca_est, (class_pred, j)) 
            err_attack.append(np.linalg.norm(x - decoder_out  - Da_blk @ ca_blk))
        err_attack = np.array(err_attack)
        attack_pred = np.argmin(err_attack)
        Ds_blk = self.sig_bi.get_block(self.Ds, class_pred)
        cs_blk = self.sig_bi.get_block(cs_est, class_pred)
        denoised = Ds_blk@cs_blk

        return cs_est, ca_est, class_pred, attack_pred, denoised
