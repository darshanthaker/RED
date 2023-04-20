import numpy as np
import torch
import copy
import pickle
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from pdb import set_trace
from block_indexer import BlockIndexer

class ProxSolver(object):

    def __init__(self, Ds, Da, decoder, num_classes, num_attacks, block_size, input_shape, max_iter=300, lambda1=3, lambda2=15):
        self.Ds = Ds
        self.Da = Da
        self.num_classes = num_classes
        self.num_attacks = num_attacks
        self.block_size = block_size
        self.input_shape = input_shape
        self.max_iter = max_iter
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.decoder = decoder

        print("Num classes: {}. Num attacks: {}".format(self.num_classes, self.num_attacks))
        self.hier_bi = BlockIndexer(self.block_size, [self.num_classes, self.num_attacks])
        self.sig_bi = BlockIndexer(self.block_size, [self.num_classes])

        assert self.Ds.shape[1] == num_classes * block_size
        assert self.Da.shape[1] == num_classes * num_attacks * block_size
    
    def prox_l2(self, vec, lam):
        vec_norm = np.linalg.norm(vec)
        if vec_norm >= lam:
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

    def get_decoder_Ds_cs(self, Ds, cs_est, normalize=False):
        inp = np.asarray(Ds @ cs_est, dtype=np.float32)
        if normalize:
            inp = 10*inp / np.linalg.norm(inp)
        torch_inp = torch.from_numpy(inp)
        #torch_inp = torch_inp / torch.norm(torch_inp)
        if torch_inp.shape[0] == 81*7*7:
            torch_inp = torch_inp.reshape((1, 81, 7, 7))
            #torch_inp = torch_inp.reshape((1, -1))
        elif torch_inp.shape[0] == 10*7*7:
            torch_inp =  torch_inp.reshape((1, 10, 7, 7))
        elif torch_inp.shape[0] == 243*8*8:
            torch_inp = torch_inp.reshape((1, 243, 8, 8))
        elif torch_inp.shape[0] == 100:
            torch_inp = torch_inp.reshape((1, 100, 1, 1))
        else:
            torch_inp = torch_inp.reshape(self.input_shape).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            torch_inp = torch_inp.cuda()
        self.decoder.eval()
        #print("inp norm: {}".format(torch.norm(torch_inp)))
        decoder_out = self.decoder(torch_inp).cpu().detach().numpy().reshape(-1)
        return torch_inp, decoder_out

    def compute_loss(self, x, cs_est, ca_est):
        _, decoder_out = self.get_decoder_Ds_cs(self.Ds, cs_est) 
        fitting = 0.5*np.linalg.norm(x - decoder_out - self.Da @ ca_est)**2
        #fitting = 0.5*np.linalg.norm(x - decoder_out)**2
        cs_norm = sum([np.linalg.norm(self.sig_bi.get_block(cs_est, i)) \
                for i in range(self.num_classes)])
        ca_norm = sum([np.linalg.norm(self.hier_bi.get_block(ca_est, (i, j))) \
                for i in range(self.num_classes) for j in range(self.num_attacks)])
        loss = fitting + self.lambda1 * cs_norm + self.lambda2 * ca_norm
        #loss = fitting + self.lambda1*cs_norm
        #loss = fitting
        return loss, fitting, cs_norm, ca_norm

    def btls(self, grad, x, c,  beta, lam, bi):
        t = 1
        assert beta > 0 and beta < 1
        while True:
            prox_update = self.prox_l1_2(c - t*grad, lam, bi)
            general_grad = (c - prox_update) / t
            fitting = 0.5*np.linalg.norm(x - self.get_decoder_Ds_cs(self.Ds, c)[1])**2
            #suff_decrease = fitting - t*(grad.T @ general_grad) + \
            #        t/2 * np.linalg.norm(general_grad)**2
            suff_decrease = fitting - t*(grad.T @ general_grad)
            fitting_update = 0.5*np.linalg.norm(x - \
                    self.get_decoder_Ds_cs(self.Ds, prox_update)[1])**2
            if fitting_update <= suff_decrease:
                break
            else:
                t *= beta
        return t

    def btls_noprox(self, grad, x, c,  beta, lam, bi):
        t = 1
        assert beta > 0 and beta < 1
        while True:
            update = c - t*grad
            general_grad = (c - update) / t
            fitting = 0.5*np.linalg.norm(x - self.get_decoder_Ds_cs(self.Ds, c)[1])**2
            #suff_decrease = fitting - t*(grad.T @ general_grad) + \
            #        t/2 * np.linalg.norm(general_grad)**2
            suff_decrease = fitting - t*(grad.T @ general_grad)
            fitting_update = 0.5*np.linalg.norm(x - \
                    self.get_decoder_Ds_cs(self.Ds, update)[1])**2
            if fitting_update <= suff_decrease:
                break
            else:
                t *= beta
        return t

    def compute_nz_blocks(self, vec, bi): 
        nz_blocks = list()
        block_norms = dict()
        if bi.hierarchical:
            for i in range(bi.num_blocks[0]):
                for j in range(bi.num_blocks[1]):
                    block = bi.get_block(vec, (i, j))
                    if not np.allclose(block, 0):
                        nz_blocks.append((i, j))
                        block_norms[(i, j)] = np.linalg.norm(block)            
        else:
            for i in range(bi.num_blocks[0]):
                block = bi.get_block(vec, i)
                block_norms[i] = np.linalg.norm(block)            
                if not np.allclose(block, 0):
                    nz_blocks.append(i)
                    block_norms[i] = np.linalg.norm(block)            
        return nz_blocks, block_norms
  
    def fitting(self, x, cs, ca):
        #torch_inp = torch.from_numpy(self.Ds).cuda() @ cs
        torch_inp = torch.from_numpy(self.Ds) @ cs
        if torch_inp.shape[0] == 81*7*7:
            torch_inp = torch_inp.reshape((1, 81, 7, 7))
            #torch_inp = torch_inp.reshape((1, -1))
        elif torch_inp.shape[0] == 243*8*8:
            torch_inp = torch_inp.reshape((1, 243, 8, 8))
        elif torch_inp.shape[0] == 100:
            torch_inp = torch_inp.reshape((1, 100, 1, 1))
        else:
            torch_inp = torch_inp.reshape(self.input_shape)[None, None, :]
        if torch.cuda.is_available():
            torch_inp = torch_inp.cuda()

        second_term = torch.from_numpy(self.Da).cuda() @ ca
        #second_term = (torch.from_numpy(self.Da) @ ca).cuda()
        
        #return torch.norm(x - self.decoder(torch_inp).reshape(-1) - torch.from_numpy(self.Da) @ ca)**2
        return torch.norm(x - self.decoder(torch_inp).reshape(-1) - second_term)**2

    def fitting_wrapper(self, cs):
        return self.fitting(self.torch_x, cs, self.ca)

    def fitting_wrapper_ca(self, ca):
        return self.fitting(self.torch_x, self.cs, ca)

    def find_lam1(self, x):
        cs_zero = np.zeros(self.Ds.shape[1], dtype=np.float32)
        grad_norms = list()
        for i in range(self.sig_bi.num_blocks[0]):
            Ds_i = self.sig_bi.get_block(self.Ds, i) 
            cs_i = self.sig_bi.get_block(cs_zero, i)
            torch_inp, _ = self.get_decoder_Ds_cs(Ds_i, cs_i)
            _, decoder_out = self.get_decoder_Ds_cs(self.Ds, cs_zero)
            #if self.input_shape[0] == 1:
            #else:
            #    v = (x - decoder_out).reshape(self.input_shape)[None, :]
            v = (x - decoder_out).reshape(self.input_shape)[None, None, :]
            v = torch.from_numpy(v)
            if torch.cuda.is_available():
                v = v.cuda()
            _, vjp = torch.autograd.functional.vjp(self.decoder, torch_inp, v=v, strict=True)
            grad_cs_fitting = - Ds_i.T @ vjp.reshape(-1).cpu().detach().numpy()
            grad_norms.append(np.linalg.norm(grad_cs_fitting))
        print(grad_norms)
        return grad_norms

    def find_lam2(self, x):
        cs_zero = np.zeros(self.Ds.shape[1], dtype=np.float32)
        ca_zero = np.zeros(self.Da.shape[1], dtype=np.float32)
        grad_norms = list()
        _, decoder_out = self.get_decoder_Ds_cs(self.Ds, cs_zero)
        v = x - decoder_out
        for i in range(self.hier_bi.num_blocks[0]):
            for j in range(self.hier_bi.num_blocks[1]):
                Da_ij = self.hier_bi.get_block(self.Da, (i, j)) 
                ca_ij = self.hier_bi.get_block(ca_zero, (i, j))
                grad_ca_fitting = -Da_ij.T @ v
                grad_norms.append(np.linalg.norm(grad_ca_fitting))
        print(grad_norms)
        return grad_norms

    def solve_coef(self, x, accelerated=True, eta_s=None, eta_a=None, cheat_y=None, dir_name=None):
        embed_d = self.Ds.shape[0]
        d = x.shape[0]
        n = self.Ds.shape[1]
        #cs_est = np.zeros(self.Ds.shape[1], dtype=np.float32)
        ca_est = np.zeros(self.Da.shape[1], dtype=np.float32)
        cs_est = np.random.randn(self.Ds.shape[1]) / 50
        #ca_est = np.random.randn(self.Da.shape[1])
        #cs_est /= np.linalg.norm(cs_est)
        #ca_est /= np.linalg.norm(ca_est)
        #cs_est_i = np.random.randn(self.block_size)
        #cs_est = self.sig_bi.set_block(cs_est, np.random.choice(list(range(10))), cs_est_i)
        if cheat_y is not None:
            print("CHEATING USING LABEL!!!!")
            cs_est_i = np.random.randn(self.block_size)
            cs_est = self.sig_bi.set_block(cs_est, cheat_y, cs_est_i)
        lip_s = 76594
        #lip_s = 100
        #lip_s = np.linalg.norm(self.Ds @ self.Ds.T, ord=2)
        #lip_a = np.linalg.norm(self.Da @ self.Da.T, ord=2)
        #lip_a = 2092
        lip_a = lip_s
        grad_clip_norm = 10
        if eta_s is None:
            eta_s = 1.0 / lip_s
            #eta_s = 1.0 / 1000
            #eta_s = 0.001
        if eta_a is None:
            #eta_a = 1.0 / 2092
            eta_a = 1.0 / lip_a
        print("eta_s: {}. eta_a: {}".format(eta_s, eta_a))

        sig_norms = self.find_lam1(x)
        att_norms = self.find_lam2(x)
        sig_norms = np.sort(sig_norms)[::-1]
        att_norms = np.sort(att_norms)[::-1]
        print("lambda1 theoretical: {}. lambda2 theoretical: {}".format(sig_norms[0], att_norms[0]))
        #self.lambda1 = (sig_norms[0] + sig_norms[1]) / 2
        #self.lambda2 = (att_norms[0] + att_norms[1]) /  2
        self.lambda1 = sig_norms[0] / 4
        self.lambda1 = 500
        #self.lambda1 = sig_norms[0] / 1.5
        self.lambda2 = att_norms[0] / 4
        #self.lambda2 = 10
        print("lambda1: {}. lambda2: {}".format(self.lambda1, self.lambda2))
        T = 150
        cs_est_prev = cs_est
        ca_est_prev = ca_est
        converged = False

        print("######################################################################")
        losses = list()
        decoder_outs = list()
        for t in range(T):
            if t % 10 == 0 and t != 0:
                loss, fitting, cs_norm, ca_norm = self.compute_loss(x, cs_est, ca_est)
                cs_nz_blocks, cs_block_norms = self.compute_nz_blocks(cs_est, self.sig_bi)
                ca_nz_blocks, ca_block_norms = self.compute_nz_blocks(ca_est, self.hier_bi)
                print("[{}] Loss: {}. Fitting: {}. cs_norm: {}. ca_norm: {}".format( \
                    t, loss, fitting, cs_norm, ca_norm))
                print("------------------------------------------------------")
                print("cs_nz_blocks: {}".format(cs_nz_blocks))
                print("cs_block_norms: {}".format(cs_block_norms))
                print("ca_block_norms: {}".format(ca_block_norms))
                print("ca_nz_blocks: {}".format(ca_nz_blocks))
                print("------------------------------------------------------")
                #losses.append(loss)

            loss, fitting, cs_norm, ca_norm = self.compute_loss(x, cs_est, ca_est)
            losses.append(loss)
            cs_nz_blocks, cs_block_norms = self.compute_nz_blocks(cs_est, self.sig_bi)
            #if len(cs_nz_blocks) <= 2:
            #    converged = True
            #    print("YAY CONVERGED!!")
            #    print("cs_nz_blocks: {}".format(cs_nz_blocks))
            #    break
            #if t == 60:
            #    print("Decaying learning rate!")
            #    eta_s = 0.0001
            #    eta_a = 2.0005
            if accelerated:
                cs_est_t1 = cs_est + (t - 1) / float(t + 2) * (cs_est - cs_est_prev)
                cs_est_prev = cs_est
            else:
                cs_est_t1 = cs_est
            torch_inp, decoder_out = self.get_decoder_Ds_cs(self.Ds, cs_est_t1)
            #print("norm g(phi ds cs): {}".format(np.linalg.norm(decoder_out)))
            #print("norm da ca: {}".format(np.linalg.norm(self.Da @ ca_est)))
            decoder_outs.append(decoder_out.reshape(self.input_shape)[None, :])
            if self.input_shape[0] == 1:
                v = (x - decoder_out - self.Da @ ca_est).reshape(self.input_shape)[None, None, :]
            else:
                v = (x - decoder_out - self.Da @ ca_est).reshape(self.input_shape)[None, :]
            v = (x - decoder_out).reshape((1, 1, 28, 28))
            v = torch.from_numpy(v)
            if torch.cuda.is_available():
                v = v.cuda()
            cs = torch.from_numpy(np.asarray(cs_est, dtype=np.float32))
            cs.requires_grad = True
            ca = torch.from_numpy(np.asarray(ca_est, dtype=np.float32))
            #ca.requires_grad = True
            torch_x = torch.from_numpy(x)
            #self.cs = cs.cuda()
            self.ca = ca.cuda()
            #self.cs = cs
            self.torch_x = torch_x.cuda()
            #hessian = torch.autograd.functional.hessian(self.fitting_wrapper, cs)
            #hessian = torch.autograd.functional.hessian(self.fitting_wrapper_ca, ca)
            #vals, vecs = torch.symeig(hessian)
            #print("top eigenvalue: {}".format(vals[-1]))
        
            #decoder_grad = torch.autograd.functional.jacobian(self.decoder, torch_inp, strict=True)
            _, vjp = torch.autograd.functional.vjp(self.decoder, torch_inp, v=v, strict=True)
            grad_cs_fitting = - self.Ds.T @ vjp.reshape(-1).cpu().detach().numpy()
            #if np.linalg.norm(grad_cs_fitting) >= grad_clip_norm:
            #    grad_cs_fitting = grad_clip_norm * grad_cs_fitting / np.linalg.norm(grad_cs_fitting)
            #print("grad norm: {}".format(np.linalg.norm(grad_cs_fitting)))
            cs_est = cs_est_t1 - eta_s * grad_cs_fitting
            cs_est = self.prox_l1_2(cs_est, self.lambda1 / lip_s, self.sig_bi)
            # Project cs so ||Ds cs|| <= 10
            cs_est = cs_est / np.linalg.norm(self.Ds @ cs_est) * 10

            if accelerated:
                ca_est_t1 = ca_est + (t - 1) / float(t + 2) * (ca_est - ca_est_prev)
                ca_est_prev = ca_est
            else:
                ca_est_t1 = ca_est
            torch_inp, decoder_out = self.get_decoder_Ds_cs(self.Ds, cs_est)
            grad_ca_fitting = - self.Da.T @ (x - decoder_out - self.Da @ ca_est_t1)
            ca_est = ca_est_t1 - eta_a * grad_ca_fitting
            ca_est = self.prox_l1_2(ca_est, self.lambda2 / lip_a, self.hier_bi)

            convergence = np.linalg.norm(self.Ds@cs_est - self.Ds@cs_est_t1) / np.linalg.norm(self.Ds@cs_est_t1)
            #print(convergence)
            if np.allclose(cs_est, 0):
                print("cs ALL ZEROS VECTOR!")
                break
            #if convergence <= 1e-4:
            #    break
            #if np.allclose(ca_est, 0):
            #    print("ca ALL ZEROS VECTOR!")

        #plt.imshow(x.reshape((28, 28)))
        #plt.savefig('decoder_outs/original.png')
        #set_trace()
        decoder_outs = torch.from_numpy(np.array(decoder_outs))
        #if self.input_shape[0] != 1:
        #    decoder_outs = decoder_outs.squeeze(1)
        if dir_name is not None:
            for idx in range(0, len(decoder_outs), 25):
                save_image(decoder_outs[idx:idx+25], "{}/{}.png".format(dir_name, idx), nrow=5, normalize=True)
            pickle.dump(losses, open('{}/losses.pkl'.format(dir_name), 'wb'))
        print("Final loss curve: {}".format(losses))
        return cs_est, ca_est, eta_s, eta_a, converged

    def solve(self, x, eta_s=None, eta_a=None, cheat_y=None, dir_name=None):
        converged = False
        #while not converged:
        #    if eta_s is not None:
        #        eta_s = eta_s * 0.1
        #        print("RESTARTING!!")
        cs_est, ca_est, eta_s, eta_a, converged = self.solve_coef(x, eta_s=eta_s, eta_a=eta_a, cheat_y=cheat_y, dir_name=dir_name) 
        err_class = list()
        for i in range(self.num_classes):
            Ds_blk = self.sig_bi.get_block(self.Ds, i)
            cs_blk = self.sig_bi.get_block(cs_est, i)
            _, decoder_out = self.get_decoder_Ds_cs(Ds_blk, cs_blk, normalize=True)
            err_class.append(np.linalg.norm(x - decoder_out - self.Da @ ca_est))
            #err_class.append(np.linalg.norm(x - decoder_out))
        err_class = np.array(err_class)
        class_pred = np.argmin(err_class)

        print(err_class)
        print("PREDICTED LABEL {}".format(class_pred))

        _, decoder_out = self.get_decoder_Ds_cs(self.Ds, cs_est)
        err_attack = [list() for i in range(self.num_classes)]
        for i in range(self.num_classes):
            for j in range(self.num_attacks):
                Da_blk = self.hier_bi.get_block(self.Da, (i, j)) 
                ca_blk = self.hier_bi.get_block(ca_est, (i, j)) 
                err_attack[i].append(np.linalg.norm(x - decoder_out  - Da_blk @ ca_blk))
        err_attack = np.array(err_attack)
        attack_pred = np.unravel_index(np.argmin(err_attack), err_attack.shape)[1]
        Ds_blk = self.sig_bi.get_block(self.Ds, class_pred)
        cs_blk = self.sig_bi.get_block(cs_est, class_pred)
        denoised = Ds_blk@cs_blk
        #set_trace()

        return cs_est, ca_est, class_pred, attack_pred, denoised
