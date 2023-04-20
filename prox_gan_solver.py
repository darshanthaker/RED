import numpy as np
import torch
import torch.nn as nn
import math
import copy
import pickle
import lpips
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from pdb import set_trace
from block_indexer import BlockIndexer

class ProxGanSolver(object):

    def __init__(self, Da, decoder, num_classes, num_attacks, block_size, input_shape):
        self.Da = Da
        self.num_classes = num_classes
        self.num_attacks = num_attacks
        self.block_size = block_size
        self.input_shape = input_shape
        self.decoder = decoder
        self.lpips_loss = lpips.LPIPS(net='vgg')
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        if torch.cuda.is_available():
            self.lpips_loss = self.lpips_loss.cuda()

        print("Num classes: {}. Num attacks: {}".format(self.num_classes, self.num_attacks))
        self.hier_bi = BlockIndexer(self.block_size, [self.num_classes, self.num_attacks])
        self.sig_bi = BlockIndexer(self.block_size, [self.num_classes])

        assert self.Da.shape[1] == num_classes * num_attacks * block_size
    
    def prox_l2(self, vec, lam):
        vec_norm = np.linalg.norm(vec)
        if vec_norm >= lam:
            out = vec - lam* vec / vec_norm
        else:
            out = np.zeros(vec.shape)
        return out

    def prox_l1_2(self, vec, lam, bi):
        if bi.hierarchical:
            vec_res = vec.reshape((bi.num_blocks[0] * bi.num_blocks[1], bi.block_size)) 
            norms = vec_res.norm(dim=1).unsqueeze(1).repeat(1, bi.block_size)
            norms[norms == 0] = 1
            cond = (norms >= lam)
            update = lam * vec_res / norms
            out_vectorized = cond * (vec_res - update)
            out_vectorized = out_vectorized.reshape(-1)
            if torch.isnan(out_vectorized).any():
                set_trace()
            #for i in range(bi.num_blocks[0]):
            #    for j in range(bi.num_blocks[1]):
            #        block = bi.get_block(vec, (i, j))
            #        block = self.prox_l2(block, lam)
            #        out = bi.set_block(out, (i, j), block)
            #assert np.allclose(out_vectorized.numpy(), out)
        else:
            assert False
        return out_vectorized

    def l_12(self, vec, bi):
        if bi.hierarchical:
            #out = sum([np.linalg.norm(bi.get_block(vec, (i, j))) \
            #        for i in range(bi.num_blocks[0]) for j in range(bi.num_blocks[1])])
            vec_res = vec.reshape((bi.num_blocks[0] * bi.num_blocks[1], bi.block_size)) 
            out_vectorized = vec_res.norm(dim=1).norm(p=1)
            #assert np.allclose(out_vectorized.numpy(), out)
            return out_vectorized
        else:
            assert False

    def compute_loss(self, x, z, ca_est, use_lpips=True):
        if use_lpips:
            decoder_out = self.decoder(z).reshape(-1).detach()
            torch_x = torch.from_numpy(x).reshape(self.input_shape).unsqueeze(0).cuda()
            #Daca = torch.matmul(self.Da, ca_est)
            Daca = self.Da @ ca_est
            Daca = torch.from_numpy(Daca).cuda()
            pred = (decoder_out + Daca).reshape(self.input_shape).unsqueeze(0)
            fitting = self.lpips_loss(torch_x, pred) + self.l1_loss(torch_x, pred)
            fitting = fitting.item()
        else:
            decoder_out = self.decoder(z).reshape(-1).detach().cpu().numpy()
            Daca = self.Da @ ca_est
            fitting = 0.5*np.linalg.norm(x - decoder_out - Daca)**2
        #ca_norm = self.l_12(ca_est, self.hier_bi).detach().cpu().numpy()
        ca_norm = self.l_12(torch.from_numpy(ca_est), self.hier_bi).detach().cpu().numpy()
        loss = fitting + self.lambda2 * ca_norm
        return loss, fitting, ca_norm

    def compute_torch_loss(self, x, z, ca_est, use_lpips=True):
        decoder_out = self.decoder(z).reshape(-1)
        #Daca = torch.matmul(self.Da, ca_est)
        Daca = self.Da @ ca_est
        if torch.cuda.is_available():
            if use_lpips:
                torch_x = torch.from_numpy(x).reshape(self.input_shape).unsqueeze(0).cuda()
            else:
                torch_x = torch.from_numpy(x).cuda()
            Daca = torch.from_numpy(Daca).cuda()
        if use_lpips:
            pred = (decoder_out + Daca).reshape(self.input_shape).unsqueeze(0)
            fitting = self.lpips_loss(torch_x, pred) + self.l1_loss(torch_x, pred)
        else:
            fitting = 0.5*torch.norm(torch_x - decoder_out - Daca)**2
        return fitting

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
  
    def find_lam2(self, x, z):
        grad_norms = list()
        decoder_out = self.decoder(z).reshape(-1)
        #Da_np = self.Da.detach().cpu().numpy()
        Da_np = self.Da
        dec_np = decoder_out.detach().cpu().numpy()
        v = x - dec_np
        #v = x
        grad_ca = -Da_np.T @ v
        b1 = self.hier_bi.num_blocks[0]
        b2 = self.hier_bi.num_blocks[1]
        grad_ca = grad_ca.reshape((b1*b2, self.hier_bi.block_size))
        grad_norms_vectorized  = np.linalg.norm(grad_ca, axis=1)
        """
        for i in range(self.hier_bi.num_blocks[0]):
            for j in range(self.hier_bi.num_blocks[1]):
                Da_ij = self.hier_bi.get_block(Da_np, (i, j)) 
                grad_ca_fitting = -Da_ij.T @ v
                grad_norms.append(np.linalg.norm(grad_ca_fitting))
        set_trace()
        """
        return grad_norms_vectorized

    def adjust_lr(self, optimizer, cur_lr, decay_rate = 0.1, global_step = 1, rec_iter = 200):
        lr = cur_lr * decay_rate ** (global_step / int(math.ceil(rec_iter * 0.8)))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def solve_coef(self, x, accelerated=True, dir_name=None, use_lpips=False):
        embed_d = 62
        #embed_d = 100
        d = x.shape[0]
        #self.Da = torch.from_numpy(self.Da)
        #if torch.cuda.is_available():
        #    self.Da = self.Da.cuda()
        #ca_est = nn.Parameter(torch.zeros(self.Da.shape[1], requires_grad=True, device="cuda"))
        ca_est = np.zeros(self.Da.shape[1]).astype(np.float32)

        pickle.dump(x, open('{}/test_x.pkl'.format(dir_name), 'wb'))

        #Da_np = self.Da.detach().cpu().numpy()
        #Da_np = self.Da
        #lip_a = np.linalg.norm(Da_np @ Da_np.T, ord=2)
        #print(lip_a)
        lip_a = 598.254 # fashionmnist
        #lip_a = 1635.532 # cifar
        #lip_a = 1046.2681 #mnist
        eta_a = 1.0 / lip_a
        #eta_a = 0.001
        #T = 500
        T = 500
        converged = False
        ca_est_prev = ca_est
        losses = list()
        decoder_outs = list()

        torch_x = torch.from_numpy(x).reshape(self.input_shape).unsqueeze(0)
        if torch.cuda.is_available():
            torch_x = torch_x.cuda()

        self.decoder.eval()

        best_z = None
        best_z_loss = float("inf")
        for rest in range(100):
            #z = torch.randn(embed_d, device="cuda")
            z = torch.randn((1, embed_d), device="cuda")
            #z = torch.randn((1, embed_d, 1, 1), device="cuda")
            decoder_out = self.decoder(z)
            if use_lpips:
                loss = self.lpips_loss(torch_x, decoder_out) + 0.1 * self.l1_loss(torch_x, decoder_out)
            else:
                loss = self.mse_loss(torch_x, decoder_out)
            if loss < best_z_loss:
                best_z_loss = loss
                best_z = z.detach()
        print("Found best from init: {}".format(loss))

        restarts = 10
        for _ in range(restarts):
            #z = nn.Parameter(torch.randn((1, embed_d, 1, 1), requires_grad=True, device="cuda"))
            z = nn.Parameter(torch.randn((1, embed_d), requires_grad=True, device="cuda"))
            #z = nn.Parameter(torch.randn(embed_d, requires_grad=True, device="cuda"))
            #self.z_optimizer = torch.optim.Adam([z], lr=0.1)
            cur_lr = 10
            self.z_optimizer = torch.optim.SGD([z], lr=cur_lr, momentum=0.7)
            mse_loss = nn.MSELoss()
            num_epochs = 500
            self.decoder.eval()
            print("########## INITIAL Z TRAINING###########")
            for epoch in range(num_epochs):
                decoder_out = self.decoder(z)
                decoder_outs.append(decoder_out.detach().cpu().numpy())
                if use_lpips:
                    loss = self.lpips_loss(torch_x, decoder_out) + 0.1 * self.l1_loss(torch_x, decoder_out)
                else:
                    loss = mse_loss(torch_x, decoder_out)
                losses.append(loss.item())
                self.z_optimizer.zero_grad()
                loss.backward()
                self.z_optimizer.step()
                cur_lr = self.adjust_lr(self.z_optimizer, cur_lr, rec_iter=num_epochs)
                if epoch % 50 == 0:
                    print("[Pretraining {}] Loss: {}".format(epoch, loss.item()))
            if loss < best_z_loss:
                print("Found best: {}".format(loss))
                best_z_loss = loss
                best_z = z.detach()

        z = best_z
        att_norms = self.find_lam2(x, z)
        att_norms = np.sort(att_norms)[::-1]
        print("lambda2 theoretical: {}".format(att_norms[0]))
        #self.lambda2 = 10
        #self.lambda2 = 0.35 * att_norms[0]
        self.lambda2 = 0.25 * att_norms[0]
        #self.lambda2 = 0.9 * att_norms[0]
        print("lambda2: {}".format(self.lambda2))

        #self.optimizer = torch.optim.Adam([z, ca_est], lr=1.1)
        #z = nn.Parameter(torch.randn((1, embed_d, 1, 1), requires_grad=True, device="cuda"))
        cur_lr = 10
        self.z_optimizer = torch.optim.SGD([z], lr=cur_lr, momentum=0.7)
        #self.z_optimizer = torch.optim.Adam([z], lr=0.1)
        #self.ca_optimizer = torch.optim.SGD([ca_est], lr=eta_a, momentum=0.9, nesterov=True)

        print("######################################################################")
        for t in range(T):
            if t % 10 == 0 and t != 0:
                loss, fitting, ca_norm = self.compute_loss(x, z, ca_est, use_lpips=use_lpips)
                #ca_nz_blocks, ca_block_norms = self.compute_nz_blocks(ca_est.detach().cpu().numpy(), self.hier_bi)
                #ca_nz_blocks, ca_block_norms = self.compute_nz_blocks(ca_est, self.hier_bi)
                print("[{}] Loss: {}. Fitting: {}. ca_norm: {}".format( \
                    t, loss, fitting, ca_norm))
                #print("------------------------------------------------------")
                #print("ca_block_norms: {}".format(ca_block_norms))
                #print("ca_nz_blocks: {}".format(ca_nz_blocks))
                #print("------------------------------------------------------")
            """
            if t % 100 == 0 and t != 0:
                #ca_nz_blocks, ca_block_norms = self.compute_nz_blocks(ca_est.detach().cpu().numpy(), self.hier_bi)
                ca_nz_blocks, ca_block_norms = self.compute_nz_blocks(ca_est, self.hier_bi)
                print("------------------------------------------------------")
                print("ca_block_norms: {}".format(ca_block_norms))
                print("ca_nz_blocks: {}".format(ca_nz_blocks))
                print("------------------------------------------------------")
            """

            loss, fitting, ca_norm = self.compute_loss(x, z, ca_est, use_lpips=use_lpips)
            losses.append(loss)

            decoder_out = self.decoder(z)
            decoder_outs.append(decoder_out.detach().cpu().numpy())
            torch_loss = self.compute_torch_loss(x, z, ca_est)
            self.z_optimizer.zero_grad()
            #self.ca_optimizer.zero_grad()
            torch_loss.backward()
            self.z_optimizer.step()
            #cur_lr = self.adjust_lr(self.z_optimizer, cur_lr, rec_iter=T)
            #print("cur_lr = {}".format(cur_lr))
            #self.ca_optimizer.step()
            #ca_est = self.prox_l1_2(ca_est.detach(), self.lambda2 / lip_a, self.hier_bi)
            if accelerated:
                ca_est_t1 = ca_est + (t - 1) / float(t + 2) * (ca_est - ca_est_prev)
                ca_est_prev = ca_est
            else:
                ca_est_t1 = ca_est
            decoder_out = self.decoder(z).detach().cpu().numpy().reshape(-1)
            grad_ca_fitting = - self.Da.T @ (x - decoder_out - self.Da @ ca_est_t1)
            ca_est = ca_est_t1 - eta_a * grad_ca_fitting
            ca_est = self.prox_l1_2(torch.from_numpy(ca_est), self.lambda2 / lip_a, self.hier_bi).numpy()

        print("########## POST Z TRAINING###########")
        num_epochs = 0
        for epoch in range(num_epochs):
            decoder_out = self.decoder(z)
            decoder_outs.append(decoder_out.detach().cpu().numpy())
            torch_x = torch.from_numpy(x).reshape(self.input_shape).unsqueeze(0)
            if torch.cuda.is_available():
                torch_x = torch_x.cuda()
            loss = mse_loss(torch_x, decoder_out)
            self.z_optimizer.zero_grad()
            loss.backward()
            self.z_optimizer.step()
            if epoch % 50 == 0:
                print("[Posttraining {}] Loss: {}".format(epoch, loss))

        #plt.imshow(x.reshape((28, 28)))
        #plt.savefig('decoder_outs/original.png')
        decoder_outs = torch.from_numpy(np.array(decoder_outs)).squeeze(1)
        #if self.input_shape[0] != 1:
        #    decoder_outs = decoder_outs.squeeze(1)
        if dir_name is not None:
            for idx in range(0, len(decoder_outs), 25):
                save_image(decoder_outs[idx:idx+25], "{}/{}.png".format(dir_name, idx), nrow=5, normalize=True)
            pickle.dump(losses, open('{}/losses.pkl'.format(dir_name), 'wb'))
        #print("Final loss curve: {}".format(losses))
        return z, ca_est

    def solve(self, x, dir_name=None):
        z, ca_est = self.solve_coef(x, dir_name=dir_name) 
        decoder_out = self.decoder(z).detach().cpu().numpy().reshape(-1)
        err_attack = [list() for i in range(self.num_classes)]
        #Da_np = self.Da.detach().cpu().numpy()
        Da_np = self.Da
        #ca_est = ca_est.detach().cpu().numpy()
        for i in range(self.num_classes):
            for j in range(self.num_attacks):
                Da_blk = self.hier_bi.get_block(Da_np, (i, j)) 
                ca_blk = self.hier_bi.get_block(ca_est, (i, j)) 
                err_attack[i].append(np.linalg.norm(x - decoder_out  - Da_blk @ ca_blk))
        err_attack = np.array(err_attack)
        attack_pred = np.unravel_index(np.argmin(err_attack), err_attack.shape)[1]
        denoised = self.decoder(z).detach().cpu().numpy().squeeze(0)
        denoised = 0.5 * (denoised + 1) 

        return ca_est, attack_pred, denoised
