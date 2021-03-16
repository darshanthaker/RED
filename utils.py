import numpy as np
import re
import torch
import math

from operator import mul
from functools import reduce
from typing import Union, Tuple
from torch.nn import Module
from kymatio.torch import Scattering2D
from pdb import set_trace

SIGNAL = 'signal'
ATTACK = 'attack'
ATTACK_F = 'attack_f'

def to_tensor_custom(pic):
    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img

class BlockIndexer():

    def __init__(self, block_size, num_blocks):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.hierarchical = len(self.num_blocks) == 2

    def _expand_block_idx(self, block_idx):
        if self.hierarchical:
            (i, j) = block_idx
            assert i < self.num_blocks[0] and j < self.num_blocks[1]
            start_idx = j*self.block_size*self.num_blocks[0] + i*self.block_size
            end_idx = j*self.block_size*self.num_blocks[0] + (i + 1)*self.block_size
        else:
            i = block_idx
            assert i < self.num_blocks[0]
            start_idx = i*self.block_size
            end_idx = (i + 1)*self.block_size
        return (start_idx, end_idx)

    def get_block(self, x, block_idx):
        start_idx, end_idx = self._expand_block_idx(block_idx)
        if len(x.shape) == 2:
            return x[:, start_idx:end_idx]
        else:
            return x[start_idx:end_idx]

    def sanity_check(self, xs):
        for x in xs:
            assert x.shape[-1] == reduce(mul, self.num_blocks) * self.block_size

    def delete_block(self, x, block_idx):
        if self.hierarchical:
            indices = list()
            for j in range(self.num_blocks[1]):
                start_idx, end_idx = self._expand_block_idx((block_idx, j))
                indices.append(np.arange(start_idx, end_idx))
            indices = np.concatenate(indices)
        else:
            start_idx, end_idx = self._expand_block_idx(block_idx)
            indices = np.arange(start_idx, end_idx)
        if len(x.shape) == 2:
            x = np.delete(x, indices, 1)
        else:
            x = np.delete(x, indices)
        return x

    def set_block(self, x, block_idx, val):
        start_idx, end_idx = self._expand_block_idx(block_idx)
        if len(x.shape) == 2:
            x[:, start_idx:end_idx] = val 
        else:
            x[start_idx:end_idx] = val
        return x

class ScatteringTransform(object):

    def __init__(self, J, L=8, shape=(32, 32)):
        self.J = J
        self.L = L
        self.shape = shape
        self.scattering = Scattering2D(J=J, L=L, shape=shape)
        if torch.cuda.is_available():
            self.scattering = self.scattering.cuda()

    def __call__(self, sample):
        if len(sample.shape) == 3:
            (C, N1, N2) = sample.shape
        else:
            (B, C, N1, N2) = sample.shape
        if torch.cuda.is_available():
            sample = sample.cuda()
        embed_sample = self.scattering(sample)
        new_C = int(C*(1 + self.L*self.J + (self.L**2*self.J*(self.J - 1))/2))
        if len(sample.shape) == 3:
            embed_sample = embed_sample.reshape((new_C, \
                    N1//2**self.J, N2//2**self.J))
        else:
            embed_sample = embed_sample.reshape((-1, new_C, \
                    N1//2**self.J, N2//2**self.J))
        return embed_sample

###### TAKEN FROM GITHUB: https://github.com/oscarknagg/adversarial/tree/master/adversarial
def project(x: torch.Tensor, x_adv: torch.Tensor, norm: Union[str, int], eps: float) -> torch.Tensor:
    """Projects x_adv into the l_norm ball around x
    Assumes x and x_adv are 4D Tensors representing batches of images
    Args:
        x: Batch of natural images
        x_adv: Batch of adversarial images
        norm: Norm of ball around x
        eps: Radius of ball
    Returns:
        x_adv: Adversarial examples projected to be at most eps
            distance from x under a certain norm
    """
    if x.shape != x_adv.shape:
        raise ValueError('Input Tensors must have the same shape')

    if norm == 'inf':
        # Workaround as PyTorch doesn't have elementwise clip
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    else:
        delta = x_adv - x

        # Assume x and x_adv are batched tensors where the first dimension is
        # a batch dimension
        mask = delta.view(delta.shape[0], -1).norm(norm, dim=1) <= eps

        scaling_factor = delta.view(delta.shape[0], -1).norm(norm, dim=1)
        scaling_factor[mask] = eps

        # .view() assumes batched images as a 4D Tensor
        delta *= eps / scaling_factor.view(-1, 1, 1, 1)

        x_adv = x + delta

    return x_adv


def random_perturbation(x: torch.Tensor, norm: Union[str, int], eps: float) -> torch.Tensor:
    """Applies a random l_norm bounded perturbation to x
    Assumes x is a 4D Tensor representing a batch of images
    Args:
        x: Batch of images
        norm: Norm to measure size of perturbation
        eps: Size of perturbation
    Returns:
        x_perturbed: Randomly perturbed version of x
    """
    perturbation = torch.normal(torch.zeros_like(x), torch.ones_like(x))
    if norm == 'inf':
        perturbation = torch.sign(perturbation) * eps
    else:
        perturbation = project(torch.zeros_like(x), perturbation, norm, eps)

    return x + perturbation


def generate_misclassified_sample(model: Module,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
    """Generates an arbitrary misclassified sample
    Args:
        model: Model that must misclassify
        x: Batch of image data
        y: Corresponding labels
        clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST
    Returns:
        x_misclassified: A sample for the model that is not classified correctly
    """
    while True:
        x_misclassified = torch.empty_like(x).uniform_(*clamp)

        if model(x_misclassified).argmax(dim=1) != y:
            return x_misclassified
