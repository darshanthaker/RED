import numpy as np
import re
import torch

from operator import mul
from functools import reduce
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

