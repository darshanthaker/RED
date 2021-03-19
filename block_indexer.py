import numpy as np

from pdb import set_trace
from operator import mul
from functools import reduce

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
