import numpy as np
import re
import torch

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

    def __init__(self, train, s_len, attack_lens):
        self.SIGNAL_BIDXS = [0]
        for i in range(len(train)):
            self.SIGNAL_BIDXS.append(self.SIGNAL_BIDXS[-1] + train[i].shape[0])
        l2_len = attack_lens[0]
        linf_len = attack_lens[1]
        self.ATTACK_BIDXS_F = [s_len + i for i in self.SIGNAL_BIDXS]
        self.ATTACK_BIDXS_F += [s_len+l2_len + i for i in self.SIGNAL_BIDXS]
        self.ATTACK_BIDXS = [s_len, s_len+l2_len, s_len+l2_len+linf_len]

    def get_block(self, x, bidx, mode):
        if mode == SIGNAL:
            if len(x.shape) == 2:
                return x[:, self.SIGNAL_BIDXS[bidx]:self.SIGNAL_BIDXS[bidx+1]]
            else:
                return x[self.SIGNAL_BIDXS[bidx]:self.SIGNAL_BIDXS[bidx+1]]
        elif mode == ATTACK:
            if len(x.shape) == 2:
                return x[:, self.ATTACK_BIDXS[bidx]:self.ATTACK_BIDXS[bidx+1]]
            else:
                return x[self.ATTACK_BIDXS[bidx]:self.ATTACK_BIDXS[bidx+1]]
        elif mode == ATTACK_F:
            assert len(bidx) == 2
            pid, aid = bidx[0], bidx[1]
            final_id = pid*(aid+1)
            if len(x.shape) == 2:
                return x[:, self.ATTACK_BIDXS_F[final_id]:self.ATTACK_BIDXS_F[final_id+1]]
            else:
                return x[self.ATTACK_BIDXS_F[final_id]:self.ATTACK_BIDXS_F[final_id+1]]

    def set_block(self, x, bidx, val, mode):
        if mode == SIGNAL:
            x[self.SIGNAL_BIDXS[bidx]:self.SIGNAL_BIDXS[bidx+1]] = val
        elif mode == ATTACK_F:
            pid, aid = bidx[0], bidx[1]
            final_id = pid*(aid+1)
            x[self.ATTACK_BIDXS_F[final_id]:self.ATTACK_BIDXS_F[final_id+1]] = val
        return x
