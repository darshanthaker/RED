import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

def linear(in_features, out_features, bias=True):
    return nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

def batchnorm_2d(in_features, eps=1e-4, momentum=0.1, affine=True):
    return nn.BatchNorm2d(in_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=True)

def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias)

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GenBlock, self).__init__()

        self.bn1 = batchnorm_2d(in_features=in_channels)
        self.bn2 = batchnorm_2d(in_features=out_channels)

        self.activation = nn.ReLU(inplace=True)

        self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x0 = x
        x = self.bn1(x)
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv2d1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode="nearest")
        x0 = self.conv2d0(x0)
        out = x + x0
        return out

class StudioGenerator(nn.Module):
    def __init__(self, z_dim, img_size, g_conv_dim):
        super(StudioGenerator, self).__init__()
        g_in_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "128": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "256": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "512": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim]
        }

        g_out_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "128": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "256": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "512": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim, g_conv_dim]
        }

        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.z_dim = z_dim
        self.in_dims = g_in_dims_collection[str(img_size)]
        self.out_dims = g_out_dims_collection[str(img_size)]
        self.bottom = bottom_collection[str(img_size)]
        self.num_blocks = len(self.in_dims)

        self.linear0 = linear(in_features=self.z_dim, out_features=self.in_dims[0] * self.bottom * self.bottom, bias=True)

        self.blocks = []
        for index in range(self.num_blocks):
            self.blocks += [[
                GenBlock(in_channels=self.in_dims[index],
                         out_channels=self.out_dims[index])
            ]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.bn4 = batchnorm_2d(in_features=self.out_dims[-1])
        self.activation = nn.ReLU(inplace=True)
        self.conv2d5 = conv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z, eval=False):
        act = self.linear0(z)
        act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                act = block(act)

        act = self.bn4(act)
        act = self.activation(act)
        act = self.conv2d5(act)
        out = self.tanh(act)
        return out


#gan = StudioGenerator(128, 32, 64)
#from pdb import set_trace
#set_trace()
#print(gan)
