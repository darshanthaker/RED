import torch
import torch.nn as nn

from pdb import set_trace

class NN(nn.Module):

    def __init__(self, d, m, num_classes, linear=False):
        super().__init__()
        self.d = d
        self.m = m
        self.lin1 = nn.Linear(self.d, self.m)
        self.lin2 = nn.Linear(self.m, self.m)
        self.lin3 = nn.Linear(self.m, num_classes)
        self.relu = nn.ReLU()
        self.is_linear = linear

    def forward(self, x):
        out = x
        if self.is_linear:
            out = self.lin1(out)
            out = self.lin2(out)
        else:
            out = self.relu(self.lin1(out))
            out = self.relu(self.lin2(out))
        out = self.lin3(out)
        return out

class CNN(nn.Module):

    def __init__(self, arch, in_channels=3, num_classes=10, linear=False):
        super().__init__()
        if arch == 'carlini_cnn':
            cfg = [32, 32, 'M', 64, 64, 'M']

        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3)
                layers += [conv2d, nn.ReLU()]
                in_channels = v
        self.features = nn.ModuleList(layers)

        self.classifier = nn.ModuleList([
            nn.Linear(64 * 4 * 4, 200),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, num_classes)])
          
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            x = layer(x)
        return x
