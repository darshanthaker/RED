import torch
import torch.nn as nn

from pdb import set_trace

class NN(nn.Module):

    def __init__(self, d, m, num_classes):
        super().__init__()
        self.d = d
        self.m = m
        self.lin1 = nn.Linear(self.d, self.m)
        self.lin2 = nn.Linear(self.m, self.m)
        self.lin3 = nn.Linear(self.m, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        out = self.relu(self.lin1(out))
        out = self.relu(self.lin2(out))
        out = self.lin3(out)
        return out

class CNN(nn.Module):

    def __init__(self, m, num_layers, in_channels=3, num_classes=10):
        super().__init__()
        layers = []
        for i in range(num_layers):
            conv2d = nn.Conv2d(in_channels, m, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU()]
            in_channels = m
        self.features = nn.ModuleList(layers)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(m*9*8, num_classes)

    def forward(self, x):
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.maxpool(x)
        x = self.features[2](x)
        x = self.maxpool(x)
        #for layer in self.features:
        #    x = layer(x)
        #x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)  
        return x


