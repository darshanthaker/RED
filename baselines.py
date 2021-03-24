import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from mnist_funcs import *
from pdb import set_trace

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

def net(pretrained=False):
    net = nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.ReLU(), nn.Linear(1024, 10))
    if pretrained:
        model_name = 'vanilla'
        model_address = "files/MNIST_Baseline_Models/{}.pt".format(model_name.upper())
        net.load_state_dict(torch.load(model_address))
    return net


def evaluate_baseline(model_name, given_examples):
    model = net()
    device_id = 0
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(int(device_id))
    model_address = "files/MNIST_Baseline_Models/{}.pt".format(model_name.upper())
    model.load_state_dict(torch.load(model_address, map_location=device))
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    with torch.no_grad():
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        num_correct = 0
        data_loader = [given_examples]
        for (bx, by) in data_loader:
            bx = torch.from_numpy(bx)
            by = torch.from_numpy(by)
            if torch.cuda.is_available():
                bx = bx.cuda()
                by = by.cuda()
            output = model.forward(bx.float())
            pred = output.data.argmax(1)
            num_correct += pred.eq(by.data.view_as(pred)).sum().item()
        set_trace()
        acc = num_correct / len(by) * 100.
        return acc

