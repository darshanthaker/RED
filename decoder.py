import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

from kymatio.torch import Scattering2D as Scattering
from kymatio.caching import get_cache_dir
from kymatio.datasets import get_dataset_dir
from pdb import set_trace


device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

class WGANGenerator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32, smooth_relu=False):
        super(WGANGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        if smooth_relu:
            self.fc = nn.Sequential(
                nn.Linear(self.input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.Softplus(beta=10),
                nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
                nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
                nn.Softplus(beta=10),
            )
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.Softplus(beta=10),
                nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
                nn.Tanh(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
                nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
                nn.ReLU(),
            )
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
                nn.Tanh(),
            )
        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class Generator(nn.Module):
    def __init__(self, num_input_channels, num_hidden_channels, num_output_channels=1, filter_size=3):
        super(Generator, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_output_channels = num_output_channels
        self.filter_size = filter_size
        self.build()

    def build(self):
        padding = (self.filter_size - 1) // 2

        self.main = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(self.num_input_channels, self.num_hidden_channels, self.filter_size, bias=False),
            nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(padding),
            nn.Conv2d(self.num_hidden_channels, self.num_hidden_channels, self.filter_size, bias=False),
            nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(padding),
            nn.Conv2d(self.num_hidden_channels, self.num_output_channels, self.filter_size, bias=False),
            nn.BatchNorm2d(self.num_output_channels, eps=0.001, momentum=0.9),
            nn.Tanh()
        )

    def forward(self, input_tensor):
        return self.main(input_tensor)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Regularized inverse scattering')
    parser.add_argument('--num_epochs', default=5, help='Number of epochs to train')
    parser.add_argument('--load_model', default=False, help='Load a trained model?')
    parser.add_argument('--dir_save_images', default='interpolation_images', help='Dir to save the sequence of images')
    args = parser.parse_args()

    num_epochs = args.num_epochs
    load_model = args.load_model
    dir_save_images = args.dir_save_images

    dir_to_save = get_cache_dir('reg_inverse_example')

    transforms_to_apply = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalization for reproducibility issues
    ])

    mnist_dir = get_dataset_dir("MNIST", create=True)
    dataset = datasets.MNIST(mnist_dir, train=True, download=True, transform=transforms_to_apply)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)

    fixed_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    fixed_batch = next(iter(fixed_dataloader))
    fixed_batch = fixed_batch[0].float().to(device)

    scattering = Scattering(J=2, shape=(28, 28)).to(device)

    scattering_fixed_batch = scattering(fixed_batch).squeeze(1)
    num_input_channels = scattering_fixed_batch.shape[1]
    num_hidden_channels = num_input_channels

    generator = Generator(num_input_channels, num_hidden_channels).to(device)
    generator.train()

    # Either train the network or load a trained model
    ##################################################
    if load_model:
        filename_model = os.path.join(dir_to_save, 'model.pth')
        generator.load_state_dict(torch.load(filename_model))
    else:
        criterion = torch.nn.L1Loss()
        optimizer = optim.Adam(generator.parameters())

        for idx_epoch in range(num_epochs):
            print('Training epoch {}'.format(idx_epoch))
            sum_loss = 0
            ct = 0
            for _, current_batch in enumerate(dataloader):
                generator.zero_grad()
                batch_images = Variable(current_batch[0]).float().to(device)
                batch_scattering = scattering(batch_images).squeeze(1)
                batch_inverse_scattering = generator(batch_scattering)
                loss = criterion(batch_inverse_scattering, batch_images)
                loss.backward()
                sum_loss += loss.item()
                ct += 1
                optimizer.step()
            print("[{}] Loss: {}".format(idx_epoch, sum_loss / ct))

        print('Saving results in {}'.format(dir_to_save))

        torch.save(generator.state_dict(), os.path.join(dir_to_save, 'model.pth'))

    generator.eval()

    """
    # We create the batch containing the linear interpolation points in the scattering space
    ########################################################################################
    z0 = scattering_fixed_batch.cpu().numpy()[[0]]
    z1 = scattering_fixed_batch.cpu().numpy()[[1]]
    batch_z = np.copy(z0)
    num_samples = 32
    interval = np.linspace(0, 1, num_samples)
    for t in interval:
        if t > 0:
            zt = (1 - t) * z0 + t * z1
            batch_z = np.vstack((batch_z, zt))

    z = torch.from_numpy(batch_z).float().to(device)
    path = generator(z).data.cpu().numpy().squeeze(1)
    path = (path + 1) / 2  # The pixels are now in [0, 1]

    # We show and store the nonlinear interpolation in the image space
    ##################################################################
    dir_path = os.path.join(dir_to_save, dir_save_images)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for idx_image in range(num_samples):
        current_image = np.uint8(path[idx_image] * 255.0)
        filename = os.path.join(dir_path, '{}.png'.format(idx_image))
        Image.fromarray(current_image).save(filename)
        plt.imshow(current_image, cmap='gray')
        plt.axis('off')
        plt.pause(0.1)
        plt.draw()
    """
