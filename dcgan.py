#prerequisites
# Taken from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np

from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from block_indexer import BlockIndexer
from torch.utils.data import Subset
from pdb import set_trace

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cuda = torch.cuda.is_available()
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

bs = 64
block_size = 200
img_shape = (1, 28, 28)
z_dim = 3969
N_CLASSES = 10
#z_dim = 100

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)
#idx = train_dataset.train_labels==0
#train_dataset.targets = train_dataset.targets[idx]
#train_dataset.data = train_dataset.data[idx]
Ds = pickle.load(open('files/Ds_mnist_inf.pkl', 'rb'))
sig_bi = BlockIndexer(block_size, [10])

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #self.label_embedding = nn.Embedding(N_CLASSES, N_CLASSES)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(z_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        #z_in = torch.cat((z.view(z.size(0), -1), self.label_embedding(labels)), -1)
        z_in = z
        img = self.model(z_in)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(N_CLASSES, N_CLASSES)

        self.model = nn.Sequential(
            nn.Linear(N_CLASSES + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        #img_flat = img.view(img.shape[0], -1)
        validity = self.model(d_in)
        return validity

def sample_z(x_lab):
    n = Ds.shape[1]
    z = list()
    for i in range(x_lab.shape[0]):
        cs = np.zeros(n, dtype=np.float32)
        cs_i = np.random.laplace(size=block_size)
        #cs_i = np.zeros(block_size, dtype=np.float32)
        #cs_i[np.random.randint(block_size)] = 1.0
        try:
            cs = sig_bi.set_block(cs, x_lab[i].numpy(), cs_i)
        except:
            set_trace()
        z_i = Ds @ cs
        z.append(z_i)
    z = np.vstack(z)
    z = Variable(torch.from_numpy(z).to(device))
    return z

def main():
    # build network
    #mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)
    mnist_dim = 784
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    lr = 0.00005
    n_epochs = 200
    clip_value = 0.01
    n_critic = 5
    sample_interval = 1000
    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(n_epochs):

        for i, (imgs, img_labs) in enumerate(train_loader):

            #print(img_labs)
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = sample_z(img_labs)
            #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], z_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, N_CLASSES, img_labs.shape[0])))
            # Generate a batch of images
            fake_imgs = generator(z, gen_labels).detach()

            if cuda:
                img_labs = img_labs.cuda()
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs, img_labs)) + torch.mean(discriminator(fake_imgs, gen_labels))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # Train the generator every n_critic iterations
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)
                if cuda:
                    gen_labels = gen_labels.cuda()
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs, gen_labels))

                loss_G.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, batches_done % len(train_loader), len(train_loader), loss_D.item(), loss_G.item())
                )

            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], "wgan_gen/scatter_images/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += 1


    gen_save_path = 'files/cwganfull_200_mnist_gen.pth'
    discrim_save_path = 'files/cwganfull_200_mnist_discrim.pth'
    torch.save(generator.state_dict(), gen_save_path)
    torch.save(discriminator.state_dict(), discrim_save_path)
    print("Saved generator + discriminator to {}".format(gen_save_path))

def test():
    decoder = Generator()
    decoder.load_state_dict(torch.load('files/cwgan_200_mnist_gen.pth', map_location=torch.device('cpu')))
    decoder.cuda()
    img_labs = torch.from_numpy(np.array([0 for i in range(25)]))
    z = sample_z(img_labs)
    gen_imgs = decoder(z, None).detach()
    set_trace()
    save_image(gen_imgs.data[:25], "wgan_gen/scatter_images/c0_images1sparse.png", nrow=5, normalize=True)

if __name__=='__main__':
    #main()
    test()
