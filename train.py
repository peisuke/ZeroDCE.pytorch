import PIL
from PIL import Image
PIL.PILLOW_VERSION = PIL.__version__

import torch
import torch.nn as nn
from torch.nn import functional as F
import glob
import numpy as np
from torchvision import transforms 
import torch.optim as optim
import torch
import glob
import pytorch_lightning as pl

class DataSet(torch.utils.data.Dataset):
    def __init__(self, filelist, transform=None):
        self.transform = transform
        self.filelist = filelist
        self.nb_data = len(filelist)
        
    def __len__(self):
        return self.nb_data

    def __getitem__(self, idx):
        filename = self.filelist[idx]
        img = Image.open(filename)

        if self.transform:
            inp = self.transform(img)

        return inp

class DCENet(pl.LightningModule):
    def __init__(self, nb_iter):
        super(DCENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(64, 3 * nb_iter, 3, stride=1, padding=1)

    def forward(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h5 = F.relu(self.conv5(torch.cat((h4, h3), dim=1)))
        h6 = F.relu(self.conv6(torch.cat((h5, h2), dim=1)))
        h7 = F.relu(self.conv7(torch.cat((h6, h1), dim=1)))

        return h7
    
    def LE(self, inp, A):
        return inp + A * inp * (1 - inp)

    def refine_image(self, x, A):
        An = torch.split(A, 3, 1)
        for A in An:
            x = self.LE(x, A)
        return x
    
    def tv_loss(self, img):
        w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
        h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
        return h_variance + w_variance

    def convert_to_gray(self, inp):
        to_gray = torch.tensor([0.3, 0.59, 0.11]).reshape(1, 3, 1, 1)
        to_gray = to_gray.type_as(inp)
        
        gray = F.conv2d(inp, to_gray)
        return gray

    def exp_loss(self, inp):
        gray = self.convert_to_gray(inp)
        E = 0.6
        Y = F.avg_pool2d(gray, 16, 16)
        M = np.asarray(Y.shape).sum()
        return (Y - E).abs().sum() / M

    def col_loss(self, inp):
        J = inp.mean(dim=(2,3))
        l0 = torch.pow((J[:,0] - J[:,1]), 2)
        l1 = torch.pow((J[:,1] - J[:,2]), 2)
        l2 = torch.pow((J[:,2] - J[:,0]), 2)
        return ((l0 + l1 + l2) / 3).sum()

    def spa_loss(self, x, y):
        K = torch.tensor(
            [[[0,-1,0], [0,1,0], [0,0,0]],
            [[0,0,0], [0,1,0], [0,-1,0]],
            [[0,0,0], [-1,1,0], [0,0,0]],
            [[0,0,0], [0,1,-1], [0,0,0]]], dtype=torch.float32)
        K = K.reshape(4, 1, 3, 3)
        K = K.type_as(x)
        
        xg = self.convert_to_gray(x)
        yg = self.convert_to_gray(y)

        xd = F.conv2d(xg, K).abs()
        yd = F.conv2d(yg, K).abs()

        return torch.pow(xd - yd, 2).mean()
    
    def total_loss(self, x, y, A):
        loss = self.spa_loss(x, y) + self.exp_loss(y) + self.col_loss(y) + self.tv_loss(A)
        return loss
    
    def training_step(self, batch, batch_idx):
        x = batch
        A = self.forward(x)
        y = self.refine_image(x, A)        
        loss = self.total_loss(x, y, A)
        tensorboard_logs = {'train_loss': loss}
        
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x = batch
        A = self.forward(x)
        y = self.refine_image(x, A)        
        loss = self.total_loss(x, y, A)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer
    
    @pl.data_loader
    def train_dataloader(self):
        filenames = sorted(glob.glob('../dataset/train/*.jpg'))

        transform = transforms.Compose([
            transforms.RandomResizedCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((512, 512)),
            transforms.ToTensor()])
        
        dataset = DataSet(filenames, transform)
        
        return torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    @pl.data_loader
    def val_dataloader(self):
        filenames = sorted(glob.glob('../dataset/test/*.jpg'))

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])
        
        dataset = DataSet(filenames, transform)
        
        return torch.utils.data.DataLoader(dataset, batch_size=8)

if __name__ == '__main__':
    nb_iter = 8
    model = DCENet(nb_iter=nb_iter)
    trainer = pl.Trainer(gpus=1, max_epochs=50)
    trainer.fit(model)
