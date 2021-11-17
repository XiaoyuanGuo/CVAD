import numpy
import numpy as np

import torch
from torch import nn


imgSize = 32
channel = 3
latent_dims = 16

class CVAE32(nn.Module):
    def __init__(self,  capacity=16, channel=3):
        super(CVAE32, self).__init__()
        self.c = capacity
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=self.c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.c*2, out_channels=self.c*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.c*4, out_channels=self.c*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc_mu = nn.Linear(in_features=self.c*8*(imgSize//16)*(imgSize//16), out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=self.c*8*(imgSize//16)*(imgSize//16), out_features=latent_dims)
        
        self.fc_mu2 = nn.Linear(in_features=self.c*4*(imgSize//4)*(imgSize//4), out_features=latent_dims*4)
        self.fc_logvar2 = nn.Linear(in_features=self.c*4*(imgSize//4)*(imgSize//4), out_features=latent_dims*4)
        
        self.fc = nn.Linear(in_features=latent_dims, out_features=self.c*8*(imgSize//16)*(imgSize//16))
        
        self.fc2 = nn.Linear(in_features=latent_dims*4, out_features=self.c*2*(imgSize//4)*(imgSize//4))
        
        self.dconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.c*8, out_channels=self.c*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.c*4, out_channels=self.c*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.c, out_channels=channel, kernel_size=4, stride=2, padding=1)
        )
        
        self.dconv22 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dconv12 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.c, out_channels=channel, kernel_size=4, stride=2, padding=1)
        )
    
    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu  
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_enc = x
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        latent = self.latent_sample(x_mu, x_logvar)
        x = self.fc(latent)
        x = x.view(-1, self.c*8, imgSize//16, imgSize//16)
        x = self.dconv4(x)
        x = self.dconv3(x)
        x_enc2 = torch.cat([x, x_enc], dim = 1)
        x = self.dconv2(x)
        x = self.dconv1(x)
        x = torch.sigmoid(x)
                
        x_enc2 = x_enc2.view(x_enc2.size(0), -1)
        x_mu2 = self.fc_mu2(x_enc2)
        x_logvar2 = self.fc_logvar2(x_enc2)
        
        latent2 = self.latent_sample(x_mu2, x_logvar2)
        x_2 = self.fc2(latent2)
        x_2 = x_2.view(-1, self.c*2, imgSize//4, imgSize//4)
        x_2 = self.dconv22(x_2)
        x_l = x_2
        x_l = x_l.view(1,-1)
        x_2 = self.dconv12(x_2)
        x_2 = torch.sigmoid(x_2)
        x_final = 0.5*(x+x_2)
        return x_final, x_mu, x_logvar, x_mu2, x_logvar2


class Discriminator32(nn.Module):
    def __init__(self, capacity=16, channel=3):
        super(Discriminator32, self).__init__()
        self.c = capacity
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=self.c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.c*2, out_channels=self.c*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.c*4, out_channels=self.c*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.linear = nn.Sequential(
            nn.Linear(self.c*8*(imgSize//16)*(imgSize//16), latent_dims),
            nn.Linear(latent_dims, latent_dims//2),
            nn.Linear(latent_dims//2, 1)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x
