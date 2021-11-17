"""
CVAD
The main CVAD model implementation.
Written by Xiaoyuan Guo
"""
import torch
from torch import nn

imgSize = 256
channel = 1
latent_dims = 512

############################################################
# CVAE architecture
############################################################


class CVAE(nn.Module):
    def __init__(self,  capacity=16, channel=3):
        super(CVAE, self).__init__()
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
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.c*8, out_channels=self.c*16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c*16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc_mu = nn.Linear(in_features=self.c*16*(imgSize//32)*(imgSize//32), out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=self.c*16*(imgSize//32)*(imgSize//32), out_features=latent_dims)
        
        self.fc_mu2 = nn.Linear(in_features=self.c*8*(imgSize//8)*(imgSize//8), out_features=latent_dims*4)
        self.fc_logvar2 = nn.Linear(in_features=self.c*8*(imgSize//8)*(imgSize//8), out_features=latent_dims*4)
        
        self.fc = nn.Linear(in_features=latent_dims, out_features=self.c*16*(imgSize//32)*(imgSize//32))
        self.fc2 = nn.Linear(in_features=latent_dims*4, out_features=self.c*4*(imgSize//8)*(imgSize//8))
        
        self.dconv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.c*16, out_channels=self.c*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
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
        
        self.dconv32 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.c*4, out_channels=self.c*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.c*2),
            nn.LeakyReLU(0.2, inplace=True),
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
        x = self.conv3(x)
        x_enc = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        latent = self.latent_sample(x_mu, x_logvar)
        x = self.fc(latent)
        x = x.view(-1, self.c*16, imgSize//32, imgSize//32)
        x = self.dconv5(x)
        x = self.dconv4(x)
        x_enc2 = torch.cat([x, x_enc], dim = 1)
        x = self.dconv3(x)
        x = self.dconv2(x)
        x = self.dconv1(x)
        x = torch.sigmoid(x)
                
        x_enc2 = x_enc2.view(x_enc2.size(0), -1)
        x_mu2 = self.fc_mu2(x_enc2)
        x_logvar2 = self.fc_logvar2(x_enc2)
        
        latent2 = self.latent_sample(x_mu2, x_logvar2)
        x_2 = self.fc2(latent2)
        x_2 = x_2.view(-1, self.c*4, imgSize//8, imgSize//8)
        x_2 = self.dconv32(x_2)
        x_2 = self.dconv22(x_2)
        x_l = x_2
        x_l = x_l.view(1,-1)
        x_2 = self.dconv12(x_2)
        x_2 = torch.sigmoid(x_2)
        
        x_final = 0.5*(x+x_2)
        
        return x_final, x_mu, x_logvar, x_mu2, x_logvar2
