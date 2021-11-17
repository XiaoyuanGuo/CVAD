"""
CVAD's discriminator
The main CVAD model implementation.
Written by Xiaoyuan Guo
"""

import torch
from torch import nn

imgSize = 256

class Discriminator(nn.Module):
    def __init__(self, channel=3, capacity = 16):
        super(Discriminator, self).__init__()
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
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.c*16*(imgSize//32)*(imgSize//32), out_features=latent_dims, bias=False),
            nn.BatchNorm1d(num_features=latent_dims,momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=latent_dims, out_features=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        xt = x
        x = self.conv5(x)
        x = x.view(len(x), -1)
        x = self.fc(x)
        return  xt, torch.sigmoid(x)
