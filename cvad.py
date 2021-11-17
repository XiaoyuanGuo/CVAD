from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
import numpy as np
from numpy import sqrt, argmax
from sklearn.metrics import roc_curve
from sklearn.metrics import auc, roc_auc_score
from datasets.build_dataset import *

from networks.CVAE import CVAE
from models.modules_tied_2vaegan_cifar10 import VAE_all32, Discriminator32
from networks.CVAE import Discriminator

from torch.utils.data import DataLoader
from datasets.IVCFilter import IVCFilter_Dataset, get_ivc_anomaly_dataset
from datasets.SIIM import SIIM_Dataset, get_siim_data
from vaegan.rsna import PneumoniaDataset,  get_rsna_data
from utils.cvaegan_utils import *
from utils.plot_loss import *

    
# load_pretrained = False    
load_pretrained = True    
imgSize = 256
latent_dims = 512
bsz = 32
channel = 3
criterion = nn.BCELoss()
capacity = 16
extra_test = True
Gepoch = 100
Depoch = 0


# dataset = "cifar10"
dataset = "ivc-filter"
# dataset = "siim"
# dataset = "rsna"
# dataset = "mnist"
# dataset = "bird"

print("dataset: ", dataset)

if dataset == "siim":
    normal_class = [0]
    channel = 3
    netG = VAE_all(capacity,channel)
    netD = Discriminator(capacity, channel)
    Gepoch = 50
    Depoch = 50
    load_pretrained = True
elif dataset == "ivc-filter":
#     load_pretrained = False
    normal_class = [11]
    channel = 1
    bsz = 64
    netG = VAE_all(capacity,channel)
    netD = Discriminator(capacity, channel)
    Gepoch = 150
    Depoch = 150
elif dataset == "rsna":
    normal_class = [0]
    channel = 1
    netG = VAE_all(capacity,channel)
    netD = Discriminator(capacity, channel)
elif dataset == "cifar10":
    normal_class = [0]
    imgSize = 32
    latent_dims = 64
    bsz = 1024
    extra_test = False
    netG = VAE_all32(capacity,channel)
    netD = Discriminator32(capacity, channel)

elif dataset == "bird":
    normal_class = [0]
    bsz = 64
    extra_test = False
    netG = VAE_all(capacity,channel)
    netD = Discriminator(capacity, channel)
    Gepoch = 150
    Depoch = 50
    load_pretrained = False
    
elif dataset == "mnist":
    normal_class = [0]
    imgSize = 32
    channel = 1
    latent_dims = 16
    bsz = 1024
    extra_test = False
#     load_pretrained = False
    netG = VAE_all32(capacity,channel)
    netD = Discriminator32(capacity, channel)
    Gepoch = 3
    Depoch = 10
    
netG = netG.cuda()
optimizerG = optim.Adam(netG.parameters(), lr=1e-4)

netD = netD.cuda()
optimizerD = optim.Adam(netD.parameters(), lr=1e-4)


if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

best_loss = np.inf
best_loss2 = np.inf


best_auc = 0.0
best_auc2 = 0.0

if load_pretrained == True:
    checkpoint = torch.load("./weights/"+dataset+"/netG_"+dataset+".pth.tar",map_location='cpu')
    netG.load_state_dict(checkpoint['model_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])
    best_auc = checkpoint['best_auc']
    print("best auc: ", best_auc)

    checkpoint = torch.load("./weights/"+dataset+"/netD_"+dataset+".pth.tar",map_location='cpu')
    netD.load_state_dict(checkpoint['model_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizer_state_dict'])
    best_auc2 = checkpoint['best_auc2']
    print("best auc2: ", best_auc2)

data_loaders, _ = build_ae_dataset(dataset, "../DATA/", bsz, normal_class)
train_loader = data_loaders['train']
val_loader = data_loaders['val']
test_loader = data_loaders['test']


def train_eval(netG, netD, optimizerG, optimizerD, dataset, train_loader, val_loader, test_loader, Gepoch, Depoch, best_auc, best_auc2, bsz, imgSize, channel ):
    
    train_loss = []
    val_loss = []
    
    train_loss2 = []
    val_loss2 = []
    
    for epoch in range(Gepoch):
        loss = []
        netG.train()
        for i, (images,_) in enumerate(train_loader):
            images = images.cuda()
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            optimizerG.zero_grad()
            L_dec_vae = loss_function2(recon_x, images, mu, logvar, mu2, logvar2, imgSize, channel, bsz)
            L_dec_vae.backward()
            optimizerG.step()      
            loss.append(L_dec_vae.item())
        
            if i % 100 == 0:
                vutils.save_image(images[0:32,:,:,:],
                                './logs/'+dataset+'/real_samples_'+dataset+'.png',normalize=True)
                vutils.save_image(recon_x.data.view(-1,channel,imgSize,imgSize)[0:32,:,:,:],
                                './logs/'+dataset+'/fake_samples_'+dataset+'.png',normalize=True)
        
        train_loss.append(np.mean(loss))
        print("Epoch:%d Trainloss: %.8f"%(epoch, np.mean(loss)))
    
        loss = []
        netG.eval()
        for i, (images,_) in enumerate(val_loader):
            images = images.cuda()
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            L_dec_vae = loss_function2(recon_x, images, mu, logvar, mu2, logvar2, imgSize, channel, bsz)
            loss.append(L_dec_vae.item())
            L_dec_vae = loss_function2(recon_x, images, mu, logvar, mu2, logvar2, imgSize, channel, bsz)
        
        val_loss.append(np.mean(loss))
        print("Epoch:%d   Valloss: %.8f"%(epoch, np.mean(loss)))  
        aucscore1, fpr1, tpr1 = cvae_evaluate(netG, test_loader, imgSize, channel, bsz)
        
        
        if aucscore1 > best_auc:
#         if np.mean(loss) < best_loss:
#             best_loss = np.mean(loss)
            best_auc = aucscore1
            print("Save a new netG! ")
            torch.save({
                'epoch': epoch,
#                 'best_loss': best_loss,
                'best_auc': best_auc,
                'model_state_dict': netG.state_dict(),
                'optimizer_state_dict': optimizerG.state_dict(),
                }, "./weights/"+dataset+"/netG_"+dataset+".pth.tar")

    print("**********CLS**********")    
    cls_loss = torch.nn.BCELoss()    
    netG.eval() 
    
    for epoch in range(Depoch):
        loss = []
        netD.train()
    
        for i, (images, targets) in enumerate(train_loader):
            images = images.cuda()
            targets = targets.cuda()
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            optimizerD.zero_grad()
            
#             preds = netD(images)
#             preds2 = netD(recon_x)
#             L_dec_vae = cls_loss(torch.squeeze(preds, dim=1),targets.float())
#             L_dec_vae += cls_loss(torch.squeeze(preds2, dim=1),1.0-targets)
            
            new_inputs = torch.cat((images, recon_x), 0)
            new_preds = netD(new_inputs)
            new_targets = torch.cat((targets.float(), 1.0-targets), 0)
            L_dec_vae = cls_loss(torch.squeeze(new_preds, dim=1), new_targets)
            
            L_dec_vae.backward()
            optimizerD.step()      
            loss.append(L_dec_vae.item())
        
        train_loss2.append(np.mean(loss))
        print("Epoch:%d Trainloss: %.8f"%(epoch, np.mean(loss)))
        
        loss = []
        netD.eval()
        for i, (images, targets) in enumerate(val_loader):
            images = images.cuda()
            targets = targets.cuda()
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            
#             preds = netD(images)
#             preds2 = netD(recon_x)
#             L_dec_vae = cls_loss(torch.squeeze(preds, dim=1),targets.float())
#             L_dec_vae += cls_loss(torch.squeeze(preds2, dim=1),1.0-targets)      
#             loss.append(L_dec_vae.item())
            
            new_inputs = torch.cat((images, recon_x), 0)
            new_preds = netD(new_inputs)
            new_targets = torch.cat((targets.float(), 1.0-targets), 0)
            L_dec_vae = cls_loss(torch.squeeze(new_preds, dim=1), new_targets)
            loss.append(L_dec_vae.item())

            
        
        val_loss2.append(np.mean(loss))        
        print("Epoch:%d   Valloss: %.8f"%(epoch, np.mean(loss)))  
        aucscore2, fpr2, tpr2 = cvae_evaluate2(netG, netD, cls_loss, test_loader, imgSize, channel, bsz)
        
#         if np.mean(loss) < best_loss2:
        if aucscore2 > best_auc2:
#             best_loss2 = np.mean(loss)
            best_auc2 = aucscore2
            print("Save a new netD! ")
            torch.save({
                'epoch': epoch,
#                 'best_loss2': best_loss2,
                'best_auc2': best_auc2,
                'model_state_dict': netD.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict(),
                }, "./weights/"+dataset+"/netD_"+dataset+".pth.tar")
                
    strclass = ""
    for c in normal_class:
        strclass += str(c)
    plot_loss(train_loss, val_loss, train_loss2, val_loss2, strclass, dataset)
    


train_eval(netG, netD, optimizerG, optimizerD, dataset, train_loader, val_loader, test_loader, Gepoch, Depoch, best_auc, best_auc2, bsz, imgSize, channel)       
        
