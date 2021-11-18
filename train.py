import os
import time
import copy
import logging
import numpy as np

import torch
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.utils as vutils
from torchvision.utils import save_image

from evaluate import *

def load_ckpt(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model

def save_checkpoint(state, filename):
    """Save checkpoint if a new best is achieved"""
    print ("=> Saving a new best")
    torch.save(state, filename)  # save checkpoint
    

def train_all(netG, netD, imgSize, variational_beta, cvae_batch_size, optimizerG, optimizerD, recon_loss, cls_loss, dataset, train_loader, val_loader, test_loader, Gepoch, Depoch, channel, device):
    
    logger = logging.getLogger()

    best_loss = np.inf
    best_loss2 = np.inf
    
    
    ################################################################################
    # train CVAE
    ################################################################################
    for epoch in range(Gepoch):
        loss = []
        netG.train()
        for i, (images,_) in enumerate(train_loader):
            images = images.to(device)
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            optimizerG.zero_grad()
            L_dec_vae = recon_loss(recon_x, images, mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)
            L_dec_vae.backward()
            optimizerG.step()      
            loss.append(L_dec_vae.item())
        
            if i % 100 == 0:
                vutils.save_image(images,
                                './logs/'+dataset+'/real_samples_'+dataset+'.png',normalize=True)
                vutils.save_image(recon_x.data.view(-1,channel,imgSize,imgSize),
                                './logs/'+dataset+'/fake_samples_'+dataset+'.png',normalize=True)

        logger.info("Epoch:%d Trainloss: %.8f"%(epoch, np.mean(loss)))
    
        loss = []
        netG.eval()
        for i, (images,_) in enumerate(val_loader):
            images = images.cuda()
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            L_dec_vae = recon_loss(recon_x, images, mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)
            loss.append(L_dec_vae.item())
            L_dec_vae = recon_loss(recon_x, images, mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)
        
        logger.info("Epoch:%d   Valloss: %.8f"%(epoch, np.mean(loss)))
        if np.mean(loss)<best_loss:
            best_loss = np.mean(loss)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': netG.state_dict(),
                    'optimizer_state_dict': optimizerG.state_dict(),
                    }, "./weights/"+dataset+"/netG_"+dataset+".pth.tar")
            
        cvae_evaluate(netG, recon_loss, test_loader, device, variational_beta, imgSize, channel, cvae_batch_size)

    ###############################################################################
    # train Discriminator
    ################################################################################    
        
    logger.info("--------CLS--------")    
    cls_loss = torch.nn.BCELoss()    
    netG.eval() 

    for epoch in range(Depoch):
        loss = []
        netD.train()
    
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            preds = netD(images)
            preds2 = netD(recon_x)
        
            optimizerD.zero_grad()
            L_dec_vae = cls_loss(torch.squeeze(preds, dim=1),targets.float())
            L_dec_vae += cls_loss(torch.squeeze(preds2, dim=1),1.0-targets)
            L_dec_vae.backward()
            optimizerD.step()      
            loss.append(L_dec_vae.item())
   
        logger.info("Epoch:%d Trainloss: %.8f"%(epoch, np.mean(loss)))
        
        loss = []
        netD.eval()
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            preds = netD(images)
            preds2 = netD(recon_x)

            L_dec_vae = cls_loss(torch.squeeze(preds, dim=1),targets.float())
            L_dec_vae += cls_loss(torch.squeeze(preds2, dim=1),1.0-targets)      
            loss.append(L_dec_vae.item())
         
        logger.info("Epoch:%d   Valloss: %.8f"%(epoch, np.mean(loss))) 
        if np.mean(loss)<best_loss2:
            best_loss2 = np.mean(loss)
            torch.save({
                'epoch': epoch,
                'model_state_dict': netD.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict(),
                }, "./weights/"+dataset+"/netD_"+dataset+".pth.tar")
            
        cvad_evaluate(netG, netD, recon_loss, cls_loss, test_loader, device, variational_beta, imgSize, channel, cvae_batch_size)



