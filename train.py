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
    



def train_cls_model(net_name, cls_model, cls_loss, cls_optimizer, cls_num_epochs, warmup_epochs, cls_dataloaders, cls_dataset_sizes, dataset_name, device, R=150, c=None):    
    assert dataset_name in ['mnist', 'cifar10', 'bird', 'rsna', 'ivc-filter', 'siim']
    logger = logging.getLogger()
    logger.info("---------Stage-2 TEND training----------")
    since = time.time()
    best_loss = np.inf
    best_fpr = np.inf
    trainloss = []
    valloss = []

    for epoch in range(cls_num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, cls_num_epochs))
        for phase in ['train','val']:
            if phase == 'train':
                cls_model.train()
                if epoch == warmup_epochs and c == None:
                    c = init_center_c(net_name, cls_dataloaders[phase], dataset_name, cls_model, device)
            else:
                cls_model.eval()
            running_loss= 0.0
                       
            for idx, inputs in enumerate(cls_dataloaders[phase]):
                
                cls_optimizer.zero_grad()
                with torch.set_grad_enabled(phase =='train'):
                    if phase == 'train':
                        data, targets = inputs
                        org_imgs, tfm_imgs = data
                        org_imgs, tfm_imgs = org_imgs.to(device), tfm_imgs.to(device)
                    
                        org_targets, tfm_targets = targets
                        org_targets, tfm_targets = org_targets.to(device), tfm_targets.to(device)
                    
                        all_imgs = torch.cat([org_imgs, tfm_imgs], dim=0)
                        all_targets = torch.cat([org_targets, tfm_targets], dim=0)
                        all_targets = torch.unsqueeze(all_targets, dim=1).float()
                        
                        preds = cls_model(all_imgs)
                        loss = cls_loss(preds, all_targets) 
                            
                        if isinstance(cls_model,torch.nn.DataParallel):
                            cls_model = cls_model.module

                        if epoch >= warmup_epochs:
                            outputs = cls_model.get_embedding(org_imgs)
                            dist = torch.sum((outputs - c) ** 2, dim=1)
                            loss += torch.mean(dist)
                    
                            toutputs = cls_model.get_embedding(tfm_imgs)
                            tdist = torch.sum((toutputs - c) ** 2, dim=1)
                            loss += torch.mean(torch.nn.functional.relu(R-tdist))
                
                        loss.backward()
                        cls_optimizer.step()
                    else:
                        data, targets = inputs
                        all_imgs = data.to(device)
                        all_targets = targets.to(device)
                        all_targets = torch.unsqueeze(all_targets, dim=1).float()
                        
                        preds = cls_model(all_imgs)
                        loss = cls_loss(preds, all_targets) 
                        if isinstance(cls_model,torch.nn.DataParallel):
                            cls_model = cls_model.module
                        if epoch >= warmup_epochs:
                            nrm_indices = torch.nonzero((targets == 0))
                            nrm_imgs = torch.index_select(data, 0, nrm_indices.squeeze(1))
                            
                            ood_indices = torch.nonzero((targets == 1))
                            ood_imgs = torch.index_select(data, 0, ood_indices.squeeze(1))
                            
                            nrm_imgs, ood_imgs = nrm_imgs.to(device), ood_imgs.to(device)
                            
                            if nrm_indices.shape[0] != 0:
                                nrm_outputs = cls_model.get_embedding(nrm_imgs)
                                nrm_dist = torch.sum((nrm_outputs - c) ** 2, dim=1)
                                loss += torch.mean(nrm_dist)
                    
                            if ood_indices.shape[0] != 0:
                                ood_outputs = cls_model.get_embedding(ood_imgs)
                                ood_dist = torch.sum((ood_outputs - c) ** 2, dim=1)
                                loss += torch.mean(torch.nn.functional.relu(R-tdist))
                        
                running_loss += loss.item() * all_imgs.shape[0]
            epoch_loss = running_loss / cls_dataset_sizes[phase]
            
            logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'train':
                trainloss.append(epoch_loss)
            elif phase == 'val' and epoch >= warmup_epochs+1: 
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_cls_model_wts = copy.deepcopy(cls_model.state_dict())
                    save_checkpoint(state={'epoch': epoch, 
                                      'model_state_dict': cls_model.state_dict(),
                                      'best_loss':best_loss,
                                      'optimizer_state_dict': cls_optimizer.state_dict()},
                                filename = './weights/'+dataset_name+"/tend_"+net_name+".pt")
                    print()
    time_elapsed = time.time()-since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    logger.info('Best val loss:{:4f}'.format(best_loss))
            
    cls_model.load_state_dict(best_cls_model_wts)
    return cls_model, c



def train_all(netG, netD, imgSize, optimizerG, optimizerD, recon_loss, cls_loss, dataset, train_loader, val_loader, test_loader, Gepoch, Depoch,channel):
    
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
            images = images.cuda()
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            optimizerG.zero_grad()
            L_dec_vae = recon_loss(recon_x, images, mu, logvar, mu2, logvar2)
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
            L_dec_vae = recon_loss(recon_x, images, mu, logvar, mu2, logvar2)
            loss.append(L_dec_vae.item())
            L_dec_vae = recon_loss(recon_x, images, mu, logvar, mu2, logvar2)
        
        logger.info("Epoch:%d   Valloss: %.8f"%(epoch, np.mean(loss)))
        if np.mean(loss)<best_loss:
            best_loss = np.mean(loss)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': netG.state_dict(),
                    'optimizer_state_dict': optimizerG.state_dict(),
                    }, "./weights/"+dataset+"/netG_"+dataset+".pth.tar")
            
        cvae_evaluate(netG, test_loader, recon_loss, device)

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
            images = images.cuda()
            targets = targets.cuda()
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
            images = images.cuda()
            targets = targets.cuda()
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
            
        cvad_evaluate(netG, netD, recon_loss, cls_loss, test_loader, device)



