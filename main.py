import os
import re
import time
import copy
import click
import logging
import numpy as np
from pathlib import Path

import torch
from torch import nn

from utils.config import Config
from utils.cvad_loss import recon_loss

from datasets.build_dataset import *
from networks.build_net import *
from train import train_all, load_ckpt



################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['cifar10', 'siim']))
@click.argument('net_name', type=click.Choice(['CVAD']))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--capacity', type=int, default=16, help='Specify Convoluation layer channel unit')
@click.option('--channel', type=int, default=1, help='Specify image channel, grayscale images for 1, RGB images for 3')
@click.option('--cvae_batch_size', type=int, default=64, help='Batch size for mini-batch training.')
@click.option('--cvae_n_epochs', type=int, default=250, help='Stage-1 autoencoder training epochs')
@click.option('--cvae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for autoencoder network training.')
@click.option('--cvae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder network training. Default=0.001')
@click.option('--cvae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--variational_beta', type=float, default=1.0, help='For CVAE loss')
@click.option('--load_cvae_model', type=bool, default=False, help='Whether load previous trained model')
@click.option('--cvae_model_path', type=click.Path(exists=True), default="./weights/", help='Model file path')
@click.option('--cls_batch_size', type=int, default=64, help='Batch size for CVAD training.')
@click.option('--cls_n_epochs', type=int, default=10, help='Stage-2 CVAD classification training epochs')
@click.option('--cls_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for CVAD discriminator network training.')
@click.option('--cls_lr', type=float, default=0.001,
              help='Initial learning rate for CVAD classification discriminator training. Default=0.001')
@click.option('--cls_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for tend.')
@click.option('--load_cls_model', type=bool, default=False, help='Whether load previous trained model')
@click.option('--cls_model_path', type=click.Path(exists=True), default="./weights/", help='Model file path')
@click.option('--normal_class', type=str, default="0",
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--load_config', type=bool, default=False, help='Whether use previous log')
@click.option('--config_path', type=click.Path(exists=True), default="./logs/",
              help='Config JSON-file path (default: None).')



def main(dataset_name, net_name, data_path, capacity, channel, cvae_batch_size, cvae_n_epochs, cvae_optimizer_name, cvae_lr, cvae_weight_decay, 
         variational_beta, load_cvae_model, cvae_model_path, cls_batch_size, cls_n_epochs, cls_optimizer_name, cls_lr, cls_weight_decay, 
         load_cls_model, cls_model_path, normal_class, load_config, config_path):
    
    # Get configuration
    cfg = Config(locals().copy())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
                                               
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = "./logs/"+ dataset_name +'/log_'+cfg.settings['net_name']+"_"+cfg.settings['normal_class']+".txt"
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    if not os.path.exists("./logs/"+ dataset_name):
        os.mkdir("./logs/"+ dataset_name)
    if os.path.exists(log_file):
        os.remove(log_file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    cvae_model_path = cfg.settings['cvae_model_path']+dataset_name+"/netG_"+dataset_name+".pt.tar"
    cls_model_path = cfg.settings['cls_model_path']+dataset_name+"/netD_"+dataset_name+".pt.tar"
    config_path = cfg.settings['config_path']+dataset_name+"/config_"+cfg.settings['net_name']+"_"+cfg.settings['normal_class']+".json"
    
    # Print arguments
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % cfg.settings['data_path'])
    logger.info('Dataset: %s' % cfg.settings['dataset_name'])
    logger.info('Net Name: %s'% cfg.settings['net_name'])
    logger.info('CVAE Conv capacity: %d' % cfg.settings['capacity'])
    logger.info('image channel: %d' % cfg.settings['channel'])
    logger.info('------------Stage-1--------------')
    logger.info('CVAE batchsize: %d' % cfg.settings['cvae_batch_size'])
    logger.info('CVAE epochs: %d' % cfg.settings['cvae_n_epochs'])
    logger.info('CVAE optimizer: %s' % cfg.settings['cvae_optimizer_name'])
    logger.info('CVAE lr: %f' % cfg.settings['cvae_lr'])
    logger.info('CVAE weight_decay: %f' % cfg.settings['cvae_weight_decay'])
    logger.info('CVAE load_model: %r' % cfg.settings['load_cvae_model'])
    logger.info('CVAE variational_beta: %f' % cfg.settings['variational_beta'])
    logger.info('CVAE model_path %s' % cvae_model_path)
    logger.info('------------Stage-2--------------')
    logger.info('CVAD batchsize: %d' % cfg.settings['cls_batch_size'])
    logger.info('CVAD epochs: %d' % cfg.settings['cls_n_epochs'])
    logger.info('CVAD optimizer: %s' % cfg.settings['cls_optimizer_name'])
    logger.info('CVAD lr: %f' % cfg.settings['cls_lr'])
    logger.info('CVAD weight_decay: %f' % cfg.settings['cls_weight_decay'])
    logger.info('CVAD load_model: %r' %  cfg.settings['load_cls_model'])
    logger.info('CVAD model_path: %s' % cls_model_path)
    logger.info('Normal class:' + cfg.settings['normal_class'])    
    logger.info("CVAE ==> embnet")
    logger.info("CVAD ==> cls_model")
    
    imgSize = 256
    if dataset_name == "cifar10":
        imgSize = 32
        channel = 3



    normal_class = re.findall(r'\d+', cfg.settings['normal_class'])
    normal_class = [int(x) for x in normal_class]
    cvae_dataloaders, cvae_dataset_sizes = build_cvae_dataset(cfg.settings['dataset_name'], cfg.settings['data_path'], cfg.settings['cvae_batch_size'], normal_class)
#     test_dataloaders, test_dataset_sizes = build_intertest_dataset(cfg.settings['dataset_name'], cfg.settings['data_path'],  cfg.settings['cvae_batch_size'], [0])
    embnet, cls_model = build_CVAD_net(cfg.settings['dataset_name'], cfg.settings['net_name'], cfg.settings['capacity'], cfg.settings['channel'])
    embnet = embnet.to(device)
    cls_model = cls_model.to(device)
                                               
    amsgrad = False
    if cfg.settings['cvae_optimizer_name'] == "amsgrad":
        amsgrad=True
    cvae_optimizer = torch.optim.Adam(embnet.parameters(), lr=cfg.settings['cvae_lr'], weight_decay=cfg.settings['cvae_weight_decay'], amsgrad=amsgrad)
    
    if load_cvae_model: # load pretrained model
        logger.info("---------CVAE load trained model--------")
        embnet = load_ckpt(cvae_model_path, embnet, cvae_optimizer)   
         
    cls_loss = nn.BCELoss()
    amsgrad = False
    if cfg.settings['cls_optimizer_name'] == "amsgrad":
        amsgrad=True
    cls_optimizer = torch.optim.Adam(cls_model.parameters(), lr=cfg.settings['cls_lr'], weight_decay=cfg.settings['cls_weight_decay'], amsgrad=amsgrad)
    
    if load_cls_model: # load pretrained model
        logger.info("---------CVAD load trained discriminator model----------")
        cls_model = load_ckpt(cls_model_path, cls_model, cls_optimizer)

# ################################################################################
# start training CVAD
# ################################################################################     
                                               
    train_all(embnet, cls_model, imgSize, variational_beta, cvae_batch_size, cvae_optimizer, cls_optimizer, recon_loss, cls_loss, cfg.settings['dataset_name'], cvae_dataloaders['train'], cvae_dataloaders['val'], cvae_dataloaders['test'], cfg.settings['cvae_n_epochs'], cfg.settings['cls_n_epochs'],cfg.settings['channel'], device)    


if __name__ == '__main__':
    main()
