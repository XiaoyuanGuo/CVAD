import logging
import numpy as np

from .SIIM import *
from .CIFAR10Dataset import *

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


def build_cvae_dataset(dataset_name, data_path, cvae_batch_size, normal_class):
    logger = logging.getLogger()
    logger.info("Build CVAE dataset for {}".format(dataset_name))
    
    assert dataset_name in ['cifar10', 'siim']
    
    if dataset_name == "cifar10":
      
        normal_x_train, normal_y_train, normal_x_val, normal_y_val, outlier_x_test, outlier_y_test = get_cifar10_data(normal_class)

        train_set = CIFAR10LabelDataset(normal_x_train, normal_y_train, cifar10_tsfms)
        validate_set = CIFAR10LabelDataset(normal_x_val, normal_y_val, cifar10_tsfms)
        test_set = CIFAR10LabelDataset(normal_x_val+outlier_x_test, normal_y_val+outlier_y_test, cifar10_tsfms)
    
    elif dataset_name == "siim":
        
        benign, malignant = get_siim_data()
        
        train_set = SIIM_Dataset("./data/SIIM/train/", benign[0:int(0.8*(len( benign)))], [0]*len(benign[0:int(0.8*(len( benign)))]))
        validate_set = SIIM_Dataset("./data/SIIM/train/", benign[int(0.8*(len( benign)))+1:], [0]*len(benign[int(0.8*(len( benign)))+1:]))
        test_set = SIIM_Dataset("./data/SIIM/train/", benign[int(0.8*(len( benign)))+1:]+malignant, [0]*len(benign[int(0.8*(len( benign)))+1:])+[1]*len(malignant))
        
    ae_dataloaders = {'train': DataLoader(train_set, batch_size = ae_batch_size, shuffle = True, num_workers = 1),
                      'val': DataLoader(validate_set, batch_size = ae_batch_size, num_workers = 1),
                      'test': DataLoader(test_set, batch_size = ae_batch_size, num_workers = 1)}
    ae_dataset_sizes = {'train': len(train_set), 'val': len(validate_set), 'test':len(test_set)}
        
    return ae_dataloaders, ae_dataset_sizes


# def build_cvad_dateset(dataset_name, data_path, cvad_batch_size, normal_class):
#     logger = logging.getLogger()
#     logger.info("Build CVAD discriminator dataset for {}".format(dataset_name))
    
#     assert dataset_name in ['cifar10', 'bird', 'siim']
    
#     if dataset_name == "cifar10":
#         normal_x_train, normal_y_train, normal_x_val, normal_y_val, outlier_x_test, outlier_y_test = get_cifar10_data(normal_class)
#         cls_train_set = CIFAR10LabelDataset(normal_x_train, normal_y_train, cifar10_tsfms)
#         cls_validate_set = CIFAR10LabelDataset(normal_x_val+outlier_x_test, normal_y_val+outlier_y_test, cifar10_tsfms)

#     elif dataset_name == "bird":
#         normal_x_train, normal_y_train, normal_x_val, normal_y_val, outlier_x_test, outlier_y_test = get_bird_data(normal_class)
#         cls_train_set = BIRD_Dataset(normal_x_train, normal_y_train)
#         cls_validate_set = BIRD_Dataset(normal_x_val+outlier_x_test, normal_y_val+outlier_y_test)     

#     elif dataset_name == "siim":
        
#         benign, malignant = get_siim_data()
#         cls_train_set = SIIM_Dataset("./data/SIIM/train/", benign[0:int(0.8*(len( benign)))], [0]*len(benign[0:int(0.8*(len( benign)))]))
#         cls_validate_set = SIIM_Dataset("./data/SIIM/train/", benign[int(0.8*(len( benign)))+1:]+malignant, [0]*len(benign[int(0.8*(len( benign)))+1:])+[1]*len(malignant))
        
#     cls_dataloaders = {'train': DataLoader(cls_train_set, batch_size = cvad_batch_size, shuffle = True, num_workers = 1),
#                         'val': DataLoader(cls_validate_set, batch_size = cvad_batch_size, num_workers = 1)}
#     cls_dataset_sizes = {'train': len(cls_train_set), 'val': len(cls_validate_set)}

#     return cls_dataloaders, cls_dataset_sizes

