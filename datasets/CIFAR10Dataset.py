
import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.dataset import Dataset


class CIFAR10LabelDataset(Dataset):
    def __init__(self, data, ydata, transform):
        self.data = data
        self.ydata = ydata
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.ydata[idx]
        x = np.array(x)
        if self.transform!=None:
            x = self.transform(x)   
        return x, label
    
    
def get_cifar10_data(normal_class):
    
    CIFAR10_PATH = "./data/"
    # extract data and targets
    train_data = datasets.CIFAR10(root=CIFAR10_PATH, train=True, download=True)
    x_train, y_train=train_data.data,train_data.targets
    test_data=datasets.CIFAR10(root=CIFAR10_PATH, train=False, download=True)
    x_test, y_test=test_data.data,test_data.targets

    outlier_classes = list(range(0, 10))

    for c in normal_class:
        outlier_classes.remove(c)
    normal_x_train = [] #train data
    normal_y_train = [] #train label
    outlier_x_test = [] #outlier data for final testing
    outlier_y_test = [] #outlier label for final testing
    for i in range(0, len(y_train)):
        if y_train[i] in normal_class:
            normal_x_train.append(x_train[i])
            normal_y_train.append(0)
        else:
            outlier_x_test.append(x_train[i])
            outlier_y_test.append(1)
            
    normal_x_val = [] #train data
    normal_y_val = [] #train label
    for i in range(0, len(y_test)):
        if y_test[i] in normal_class:
            normal_x_val.append(x_test[i])
            normal_y_val.append(0)
        else:
            outlier_x_test.append(x_test[i])
            outlier_y_test.append(1)
            
    return normal_x_train, normal_y_train, normal_x_val, normal_y_val, outlier_x_test, outlier_y_test
                
    
    
    
