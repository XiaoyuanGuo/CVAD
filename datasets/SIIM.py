import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms
from torch.utils.data.dataset import Dataset

imgSize = 256

class SIIM_Dataset(Dataset):
    def __init__(self, traindir, imagenames, labels):
        self.traindir = traindir
        self.imagenames = imagenames
        self.labels = labels
        self.transformations = transforms.Compose([
                                     transforms.Resize((imgSize,imgSize)),
                                     transforms.ToTensor()
                                    ])
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.traindir, self.imagenames[idx]))
        if self.transformations != None:
            img = self.transformations(img)
        return img, self.labels[idx]
    
    def __len__(self): 
        return len(self.imagenames)


    
def get_siim_data():
    rootdir = "./data/SIIM/jpeg/"
    traindir = rootdir+"train"
    testdir = rootdir+"test"
    train_images = os.listdir(traindir)
    test_images = os.listdir(testdir)
    train_csv = pd.read_csv('./data/SIIM/train.csv')[1:].reset_index(drop=True)
    benign = train_csv.loc[train_csv["benign_malignant"]=="benign"]
    malignant = train_csv.loc[train_csv["benign_malignant"]=="malignant"]
    
    benigns = []
    for i in range(0, len(benign["image_name"])):
        benigns.append(benign["image_name"].iloc[i]+".jpg")
        
    malignants = []
    for i in range(0, len(malignant["image_name"])):
        malignants.append(malignant["image_name"].iloc[i]+".jpg")    
    
    return benigns, malignants
