# CVAD

This repository contains PyTorch implementation of the following paper: CVAD: A generic medical anomaly detector based on Cascade VAE (https://arxiv.org/abs/2110.15811)
 
##  1. Table of Contents

- [CVAD](#cvad)
    - [Table of Contents](#table-of-contents)
    - [Installation](#installation)
    - [Experiment](#experiment)
        - [Training on CIFAR10 Dataset](#training-on-cifar10)
        - [Training on SIIM-ISIC Dataset](#train-on-siim-dataset)
    - [Citing CVAD](#citing-cvad)
 
## 2. Installation
1. First clone the repository
   ```
   git clone https://github.com/XiaoyuanGuo/CVAD.git
   ```
2. Create the virtual environment via conda
    ```
    conda create -n cvad python=3.7 
    ```
3. Activate the virtual environment.
    ```
    conda activate cvad
    ```
3. Install the dependencies.
   ```
   pip install --user --requirement requirements.txt
   ```
## 3. Experiment
To train the model on CIFAR10/SIIM-ISIC  datasets, run the following commands:

##### CIFAR10
``` shell
python -u main.py cifar10 CVAD ./ --channel 3 --cvae_n_epochs 100 --cls_n_epochs 20 --normal_class 0 
```
##### SIIM-ISIC  
To use the model for SIIM-ISIC dataset, please download the data from https://www.kaggle.com/c/siim-isic-melanoma-classification/data into ./data/ folder.
``` shell
python -u main.py siim CVAD ./ --channel 3 --cvae_n_epochs 100 --cls_n_epochs 20 

```

## 4. Citing CVAD
If you use this repository or would like to refer the paper, please use the following BibTeX entry
```
@article{guo2021cvad,
  title={CVAD: A generic medical anomaly detector based on Cascade VAE},
  author={Guo, Xiaoyuan and Gichoya, Judy Wawira and Purkayastha, Saptarshi and Banerjee, Imon},
  journal={arXiv preprint arXiv:2110.15811},
  year={2021}
}
```
