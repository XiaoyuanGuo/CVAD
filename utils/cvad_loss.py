import torch
from torch.nn import functional as F

def recon_loss(recon_x, x, mu, logvar, mu2, logvar2, variational_beta, imgSize, channel,  bsz=64):
    MSE = F.mse_loss(recon_x,x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD += -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= bsz * imgSize * imgSize * channel
    return MSE + variational_beta * KLD

