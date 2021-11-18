import logging
import torch
import numpy as np
from numpy import sqrt, argmax
from sklearn.metrics import auc, roc_curve

def get_fpr_tpr_auc(Y_label, Y_preds): 
    fpr, tpr, thresholds = roc_curve(Y_label, Y_preds)
    aucscore = auc(fpr, tpr)
    gmeans = sqrt(tpr * (1-fpr))
    ix = argmax(gmeans)
    logger = logging.getLogger()
    logger.info('Best Threshold=%f, G-Mean=%.3f, FPR=%.3f, TPR=%.3f, AUC=%.3f' % (thresholds[ix], gmeans[ix], fpr[ix], tpr[ix], aucscore))
    return fpr, tpr, aucscore


def cvae_evaluate(embnet, recon_loss, test_dataloader, device, variational_beta, imgSize, channel, cvae_batch_size):
    logger = logging.getLogger()
    logger.info("----------- CVAE evaluating------------")
    Targets = []
    anomaly_score = []

    with torch.set_grad_enabled(False):   
        for idx, (images, targets) in enumerate(test_dataloader):
            images = images.to(device)

            for i in range(0, images.shape[0]):
                recon_x, mu, logvar, mu2, logvar2 = embnet(images[i].unsqueeze(0))
                cvae_loss = recon_loss(recon_x, images[i].unsqueeze(0), mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)

                if not np.isnan(cvae_loss.item()) and not np.isinf(cvae_loss.item()):
                    anomaly_score.append(cvae_loss.item())
                    Targets.append(targets[i].detach().cpu().numpy())
            
    Y_label = np.array(np.vstack(Targets).squeeze(1),dtype=int).tolist() 
    Y_preds = []
    for s in anomaly_score:
        Y_preds.append((s-np.min(np.array(anomaly_score)))/(np.max(np.array(anomaly_score))-np.min(np.array(anomaly_score))))
    fpr, tpr, aucscore = get_fpr_tpr_auc(Y_label, Y_preds) 
    return fpr, tpr, aucscore  


def cvad_evaluate(embnet, cls_model, recon_loss, cls_loss, test_dataloader, device, variational_beta, imgSize, channel, cvae_batch_size):
    logger = logging.getLogger()
    logger.info("----------- CVAD evaluating------------")
    Targets = []
    anomaly_score1 = []
    anomaly_score2 = []

    with torch.set_grad_enabled(False):    
        for idx, (images, targets)  in enumerate(test_dataloader):

            images = images.to(device)

            for i in range(0, images.shape[0]):
                recon_x, mu, logvar, mu2, logvar2 = embnet(images[i].unsqueeze(0))
                outputs = cls_model(images[i].unsqueeze(0))
                cvae_loss = recon_loss(recon_x, images[i].unsqueeze(0), mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)

                if not np.isnan(cvae_loss.item()+outputs.detach().cpu().numpy()[0][0]) and not np.isinf(cvae_loss.item()+outputs.detach().cpu().numpy()[0][0]):
                    anomaly_score1.append([cvae_loss.item()])
                    anomaly_score2.append([outputs.detach().cpu().numpy()[0][0]])
                    Targets.append(targets[i].detach().cpu().numpy())
            
    Y_label = np.array(np.vstack(Targets).squeeze(1),dtype=int).tolist() 
    Y_preds = []
    for s1, s2 in zip(anomaly_score1, anomaly_score2):
        Y_preds.append(0.5*((s1-np.min(np.array(anomaly_score1)))/(np.max(np.array(anomaly_score1))-np.min(np.array(anomaly_score1))) + s2))
    aucscore = None
    fpr, tpr, aucscore = get_fpr_tpr_auc(Y_label, Y_preds) 
    return fpr, tpr, aucscore  
