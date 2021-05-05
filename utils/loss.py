
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve, auc


def calc_auc(gt, pred_proba):
    fpr, tpr, thresholds = roc_curve(gt, pred_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc;


def mse_loss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()

def loss_img_mse(input, target, norm_value=None):
    reconstruct_error_img = F.mse_loss(input, target, reduction='sum');
    if norm_value is not None:
        reconstruct_error_img /= norm_value;
    return reconstruct_error_img;

def log_prob_img(output_dist, target, norm_value):
    log_prob = output_dist.log_prob(target).sum();
    mean_val_logprob = log_prob/norm_value;
    return mean_val_logprob;

def log_prob_text(output_dist, target, norm_value):
    log_prob = output_dist.log_prob(target).sum();
    mean_val_logprob = log_prob/norm_value;
    return mean_val_logprob;

def loss_img_bce(input, target, norm_value=None):
    reconstruct_error_img = F.binary_cross_entropy_with_logits(input, target, reduction='sum');
    if norm_value is not None:
        reconstruct_error_img /= norm_value;
    return reconstruct_error_img;

def loss_text(input, target, norm_value=None):
    reconstruct_error_text = F.binary_cross_entropy_with_logits(input, target, reduction='sum');
    if norm_value is not None:
        reconstruct_error_text /= norm_value;
    return reconstruct_error_text

def clf_loss(estimate, gt):
    loss = F.binary_cross_entropy(estimate, gt, reduction='mean');
    return loss

def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement()
