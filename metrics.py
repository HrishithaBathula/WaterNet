
import torch
import torch.nn.functional as F
from math import log10
from skimage.metrics import structural_similarity as ssim_metric
import numpy as np

def psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100
    return 20 * log10(1.0 / (mse ** 0.5))

def mse(pred, target):
    return F.mse_loss(pred, target).item()

def mae(pred, target):
    return F.l1_loss(pred, target).item()

def ssim(pred, target):
    pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0)
    target_np = target.detach().cpu().numpy().transpose(1, 2, 0)
    return ssim_metric(pred_np, target_np, channel_axis=2, data_range=1.0)

def ssim_loss(pred, target):
    pred_np = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
    target_np = target.detach().cpu().numpy().transpose(0, 2, 3, 1)
    losses = []
    for i in range(pred_np.shape[0]):
        ssim_val = ssim_metric(pred_np[i], target_np[i], channel_axis=2, data_range=1.0)
        losses.append(1 - ssim_val)
    return torch.tensor(losses).mean()
