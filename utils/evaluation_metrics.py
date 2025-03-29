# utils/evaluation_metrics.py
import torch

def iou_score(pred, mask, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * mask).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def dice_loss(pred, mask, threshold=0.5):
    pred = (pred > threshold).float()
    smooth = 1e-6
    intersection = (pred * mask).sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) + smooth)
    return 1 - dice.mean().item()

