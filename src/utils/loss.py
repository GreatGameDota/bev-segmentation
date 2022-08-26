import torch

import segmentation_models_pytorch as smp

JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

# Modified from: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch
def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = torch.from_numpy(y_true)
    y_pred = torch.from_numpy(y_pred)
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = torch.from_numpy(y_true)
    y_pred = torch.from_numpy(y_pred)
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

def criterion(y_pred, y_true):
    return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)

def criterion1(pred1, targets):
  # torch.use_deterministic_algorithms(False)
  # l1 = F.mse_loss(pred1, targets)
  # l1 = F.binary_cross_entropy_with_logits(pred1, targets)
  l1 = criterion(pred1, targets)
  # l1 = F.cross_entropy(pred1, targets) # Multiclass in one channel (pred: B x n_classes x W x H | targets: B x W x H)
  # l1 = bce_dice_loss(pred1, targets) # Multiclass across multiple channels as 0-1
  # torch.use_deterministic_algorithms(True)
  return l1