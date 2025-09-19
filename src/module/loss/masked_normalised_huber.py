import torch
import torch.nn.functional as F
import torch.nn as nn

class MaskedNormalisedHuber(nn.Module):
    def __init__(self, reduction, beta, max_value, min_value):
        super(MaskedNormalisedHuber, self).__init__()
        self.reduction = reduction
        self.beta = beta
        self.max_value = max_value
        self.min_value = min_value

    def forward(self, pred, target):
        mask = ~torch.isnan(target)  # create a mask where target is not NaN
        
        # Normalize values based on sign
        pred_masked = pred[mask]
        target_masked = target[mask]
        
        # For negative values, normalize by 50
        negative_mask = target_masked < 0
        pred_masked[negative_mask] = pred_masked[negative_mask] / self.max_value
        target_masked[negative_mask] = target_masked[negative_mask] / self.max_value
        
        # For positive values, normalize by 5
        positive_mask = target_masked >= 0
        pred_masked[positive_mask] = pred_masked[positive_mask] / self.min_value
        target_masked[positive_mask] = target_masked[positive_mask] / self.min_value
        
        return F.smooth_l1_loss(pred_masked, target_masked, reduction=self.reduction, beta=self.beta)
