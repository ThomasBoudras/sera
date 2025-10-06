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
        nan_mask = ~torch.isnan(target)  # create a mask where target is not NaN
        
        # Normalize values based on sign
        pred_masked = pred[nan_mask]
        target_masked = target[nan_mask]

        positive_pred_mask = pred_masked >= 0
        positive_target_mask = target_masked >= 0

        pred_masked[positive_pred_mask] = pred_masked[positive_pred_mask] / self.max_value
        target_masked[positive_target_mask] = target_masked[positive_target_mask] / self.max_value

        pred_masked[~positive_pred_mask] = pred_masked[~positive_pred_mask] / -self.min_value
        target_masked[~positive_target_mask] = target_masked[~positive_target_mask] / -self.min_value


        
        return F.smooth_l1_loss(pred_masked, target_masked, reduction=self.reduction, beta=self.beta)
