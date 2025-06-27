import torch
import torch.nn.functional as F
import torch.nn as nn

class weighted_and_masked_smooth_l1(nn.Module):
    def __init__(self, bins):
        super(weighted_and_masked_smooth_l1, self).__init__()
        self.bins = bins

    def forward(self, pred, target):
        mask_bins = {}
        mask_bins[f".-{self.bins[0]}"] = (target < self.bins[0])
        for i in range(len(self.bins) - 1):
            bin_mask = ((target >= self.bins[i]) & (target < self.bins[i+1]))
            mask_bins[f"{self.bins[i]}-{self.bins[i+1]}"] = bin_mask 
            
        mask_bins[f"{self.bins[-1]}-."] = (target >= self.bins[-1])

        nan_mask = ~torch.isnan(target)  # Crée un masque où target n'est pas NaN
        
        loss_bins = {}
        for bin, mask_bin in mask_bins.items() :
            mask = nan_mask & mask_bin
            if mask.sum().item() > 0:
                loss_bins[bin] = F.smooth_l1_loss(pred[mask], target[mask])
            else :
                loss_bins[bin] = torch.tensor(float('nan'))

            
        loss_bins = [loss for bin, loss in loss_bins.items() if not torch.isnan(loss)]
        return torch.stack(loss_bins).mean()
