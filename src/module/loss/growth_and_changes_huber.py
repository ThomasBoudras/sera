import torch
import torch.nn.functional as F
import torch.nn as nn
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import shape
from rasterio import features
from affine import Affine
from pathlib import Path
import matplotlib.pyplot as plt


class GrowthAndChangesHuber(nn.Module):
    def __init__(self, reduction, beta, delta_threshold, resolution, min_area):
        super(GrowthAndChangesHuber, self).__init__()
        self.reduction = reduction
        self.beta = beta
        self.delta_threshold = delta_threshold
        self.min_area = min_area
        self.resolution = resolution
        # self.mean_growth_loss = torch.tensor(0.6377) # v3  gc_loss
        # self.mean_changes_loss = torch.tensor(19.8753) # v3  gc_loss
        self.mean_growth_loss = torch.tensor(0.6371) # v4  gc_loss
        self.mean_changes_loss = torch.tensor(19.9152) # v4  gc_loss
        # self.mean_growth_loss_list = [] v5
        # self.mean_changes_loss_list = [] v5
        # self.max_nb_loss = max_nb_loss v5

    def forward(self, pred, target):
        pred_tensor = pred.squeeze()
        target_tensor = target.squeeze()

        growth_mask = (target_tensor >= 0) & ~torch.isnan(target_tensor) & ~torch.isnan(pred_tensor)

        # Detach mean tracking to avoid autograd issues
        if growth_mask.sum() > 0:
            growth_loss = F.smooth_l1_loss(
                pred_tensor[growth_mask],
                target_tensor[growth_mask],
                reduction=self.reduction,
                beta=self.beta,
            )
            growth_loss = growth_loss / self.mean_growth_loss
        else:
            growth_loss = torch.tensor(0)

        changes_target_mask = self._get_changes(target_tensor)
        changes_mask = changes_target_mask # v4  gc_loss
        # changes_pred_mask = self._get_changes(pred_tensor) # v3 gc_loss
        # changes_mask = changes_target_mask | changes_pred_mask # v3 gc_loss
        changes_mask = changes_mask.to(target_tensor.device) & (~torch.isnan(target_tensor)) & (~torch.isnan(pred_tensor))

        if changes_mask.sum() > 0:
            changes_loss = F.smooth_l1_loss(
                pred_tensor[changes_mask],
                target_tensor[changes_mask],
                reduction=self.reduction,
                beta=self.beta
            )
            changes_loss = changes_loss / self.mean_changes_loss
        else:
            changes_loss = torch.tensor(0)


        if torch.isnan(growth_loss) or torch.isnan(changes_loss):
            print(f"growth_loss: {growth_loss}, changes_loss: {changes_loss}")
            print(f"nb nan pred: {(~torch.isfinite(pred_tensor)).sum()}/{pred_tensor.numel()}, nb nan target: {(~torch.isfinite(target_tensor)).sum()}/{target_tensor.numel()}")
            print(f"taille change : pred: {len(pred_tensor[changes_mask])}  target: {len(target_tensor[changes_mask])}")
            print(f"taille growth : pred: {len(pred_tensor[growth_mask])}  target: {len(target_tensor[growth_mask])}")
            print(f"exemple pred change: {pred_tensor[changes_mask]}")
            print(f"exemple pred growth: {pred_tensor[growth_mask]}")
            print(f"pred {pred_tensor}")
        
        # growth_loss_mean_computed = growth_loss * self.mean_growth_loss
        # changes_loss_mean_computed = changes_loss * self.mean_changes_loss

        # if len(self.mean_growth_loss_list) < 100 and growth_loss.item() != 0:
        #     self.mean_growth_loss_list.append(growth_loss_mean_computed)
        # if len(self.mean_changes_loss_list) < 100 and changes_loss.item() != 0:
        #     self.mean_changes_loss_list.append(changes_loss_mean_computed)
        # mean_growth_loss = torch.mean(torch.stack(self.mean_growth_loss_list)) if len(self.mean_growth_loss_list) > 0 else torch.tensor(0)
        # mean_changes_loss = torch.mean(torch.stack(self.mean_changes_loss_list)) if len(self.mean_changes_loss_list) > 0 else torch.tensor(0)
        
        return growth_loss + changes_loss
    
    
    def _get_changes(self, diff_tensor):
        """
        Compute the change mask between pred and target.

        Args:
            target (np.ndarray or torch.Tensor): Target values.

        Returns:
            torch.Tensor: The final change mask as a torch tensor.
        """

        diff_np = diff_tensor.detach().cpu().numpy().squeeze()
        nan_mask = np.isnan(diff_np)
        changes_init = diff_np < self.delta_threshold
        changes_init = changes_init.astype(np.int16)

        # Use an identity affine transform to avoid the 'NoneType' error for transform
        identity_transform = Affine.identity()
        polygons = [
            shape(geom)
            for geom, value in features.shapes(changes_init.squeeze().astype(np.int16), transform=identity_transform)
            if value == 1
        ]

        # Convert polygons to GeoDataFrame
        gdf = gpd.GeoDataFrame({"geometry": polygons})

        min_nb_pixel = self.min_area / self.resolution**2
        # Filter with min_area
        gdf_filtered = gdf[gdf.geometry.area > min_nb_pixel]

        # Mask the changes
        if len(gdf_filtered["geometry"].values):
            changes_final = rasterio.features.geometry_mask(
                gdf_filtered["geometry"].values,
                transform=identity_transform,
                invert=True,
                out_shape=changes_init.shape,
            )
        else:
            changes_final = np.zeros(changes_init.shape)


        return torch.from_numpy(changes_final.astype(bool))