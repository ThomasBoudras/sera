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


class TrueChangesRMSE(nn.Module):
    def __init__(self, reduction, delta_threshold, resolution, min_area):
        super(TrueChangesRMSE, self).__init__()
        self.reduction = reduction
        self.delta_threshold = delta_threshold
        self.min_area = min_area
        self.resolution = resolution


    def forward(self, pred, target):
        pred_tensor = pred.squeeze()
        target_tensor = target.squeeze()

        growth_mask = (target_tensor >= -2) # we take a margin of 2 to avoid the noise
        changes_target_mask = self._get_changes(target_tensor).to(target_tensor.device)
        
        loss_mask = (growth_mask | changes_target_mask) & (~torch.isnan(target_tensor)) & (~torch.isnan(pred_tensor))

        loss = torch.sqrt(F.mse_loss(
            pred_tensor[loss_mask],
            target_tensor[loss_mask],
            reduction=self.reduction,
        ))

        return loss
    
    
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