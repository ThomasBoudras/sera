import numpy as np
import os
import torch
from src.datamodules.dataset_utils import get_window

class get_lidar_images :
    def __init__(self, target_path, target_resolution, replace_nan_by_zero_in_target):
        self.target_path = target_path
        self.target_resolution = target_resolution
        self.replace_nan_by_zero_in_target = replace_nan_by_zero_in_target


    def __call__(self, bounds, row, year_number, transform) :
        target_unit = row[f"unit_lidar_{year_number}"]
        area = row[f"area"]
        lidar_date = row[f"lidar_acquisition_date_{year_number}"]
        if len(lidar_date) == 6 : #If you only have the month, we take the middle of the month
            lidar_date = lidar_date + "15"

        vrt_path  = os.path.join(self.target_path, area, f"lidar_{lidar_date[:4]}/lidar_masked/full.vrt")

        target = get_window(
            image_path=vrt_path,
            bounds=bounds,
            resolution=self.target_resolution
        )
        target = target.astype(np.float32).transpose(1, 2, 0)

        if self.replace_nan_by_zero_in_target:
            target[np.isnan(target)] = 0

        scaling_factor = {"m": 1, "dm": 10, "cm": 100}
        target = target / scaling_factor[target_unit]

        if transform:
            target = transform(target)
        else :
            target = torch.from_numpy(target)

        return target