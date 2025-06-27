import numpy as np
import os
from src.datamodules.dataset_utils import get_window
import torch

class get_spot_composites_images:
    def __init__(self, input_path, input_resolution):
        self.input_path = input_path
        self.input_resolution = input_resolution

        
    def __call__(self, bounds, row, year_number, transform):
        lidar_date = row[f"lidar_acquisition_date_{year_number}"]
        if len(lidar_date) == 6 : #If you only have the month, we take the middle of the month
            lidar_date = lidar_date + "15"

        vrt_path = os.path.join(self.input_path, str(lidar_date[:4]), f"full_spot.vrt")

        input = get_window(
            image_path=vrt_path,
            bounds=bounds,
            resolution=(self.input_resolution)
            )
        input = input.astype(np.float32).transpose(1, 2, 0)
        input[~np.isfinite(input)] = 0

        if transform:
            input = transform(input)
        return input, torch.tensor([int(lidar_date)])