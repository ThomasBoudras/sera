import numpy as np
import os
from src.datamodules.dataset_utils import get_window

class get_s1_s2_composites_images:
    def __init__(self, input_path, input_resolution):
        self.input_path = input_path
        self.input_resolution = input_resolution

        
    def __call__(self, bounds, row, year_number, transform):
        lidar_date = row[f"lidar_acquisition_date_{year_number}"]
        if len(lidar_date) == 6 : #If you only have the month, we take the middle of the month
            lidar_date = lidar_date + "15"

        # Load input
        input = []
        for source in ["s1","s2"] :
            vrt_path = os.path.join(self.input_path, f"lidar_date_{lidar_date[:6]}" , source, f"{source}_EPSG2154.vrt")

            input_source = get_window(
                image_path=vrt_path,
                bounds=bounds,
                resolution=(self.input_resolution)
                )

            input_source = input_source.astype(np.float32).transpose(1, 2, 0)
            input_source[~np.isfinite(input_source)] = 0
            input.append(input_source)
        input = np.concatenate(input, axis=2)

        if transform:
            input = transform(input)

        return input, lidar_date