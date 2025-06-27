import numpy as np
import os
import torch
from datetime import datetime
from src.datamodules.dataset_utils import get_window

class get_s1_s2_timeseries_images :
    def __init__(self, input_path, input_resolution, nb_timeseries_image, duplication_level_noise):
        self.input_path = input_path
        self.input_resolution = input_resolution
        self.nb_timeseries_image = nb_timeseries_image
        self.duplication_level_noise = duplication_level_noise

    def get_date_from_vrt_name(self, vrt_name) :
        return datetime.strptime(vrt_name.split('_')[1].split('.')[0], "%Y%m%d")
    
    def get_n_closest_dates(
        self,
        tuples_list,
        reference_date,
    ) :
        """
        Returns the n tuples whose first date is closest to the reference date,
        sorted by ascending order of the first date in each tuple.
        """
        
        ref_date = datetime.strptime(reference_date, "%Y%m%d")

        n = min(self.nb_timeseries_image, len(tuples_list))

        # Sort the list by proximity to the reference date
        sorted_by_proximity = sorted(
            tuples_list,
            key=lambda x: abs(datetime.strptime(x[0].split('_')[1].split('.')[0], "%Y%m%d") - ref_date)
        )
        
        # Take the n closest elements and sort them by ascending order of date
        closest_n = sorted(
        sorted_by_proximity[:n],
        key=lambda x: datetime.strptime(x[0].split('_')[1].split('.')[0], "%Y%m%d")
        )
        
        return closest_n

    def __call__(self, bounds, row, year_number, transform) :        
        vrt_list = row[f"vrt_list_{year_number}"]
        lidar_date = row[f"lidar_acquisition_date_{year_number}"]
        if len(lidar_date) == 6 : #If you only have the month, we take the middle of the month
            lidar_date = lidar_date + "15"

        s1_vrt_list_path = os.path.join(self.input_path, f"year_{year_number}/lidar_date_{lidar_date[:6]}/s1/vrt_files")
        s2_vrt_list_path = os.path.join(self.input_path, f"year_{year_number}/lidar_date_{lidar_date[:6]}/s2/vrt_files")
        
        vrt_list = self.get_n_closest_dates(vrt_list, lidar_date)

        #Load input
        input = []
        input_date = []
        for s2_s1_vrt in vrt_list[:self.nb_timeseries_image] :
            s2_vrt = os.path.join(s2_vrt_list_path, s2_s1_vrt[0])
            s1_vrt = os.path.join(s1_vrt_list_path, s2_s1_vrt[1])
            s2_image = get_window(
                image_path=s2_vrt,
                bounds=bounds,
                resolution=self.input_resolution
            )
            s1_image = get_window(
                image_path=s1_vrt,
                bounds=bounds,
                resolution=self.input_resolution
            )
            image = np.concatenate((s2_image,s1_image), axis=0)
            image = image.astype(np.float32).transpose(1, 2, 0)
            image[~np.isfinite(image)] = 0
            input.append(image)
            input_date.append(int(self.get_date_from_vrt_name(s2_s1_vrt[0]).timetuple().tm_yday))

        while len(input) < self.nb_timeseries_image :
            idx = np.random.randint(0, len(input))
            image_added = input[idx] 
            date_added = input_date[idx]

            if self.duplication_level_noise : 
                noise = np.random.normal(0, self.duplication_level_noise, image_added.shape).astype(np.float32)
                image_added = image_added + noise 

            input.insert(idx + 1, image_added)
            input_date.insert(idx + 1, date_added)
        
        if transform:
            input = torch.stack([transform(image) for image in input], dim=0)
        else : 
            input = torch.from_numpy(np.stack(input, axis=0).mean(axis=0))

        return input, input_date