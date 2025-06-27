import geopandas as gpd
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import  Dataset
from pathlib import Path
from torchvision.transforms import v2

from src.datamodules.dataset_utils import generate_random_bounds, expand_geometries
import json
from omegaconf import OmegaConf


class changeDataset(Dataset):
    def __init__(
        self,
        input_resolution : int,
        patch_size_input, 
        method_change, 
        prediction_mode = False,
        geometries_path = "./",
        data_augmentation = False, 
        get_inputs = None,
        get_targets = None,
        additional_transform_input : v2 = None,
        additional_transform_target: v2 = None,
        split = "train",
    ):
        self.input_resolution = input_resolution
        self.patch_size_input = patch_size_input
        self.max_bounds_size = patch_size_input*input_resolution # we take a +1 pixel security
        self.method_change = method_change
        self.data_augmentation = data_augmentation
        self.prediction_mode = prediction_mode

        self.geometries_path = geometries_path
        self.split = split
        self.geometries = self.get_split_geometries(geometries_path, split, prediction_mode)
        
        self.get_inputs = get_inputs
        self.get_targets = get_targets

        self.transform_input = None
        self.transform_target = None
        self.additional_transform_input =  OmegaConf.to_container(additional_transform_input, resolve=True)
        self.additional_transform_target =  OmegaConf.to_container(additional_transform_target, resolve=True)


    def __len__(self):
        return len(self.geometries)
    

    def update_transforms(self, transform_input : v2 = None, transform_target: v2 = None):
        self.transform_input = v2.Compose(transform_input + self.additional_transform_input)
        self.transform_target = v2.Compose(transform_target + self.additional_transform_target)


    def get_split_geometries(self, geometries_path, split, prediction_mode) :
        geometries = gpd.read_file(geometries_path)
        if split is not None:
            geometries = geometries[geometries['split'] == split ].reset_index(drop=True)   
        if "vrt_list_1" in geometries :
            geometries[f"vrt_list_1"] = geometries['vrt_list_1'].apply(json.loads) 
            geometries[f"vrt_list_2"] = geometries['vrt_list_2'].apply(json.loads) 
        
        if prediction_mode :
            crop_size = int(self.max_bounds_size/12)
            geometries = expand_geometries(geometries, patch_size=self.max_bounds_size, crop_size=crop_size)
        return geometries


    def custom_collate_fn(self, batch):
        input = [item[0] for item in batch]  
        target = [item[1] for item in batch]  
        meta_data = [item[2] for item in batch] 

        batch_input = torch.stack(input, dim=0)
        batch_target = torch.stack(target, dim=0)

        batch_meta_data = {key: torch.stack([d[key] for d in meta_data], dim=0) for key in meta_data[0]}

        return batch_input, batch_target, batch_meta_data


    def __getitem__(self, ix):
        
        row_gdf = self.geometries.loc[ix]
        bounds = list(row_gdf["geometry"].bounds)
        if self.data_augmentation and self.prediction_mode :
            bounds = generate_random_bounds(bounds=bounds, patch_size=self.max_bounds_size, resolution_x=self.input_resolution, resolution_y=self.input_resolution)
            mirror = np.random.randint(0,3)
            rotation = np.random.randint(0,4)
        else :
            bounds[2], bounds[3] = bounds[0] + self.max_bounds_size, bounds[1] + self.max_bounds_size
            mirror = 0
            rotation = 0
        
        input_year_1, input_dates_year_1 =self.get_inputs(bounds, row = row_gdf, year_number=1, transform=self.transform_input)
        target_year_1 = self.get_targets(bounds, row = row_gdf, year_number=1, transform=self.transform_target)

        input_year_2, input_dates_year_2 =self.get_inputs(bounds, row = row_gdf, year_number=2, transform=self.transform_input)
        target_year_2 = self.get_targets(bounds, row = row_gdf, year_number=2, transform=self.transform_target)
        
        input = torch.cat([input_year_1, input_year_2], dim = 0)
        target = self.method_change(target_year_1, target_year_2)
       
        if rotation > 0 :
            input = torch.rot90(input, k=rotation, dims=(-1, -2))
            target = torch.rot90(target, k=rotation, dims=(-1, -2))
        
        if mirror > 0 :
            input = torch.flip(input, dims=[-mirror])  
            target = torch.flip(target, dims=[-mirror])  
        
        bounds = torch.tensor(bounds)
        input_dates = input_dates_year_1 + input_dates_year_2  
        input_date = torch.from_numpy(np.stack(input_dates, axis=0))  

        meta_data = {"bounds" : bounds, "dates" : input_date}

        if self.prediction_mode :
            meta_data["number_geometry"] = torch.tensor(row_gdf["number_geometry"])
        return input, target, meta_data
    
