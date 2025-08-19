import geopandas as gpd
import numpy as np
import torch
from torch.utils.data import  Dataset
from torchvision.transforms import v2

from src.datamodule.datamodule_utils import generate_random_bounds, expand_gdf
from src.global_utils import get_logger

log = get_logger(__name__)

class changeMapDataset(Dataset):
    def __init__(
        self,
        resolution_input,
        resolution_target,
        patch_size_input,
        patch_size_target,
        gdf_path,
        data_augmentation, 
        get_inputs_t1,
        get_inputs_t2,
        get_targets_t1,
        get_targets_t2,
        get_changes,
        stage,
    ):
        self.resolution_input = resolution_input
        self.resolution_target = resolution_target
        self.patch_size_input = patch_size_input
        self.patch_size_target = patch_size_target
        self.patch_size_real = patch_size_input*resolution_input 
        self.data_augmentation = data_augmentation
        
        self.gdf_path = gdf_path
        self.get_inputs_t1 = get_inputs_t1
        self.get_inputs_t2 = get_inputs_t2
        self.get_targets_t1 = get_targets_t1
        self.get_targets_t2 = get_targets_t2
        self.get_changes = get_changes
        self.stage = stage
        self.gdf = self.prepare_gdf()

        self.transform_input = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.transform_target = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.gdf)

    def prepare_gdf(self):
        gdf = gpd.read_file(self.gdf_path)

        if self.stage != "predict" :
            gdf = gdf[gdf['split'] == self.stage ].reset_index(drop=True)
        
        gdf = self.get_inputs_t1.prepare_gdf_for_inputs(gdf)
        gdf = self.get_inputs_t2.prepare_gdf_for_inputs(gdf)
        log.info(f"Number of {self.stage} samples after filtering : {len(gdf)}")

        if self.stage == "predict" :
            # In prediction mode, we want to predict the entire geometry,
            # while respecting the patch size we can give the model, 
            # so we recover all the subpatches in the bounds. To each 
            # sub-patch we add a margin on its edges for the edge effect.

            margin_size = int(np.ceil(self.patch_size_real/12)) 
            # As we apply 3 margins, with a margin length of 1/6 of 
            # the patch size, we have a complete overlap of each patch. 
            # By convention, we take a margin 2 times smaller.
            
            gdf = expand_gdf(gdf, patch_size=self.patch_size_real, margin_size=margin_size, resolution=self.resolution_input)

    def custom_collate_fn(self, batch):
        inputs = [item[0] for item in batch]  
        targets = [item[1] for item in batch]  
        meta_data = [item[2] for item in batch] 

        batch_input = torch.stack(inputs, dim=0)
        batch_target = torch.stack(targets, dim=0)

        batch_meta_data = {key: torch.stack([d[key] for d in meta_data], dim=0) for key in meta_data[0]}

        return batch_input, batch_target, batch_meta_data

    def __getitem__(self, ix):
        row_gdf = self.gdf.loc[ix]
        bounds = list(row_gdf["geometry"].bounds)
        classification_year = row_gdf["classification_year"]

        #Random Crop
        if self.data_augmentation and self.stage != "predict" : 
            bounds = generate_random_bounds(bounds=bounds, patch_size=self.patch_size_real, resolution=self.resolution_input)
            mirror = np.random.randint(0,3)
            rotation = np.random.randint(0,4)

        #Bottom Left crop
        else : 
            bounds[2], bounds[3] = bounds[0] + self.patch_size_real, bounds[1] + self.patch_size_real
            mirror = 0
            rotation = 0
        
        inputs_t1, metadata_inputs_t1 = self.get_inputs_t1(bounds, row_gdf, self.transform_input)
        targets_t1, metadata_targets_t1 = self.get_targets_t1(bounds, row_gdf, self.transform_target)

        inputs_t2, metadata_inputs_t2 =self.get_inputs_t2(bounds, row_gdf, self.transform_input)
        targets_t2, metadata_targets_t2 = self.get_targets_t2(bounds, row_gdf, self.transform_target)
        
        inputs = torch.cat([inputs_t1, inputs_t2], dim = 0)
        #To simplify the code, we create a single input tensor by concatenating the inputs 
        # for the two dates. To separate the two, we'll cut this single tensor in the middle

        targets = self.get_changes(targets_t1, targets_t2)
       
        if rotation > 0 :
            inputs = torch.rot90(inputs, k=rotation, dims=(-1, -2))
            targets = torch.rot90(targets, k=rotation, dims=(-1, -2))
        
        if mirror > 0 :
            inputs = torch.flip(inputs, dims=[-mirror])  
            targets = torch.flip(targets, dims=[-mirror])  
        
    
        meta_data = {}
        for key, value in metadata_inputs_t1.items() :
            meta_data[key + "_t1"] = value
        for key, value in metadata_targets_t1.items() :
            meta_data[key + "_t1"] = value
        for key, value in metadata_inputs_t2.items() :
            meta_data[key + "_t2"] = value
        for key, value in metadata_targets_t2.items() :
            meta_data[key + "_t2"] = value

        meta_data["classification_years"] = torch.tensor(int(classification_year))

        meta_data["bounds"] = torch.tensor(bounds)
        
        return inputs, targets, meta_data
    
