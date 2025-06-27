import os
import lightning as L
import numpy as np
import torch 
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from pathlib import Path

from src.datamodules.dataset_utils import BottomLeftCrop, SubsetSampler

from src import train_utils as utils

log = utils.get_logger(__name__)

class Datamodule(L.LightningDataModule):

    def __init__(
        self,
        patch_size_input : int,
        patch_size_target : int,
        batch_size : int,
        num_workers: int,
        persistent_workers: bool,
        max_n_inputs_for_moments_computation: int,
        max_n_inputs_per_epoch: int,
        train_dataset,
        val_dataset,
        test_dataset,
        predict_dataset,
        normalization_save_path: Path,
        normalization_constants : dict[str, list]= None, #{"mean": [128, 128, 128...], "std": [128, 128, 128...]}
    ):
        super().__init__()
        self.max_n_inputs_per_epoch = max_n_inputs_per_epoch
        self.patch_size_input = patch_size_input
        self.patch_size_target = patch_size_target
        self.batch_size = batch_size
        self.max_n_inputs_for_moments_computation = max_n_inputs_for_moments_computation
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.normalization_save_path = normalization_save_path
        self.normalization_constants = normalization_constants
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        assert np.round(self.patch_size_input/self.patch_size_target, 2) == np.round(self.train_dataset.get_targets.target_resolution/self.train_dataset.get_inputs.input_resolution, 2) #to make sure we haven't made a mistake in the config

    def prepare_data(self):
        log.info(f"## Local save dir {self.trainer.log_dir} ##")

        # Get normalization constants (computed on training data only)
        if (self.normalization_constants is None) and (
            not os.path.exists(f"{self.normalization_save_path}/mean.npy") 
        ):
            if not os.path.exists(self.normalization_save_path):
                os.makedirs(self.normalization_save_path)

            def get_mean_std(x, axis):
                x = torch.from_numpy(x).permute(2,1,0)
                return x.mean(axis=axis), x.std(axis=axis)

            # sample geometries to avoid super long computations
            geometries = self.train_dataset.geometries.sample(
                n=min(self.train_dataset.geometries.shape[0], self.max_n_inputs_for_moments_computation)
            )
            
            # Compute mean and std iteratively to avoid loading all images in RAM
            mean_per_image, std_per_image = zip(
                *[
                    get_mean_std(self.train_dataset[ix][0].squeeze().astype(np.float32), axis=(1, 2))
                    # cast to float32 to avoid overflow in std computation
                    for ix in tqdm(geometries.index.tolist(), desc="Computing mean and std of inputs")
                ]
            )
            self.input_mean = np.mean(np.array(mean_per_image), axis=0)
            # NB: this actually the average std per image per band, not the std for all pixels per band
            # (unless pixels of different images are independent)
            self.input_std = np.mean(np.array(std_per_image), axis=0)

            np.save(os.path.join(self.normalization_save_path, "mean.npy"), self.input_mean)
            np.save(os.path.join(self.normalization_save_path, "std.npy"), self.input_std)

        elif self.normalization_constants is not None:
            log.info(f"Normalization constants defined with {self.normalization_constants}")
            self.input_mean = self.normalization_constants["mean"]
            self.input_std = self.normalization_constants["std"]

        else:
            log.info(f"Normalization constants already found in {self.normalization_save_path}")
            self.input_mean = np.load(os.path.join(self.normalization_save_path, "mean.npy"))
            self.input_std = np.load(os.path.join(self.normalization_save_path, "std.npy"))

        # TODO add filtering of train set based on classification mask cover


    def setup(self, stage: str):
        log.info(f"## values mean : {self.input_mean}  ##")
        log.info(f"## values std : {self.input_std}  ##")

        # First apply croping to both input and target, then apply normalization to input only

        transform_input = [v2.ToTensor(), BottomLeftCrop(self.patch_size_input), v2.Normalize(mean=self.input_mean, std=self.input_std)] # does not scale by 255
        transform_target = [v2.ToTensor(), BottomLeftCrop(self.patch_size_target)]

        self.train_dataset.update_transforms(transform_input, transform_target)
        self.val_dataset.update_transforms(transform_input, transform_target)
        self.test_dataset.update_transforms(transform_input, transform_target)
        self.predict_dataset.update_transforms(transform_input, transform_target)


    def train_dataloader(self):
        if self.max_n_inputs_per_epoch :
            sampler = SubsetSampler(self.predict_dataset, self.max_n_inputs_per_epoch)
            shuffle = False
        else : 
            sampler = None
            shuffle = True
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            sampler=sampler,
            collate_fn=self.train_dataset.custom_collate_fn,
            timeout=600,

        )

    def val_dataloader(self):
        if self.max_n_inputs_per_epoch :
            sampler = SubsetSampler(self.predict_dataset, self.max_n_inputs_per_epoch)
            shuffle = False
        else : 
            sampler = None
            shuffle = True
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            sampler=sampler,
            collate_fn=self.val_dataset.custom_collate_fn,
            timeout=600,
        )

    def test_dataloader(self):
        if self.max_n_inputs_per_epoch :
            sampler = SubsetSampler(self.predict_dataset, self.max_n_inputs_per_epoch)
            shuffle = False
        else : 
            sampler = None
            shuffle = True
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            sampler=sampler,
            collate_fn=self.test_dataset.custom_collate_fn,
            timeout=600,
        )
    

    def predict_dataloader(self):
        sampler = SubsetSampler(self.predict_dataset, len(self.predict_dataset))
        
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            sampler=sampler,
            collate_fn=self.predict_dataset.custom_collate_fn,
            timeout=600,
        )

    

