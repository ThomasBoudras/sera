import lightning as L
import numpy as np
import torch 
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from pathlib import Path

from src.datamodule.datamodule_utils import SubsetSampler
from src import global_utils as utils

log = utils.get_logger(__name__)

class Datamodule(L.LightningDataModule):

    def __init__(
        self,
        batch_size : int,
        num_workers: int,
        persistent_workers: bool,
        max_n_inputs_for_moments_computation: int,
        max_n_inputs_per_epoch: int,
        normalization_save_path,
        train_dataset,
        val_dataset,
        test_dataset,
        predict_dataset,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_n_inputs_per_epoch = max_n_inputs_per_epoch
        self.max_n_inputs_for_moments_computation = max_n_inputs_for_moments_computation
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.normalization_save_path = Path(normalization_save_path).resolve()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        
    def prepare_data(self):
        log.info(f"## Local save dir {self.trainer.log_dir} ##")

        #Calculate normalisation moments for input images
        if not (self.normalization_save_path / "mean.npy").exists():
            if not self.normalization_save_path.exists():
                self.normalization_save_path.mkdir()

            def get_mean_std(x, axis):
                if len(x.shape) == 4 :
                    x = x[0]
                x[~torch.isfinite(x)] = 0 # replace nan or inf with 0
                return x.mean(axis=axis), x.std(axis=axis)
            
            # sample image to avoid super long computations
            sampled_images = self.train_dataset.gdf.sample(
                n=min(len(self.train_dataset.gdf), self.max_n_inputs_for_moments_computation)
            )
            
            # Compute mean and std iteratively to avoid loading all images in RAM
            mean_per_image, std_per_image = zip(
                *[
                    get_mean_std(self.train_dataset[ix][0].squeeze().to(torch.float32), axis=(1, 2))
                    # cast to float32 to avoid overflow in std computation
                    for ix in tqdm(sampled_images.index.tolist(), desc="Computing mean and std of inputs")
                ]
            )

            # mean_per_image is the average per image per band, std_per_image is the std per image per band
            # we take the mean of the mean and std per image to get the mean and std per band
            self.input_mean = np.mean(np.array(mean_per_image), axis=0)
            self.input_std = np.mean(np.array(std_per_image), axis=0)

            np.save(self.normalization_save_path / "mean.npy", self.input_mean)
            np.save(self.normalization_save_path / "std.npy", self.input_std)
        
        else :
            self.input_mean = np.load(self.normalization_save_path / "mean.npy")
            self.input_std = np.load(self.normalization_save_path / "std.npy")

    def setup(self, stage: str):
        log.info(f"## values mean : {self.input_mean}  ##")
        log.info(f"## values std : {self.input_std}  ##")

        # Once we have moment we add normalization transform to dataset
        transform_input = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=self.input_mean, std=self.input_std)])
        if self.train_dataset is not None :
            self.train_dataset.transform_input = transform_input
        if self.val_dataset is not None :
            self.val_dataset.transform_input = transform_input
        if self.test_dataset is not None :
            self.test_dataset.transform_input = transform_input
        if self.predict_dataset is not None :
            self.predict_dataset.transform_input = transform_input


    def train_dataloader(self):
        if self.max_n_inputs_per_epoch :
            sampler = SubsetSampler(self.train_dataset, self.max_n_inputs_per_epoch, True)
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
            sampler = SubsetSampler(self.val_dataset, self.max_n_inputs_per_epoch, False)
        else : 
            sampler = None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            sampler=sampler,
            collate_fn=self.val_dataset.custom_collate_fn,
            timeout=600,
        )

    def test_dataloader(self):
        if self.max_n_inputs_per_epoch :
            sampler = SubsetSampler(self.test_dataset, self.max_n_inputs_per_epoch, False)
        else : 
            sampler = None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            sampler=sampler,
            collate_fn=self.test_dataset.custom_collate_fn,
            timeout=600,
        )
    

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            sampler=None,
            collate_fn=self.predict_dataset.custom_collate_fn,
            timeout=600,
        )

    

