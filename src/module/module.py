from typing import Any
from pathlib import Path
import torch
from lightning import LightningModule
import numpy as np
import torch.nn.functional as F

class Module(LightningModule):
    def __init__(
        self,
        super_resolution_model,
        regression_model,
        loss,
        train_metrics,
        val_metrics,
        test_metrics,
        scheduler,
        optimizer,
        predictions_save_dir,
        save_target,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["super_resolution_model", "regression_model", "loss", "train_metrics", "val_metrics", "test_metrics"]) # We do not save the models in the hyperparameters of nn.Module to avoid duplication

        self.super_resolution_model = super_resolution_model
        self.regression_model = regression_model
        self.loss = loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.predictions_save_dir = Path(predictions_save_dir).resolve()
        self.save_target = save_target
    
    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return {"optimizer": optimizer}

    def forward(self, x: torch.Tensor, meta_data):
        if self.super_resolution_model is not None :
            x = self.super_resolution_model(x, meta_data)
        preds = self.regression_model(x, meta_data)
        return preds

    def step(self, batch: Any, stage, metrics_function):
        inputs, targets, meta_data = batch
        preds = self.forward(inputs, meta_data)
        
        loss = self.loss(preds, targets)

        self.log(
            name= f"{stage}/loss",
            value=loss, 
            on_step=True,
            on_epoch=True, 
            prog_bar=False,
        )
    
        if metrics_function :
            metrics_function.update(preds, targets, meta_data)
        return loss, preds, targets
    

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch=batch, stage="train", metrics_function=self.train_metrics)
        return {"loss": loss, "preds": preds, "targets": targets}


    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch=batch, stage="val", metrics_function=self.val_metrics)
        return {"loss": loss, "preds": preds, "targets": targets}


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch=batch, stage="test", metrics_function=self.test_metrics)
        return {"loss": loss, "preds": preds, "targets": targets}


    def final_step(self, stage, metrics_function):
        if metrics_function is not None :
            metrics = metrics_function.compute()
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"{stage}/{metric_name}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    )
            metrics_function.reset()   

    def on_train_epoch_end(self):
        self.final_step("train", self.train_metrics)
    
    def on_validation_epoch_end(self):
        self.final_step("val", self.val_metrics)

    def on_test_epoch_end(self):
        self.final_step("test", self.test_metrics)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # At predict time, there are (normally) only inputs, no targets
        input, target, meta_data = batch
        res = self(input, meta_data)

        def crop_tensor(tensor):
            crop_size = int(tensor.shape[-1]/12) 
            tensor[...,:crop_size, :] = np.nan 
            tensor[...,-crop_size:, :] = np.nan  
            tensor[...,:, :crop_size] = np.nan  
            tensor[...,:, -crop_size:] = np.nan  
            return tensor
        
        res = crop_tensor(res)
        input = crop_tensor(input)

        bounds = meta_data["bounds"]
        geometry_id = meta_data["geometry_id"]
        
        if self.predictions_save_dir is not None:
            np.save(
                self.predictions_save_dir / "pred" / f"{batch_idx}.npy",
                res.cpu().numpy().astype(np.float32),
            )
            np.save(
                self.predictions_save_dir / "input" / f"{batch_idx}.npy",
                input.cpu().numpy().astype(np.float32),
            )
            np.save(
                self.predictions_save_dir / "bounds" / f"{batch_idx}.npy",
                bounds.cpu().numpy().astype(np.float32),
            )
            np.save(
                self.predictions_save_dir / "geometry_id" / f"{batch_idx}.npy",
                geometry_id.cpu().numpy().astype(np.float32),
            )
            if self.save_target : 
                target = crop_tensor(target)
                np.save(
                    self.predictions_save_dir / "target" / f"{batch_idx}.npy",
                    target.cpu().numpy().astype(np.float32),
                )

        else:
            raise Exception("Please give a name for the prediction dir ")