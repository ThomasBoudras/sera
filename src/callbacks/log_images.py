import torch
from torchvision.utils import make_grid
from src.callbacks.log_images_utils import get_tensorboard_logger
from lightning.pytorch import Callback
from lightning.pytorch import Trainer

class LogImages(Callback):
    """
    Logs a batch of validation samples and their predictions to TensorBoard.

    Args:
        num_samples (int): Number of samples to log.
        freq_train: Frequency of logging during training, if None, no logging during training.
        prepare_images: class to prepare images for logging. To be adapted to the nature of the images.
        log_inputs: class to log input images. To be adapted to the nature of the images.
    """

    def __init__(self, num_samples: int, freq_train, prepare_images, log_inputs):
        super().__init__()
        self.num_samples = num_samples
        self.freq_train = freq_train
        self.prepare_images = prepare_images
        self.log_inputs = log_inputs

    def on_validation_epoch_end(self, trainer, pl_module):
        # Only save images if not in sanity check, using trainer.running_sanity_check
        if not trainer.sanity_checking :
            self._save_images(trainer, pl_module, trainer.datamodule.val_dataset, stage="val")

    def on_test_end(self, trainer, pl_module):
        self._save_images(trainer, pl_module, trainer.datamodule.test_dataset, stage="test")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq_train is not None and batch_idx % self.freq_train == 0:
            self._save_images(trainer, pl_module, trainer.datamodule.train_dataset, stage=f"train_step_{batch_idx}")

    def _save_images(self, trainer, pl_module, dataset, stage):
        logger = get_tensorboard_logger(trainer=trainer)
        experiment = logger.experiment
        self.device = pl_module.device

        # get a batch from the dataset
        samples = [dataset[i] for i in range(self.num_samples)]
        inputs, targets, meta_data = zip(*samples)
        inputs = torch.stack(inputs).to(device=pl_module.device)
        targets = torch.stack(targets).to(device=pl_module.device)
        meta_data =  {key: torch.stack([d[key] for d in meta_data], dim=0) for key in meta_data[0]}

        preds = pl_module(inputs, meta_data).to(device=pl_module.device)

        target_images = []
        pred_images = []
        for i, pred in enumerate(preds):
            pred_image = pred
            target_image = targets[i]

            pred_image, target_image = self.prepare_images(pred_image, target_image)

            target_images.append(target_image)
            pred_images.append(pred_image)
        
        # log input images
        self.log_inputs(inputs, experiment, stage)

        # log target and predicted images
        curr_epoch = int(trainer.current_epoch)

        target_images_grid = make_grid(target_images)
        pred_images_grid = make_grid(pred_images)
        # TensorBoard expects tensors (C, H, W) or (N, C, H, W)
        experiment.add_image(f"target_images/{stage}", target_images_grid, global_step=0) # As the target are the same each epoch, we dont specify the epoch
        experiment.add_image(f"predicted_images/{stage}/epoch_{curr_epoch}", pred_images_grid, global_step=curr_epoch)
