import subprocess
from pathlib import Path
import matplotlib.colors as mcolors
import torch
import wandb
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torchvision.utils import make_grid
from matplotlib import cm

def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, list):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log_value = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module) :
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log_value, log_freq=self.log_freq)   

    @rank_zero_only
    def on_validation_start(self, trainer, pl_module) :
        logger = get_wandb_logger(trainer=trainer)
        logger.experiment.unwatch()

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module) :
        logger = get_wandb_logger(trainer=trainer)
        logger.experiment.watch(trainer.model, log=self.log_value, log_freq=self.log_freq)

    


class UploadCodeAsArtifact(Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str, use_git: bool = True):
        """

        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")

        if self.use_git:
            # get .git folder
            # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
            git_dir_path = Path(
                subprocess.check_output(["git", "rev-parse", "--git-dir"]).strip().decode("utf8")
            ).resolve()

            for path in Path(self.code_dir).resolve().rglob("*"):
                if (
                    path.is_file()
                    # ignore files in .git
                    and not str(path).startswith(str(git_dir_path))  # noqa: W503
                    # ignore files ignored by git
                    and (  # noqa: W503
                        subprocess.run(["git", "check-ignore", "-q", str(path)]).returncode == 1
                    )
                ):
                    code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        else:
            for path in Path(self.code_dir).resolve().rglob("*.py"):
                code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)


class LogImagePredictions(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int, freq_train, min_value_normalization, max_value_normalization, mode):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True
        self.freq_train = freq_train
        self.min_value_normalization = min_value_normalization
        self.max_value_normalization = max_value_normalization
        self.mode = mode

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False
        
    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        self._save_images(trainer, pl_module, trainer.datamodule.val_dataset, stage="val")

    def on_test_end(self, trainer, pl_module):
        self._save_images(trainer, pl_module, trainer.datamodule.test_dataset, stage="test")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq_train and batch_idx % self.freq_train == 0:
            self._save_images(trainer, pl_module, trainer.datamodule.train_dataset, stage=f"train_step_{batch_idx}")

    def _save_images(self, trainer, pl_module, dataset, stage):

        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            self.device = pl_module.device


            # get a validation batch from the validation dat loader
            samples = [dataset[i] for i in range(self.num_samples)]
            inputs, targets, meta_data = zip(*samples)
            inputs = torch.stack(inputs).to(device=pl_module.device)
            targets = torch.stack(targets).to(device=pl_module.device)
            meta_data =  {key: torch.stack([d[key] for d in meta_data], dim=0) for key in meta_data[0]}


            preds = pl_module(inputs, meta_data).to(device=pl_module.device)


            target_images = []
            pred_images = []
            for i, pred in enumerate(preds) :
                pred_image = pred
                target_image = targets[i]

                if self.mode == "height" :
                    pred_image, target_image = self._normalize_pred_and_target(pred_tensor=pred_image, target_tensor=target_image)
                    target_image = self._color_transform(target_image)
                    pred_image = self._color_transform(pred_image)

                elif self.mode == "difference" : 
                    norm = mcolors.TwoSlopeNorm(vmin=self.min_value_normalization, vcenter=0, vmax=self.max_value_normalization)
                    pred_image = norm(pred_image.cpu().numpy())
                    target_image = norm(target_image.cpu().numpy())
                    colormap = cm.get_cmap("RdYlGn")
                    pred_image = colormap(pred_image)
                    pred_image = torch.from_numpy(pred_image).squeeze(0).permute(2, 0, 1)
                    target_image = colormap(target_image)
                    target_image = torch.from_numpy(target_image).squeeze(0).permute(2, 0, 1)
                    pred_image = pred_image[:3, :, :]
                    target_image = target_image[:3, :, :]
                
                elif self.mode == "change" :
                    target_image = target_image.float()


                target_images.append(target_image)
                pred_images.append(pred_image)
            
            curr_epoch = str(trainer.current_epoch)

            if inputs.dim() == 4 :
                input_images = inputs[:,[2,1,0],:,:]
                input_images = make_grid(input_images)

                experiment.log(
                    {
                        f"input_images/{stage}": wandb.Image(input_images),
                    }
                )

            if inputs.dim() == 5:

                for t in range(inputs.shape[1]):
                    input_images = inputs[:, t,[2,1,0],:,:]
                    input_images = make_grid(input_images)
                    experiment.log(
                        {
                            f"input_images/{stage}/time_{t}": wandb.Image(input_images),
                        }
                    )

            target_images = make_grid(target_images)
            pred_images = make_grid(pred_images)

            experiment.log(
                {
                    f"target_images/{stage}": wandb.Image(target_images),
                    f"predicted_images/{stage}/epoch_{curr_epoch}": wandb.Image(pred_images),
                }
        )
                


    def _normalize_pred_and_target(self, pred_tensor, target_tensor,):
        normalized_pred = (pred_tensor - self.min_value_normalization) / (self.max_value_normalization - self.min_value_normalization)
        normalized_target = (target_tensor - self.min_value_normalization) / (self.max_value_normalization - self.min_value_normalization)
        
        # Clamp pour s'assurer que les valeurs sont entre 0 et 1
        normalized_pred = normalized_pred.clamp(0, 1)
        normalized_target = normalized_target.clamp(0, 1)
        
        return normalized_pred, normalized_target

    def _color_transform(self, tensor):
        # Convertir le tensor normalisé en RGB avec noir -> blanc et blanc -> vert
        white = torch.tensor([1, 1, 1]).view(3, 1, 1).to(device=self.device)  # Blanc
        green = torch.tensor([0, 0.5, 0]).view(3, 1, 1).to(device=self.device)  # Vert

        # Étendre le tensor normalisé à 3 canaux (R, G, B)
        tensor_rgb = torch.zeros(3, *tensor.shape[1:]).to(device=self.device)
        
        # L'interpolation entre le blanc et le vert
        tensor_rgb = tensor * green + (1 - tensor) * white
        
        return tensor_rgb
