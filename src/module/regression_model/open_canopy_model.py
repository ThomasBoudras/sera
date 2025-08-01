import timm
import torch.nn as nn

from src.module.regression_model.components.open_canopy_utils import (
    SimpleSegmentationHead,
    infer_output,
    set_first_layer,
)
from src.global_utils import get_logger

log = get_logger(__name__)


class OpenCanopyModel(nn.Module):
    """Segmentation Network using timm models as backbone."""

    def __init__(
        self,
        backbone,
        num_classes,
        num_channels,
        segmentation_head,
        pretrained,
        pretrained_path,
        img_size,
    ):
        """Initialize the timmNet model.

        Args:
            backbone (str): Name of the backbone architecture. Default is "vit_base_patch16_384".
            num_classes (int): Number of output classes. Default is 4.
            num_channels (int): Number of input channels. Default is 1.
            pretrained (bool): Whether to use pretrained weights. Default is True.
            pretrained_path (str): Path to the pretrained weights file. Default is None.
            img_size (int): Size of the input image. Default is 512.
            chkpt_path (str): Path to the checkpoint file. Default is None.
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.img_size = img_size

        if pretrained_path is not None:
            pretrained_cfg_overlay = dict(file=pretrained_path)
        else:
            pretrained_cfg_overlay = dict()

        log.info(f"Using timm version: {timm.__version__}")

        # Load model
        if backbone.startswith("swin"):
            additional_arg = {
                "img_size": self.img_size,
                "features_only": True,
            }
            print(f"{additional_arg=}")
        elif backbone.startswith("pvt_v2"):
            additional_arg = {"features_only": True}
        elif backbone.startswith("twins_pcpvt"):
            additional_arg = {}
        elif backbone.startswith("vit_base_r50"):
            additional_arg = {"num_classes": 0}  # {"pretrained_strict": False}
            # No classifier in checkpoint
        else:
            additional_arg = {}
            log.warning(
                f"Backbone {backbone} not recognized, using default arguments"
            )
        # Load timm model
        self.model = timm.create_model(
            self.backbone,
            pretrained=self.pretrained,
            in_chans=3,  # load as RGB
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            **additional_arg,
        )

        set_first_layer(
            self.model, num_channels
        )  # convert to correct number of features

        # if features_only is True, map the output to forward_features
        if not hasattr(self.model, "forward_features"):
            self.model.forward_features = self.model.forward

        if backbone.startswith("swin"):
            # convert output from BHWC to BCHW
            class reorder_swin(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    return [
                        feature.permute(0, 3, 1, 2)
                        for feature in self.model(x)
                    ]

                def forward_features(self, x):
                    return self.forward(x)

            self.model = reorder_swin(self.model)


        (
            self.embed_dim,
            self.downsample_factor,
            self.feature_size,
            self.features_format,
            self.remove_cls_token,
        ) = infer_output(self.model, self.num_channels, self.img_size)

        # Add segmentation head
        self.seg_head = segmentation_head(
            self.embed_dim,
            self.downsample_factor,
            self.remove_cls_token,
            self.features_format,
            self.feature_size,
            self.num_classes,
        )

    def forward(self, x, metas=None):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing the output tensor.
        """
        x = self.model.forward_features(x)

        x = self.seg_head(x)
        return {"out": x}
