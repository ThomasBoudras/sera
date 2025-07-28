import torch
import torch.nn as nn
import torch.nn.functional as F
from src.module.regression_model.components.unet_utils import DoubleConv, Down, Up, OutConv

# Inspired from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
# Full assembly of the parts to form the complete network
class UNet(nn.Module):
    def __init__(
            self,
            n_channels_in,
            bilinear,
            out_activation
            ):
        
        super(UNet, self).__init__()
        self.n_channels = n_channels_in
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)
        self.out_activation = None
        if out_activation is not None:
            if (out_activation == "None") or (out_activation is None) or (out_activation == "null"):
                self.out_activation = None
            else:
                self.out_activation = out_activation

    def forward(self, x, meta_data):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
                                  
        if self.out_activation is not None:
            output = self.out_activation(output)  # eg relu to avoid negative predictions
        # Now the output will have the same WxH as the input
        return output

