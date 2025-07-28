"""
U-TAE Implementation
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import torch
import torch.nn as nn

from src.module.regression_model.components.ltae import LTAE2d
from src.module.regression_model.components.utae_utils import DownConvBlock, UpConvBlockMF, ConvBlock, Temporal_Aggregator

class UTAEMF(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_widths,
        decoder_widths,
        out_conv,
        str_conv_k,
        str_conv_s,
        str_conv_p,
        agg_mode,
        encoder_norm,
        n_head,
        d_model,
        d_k,
        pad_value,
        padding_mode,
        last_relue,
        coupling_mode, 
    ):
        """
        U-TAE middle fusion architecture for spatio-temporal encoding of satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths of the convolutional encoder.
            This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
            in the architecture.
            The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
            decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which the number of
            channels should be given is also from top to bottom. If this argument is not specified the decoder
            will have the same configuration as the encoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the
            str_conv_k (int): Kernel size of the strided up and down convolutions.
            str_conv_s (int): Stride of the strided up and down convolutions.
            str_conv_p (int): Padding of the strided up and down convolutions.
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE
            d_k (int): Key-Query space dimension
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
            coupling_mode (str): Coupling mode for the middle fusion. Can either be:
                - difference : Difference between the two skip connections.
                - concat : Concatenation of the two skip connections.
        """
        super(UTAES, self).__init__()
        self.n_stages = len(encoder_widths)
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.last_relue = last_relue
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value

        self.coupling_mode = coupling_mode
        assert coupling_mode == "difference"  or coupling_mode == "concat"

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        coupling_factor = 1 if self.coupling_mode == "difference" else 2
        self.up_blocks_samiese = nn.ModuleList(
            UpConvBlockMF(
                d_in=decoder_widths[i]*coupling_factor,
                d_out=decoder_widths[i - 1]*coupling_factor,
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                coupling_mode=coupling_mode,
                norm="batch",
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)
        out_conv_nkernels= [decoder_widths[0]*coupling_factor] + out_conv
        out_conv_nkernels[:-1] *= coupling_factor
        self.out_conv_samiese = ConvBlock(nkernels=out_conv_nkernels, padding_mode=padding_mode, last_relu=self.last_relue)

    def forward_encoder(self, input, batch_positions):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        out = self.in_conv.smart_forward(input)
        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
        # TEMPORAL ENCODER
        out, att = self.temporal_encoder(
            feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask
        )
        return out, att, feature_maps, pad_mask

    def forward(self, input, meta_data=None, return_att=False):
        batch_positions_t1 = meta_data["inputs_dates_t1"]
        batch_positions_t2 = meta_data["inputs_dates_t2"]
        input_separation = input.shape[1]//2

        input_t1 = input[:, :input_separation,:, :, :].contiguous()
        input_t2 = input[:, input_separation:, :, :, :].contiguous()
        
        # SPATIAL and TEMPORAL ENCODER
        out_t1, att_t1, feature_maps_t1, pad_mask_t1 = self.forward_encoder(input_t1, batch_positions_t1)
        out_t2, att_t2, feature_maps_t2, pad_mask_t2 = self.forward_encoder(input_t2, batch_positions_t2)

        assert out_t1.shape == out_t2.shape
        if self.coupling_mode == "difference" :
            out = out_t2 - out_t1
        if self.coupling_mode == "concat" :
            out = torch.cat([out_t1, out_t2], dim=1)
            
        # SPATIAL DECODER
        for i in range(self.n_stages - 1):
            skip_t1 = self.temporal_aggregator(
                feature_maps_t1[-(i + 2)], pad_mask=pad_mask_t1, attn_mask=att_t1
            )
            skip_t2 = self.temporal_aggregator(
                feature_maps_t2[-(i + 2)], pad_mask=pad_mask_t2, attn_mask=att_t2
            )
            out = self.up_blocks_samiese[i](input=out, skip_t1=skip_t1, skip_t2=skip_t2)
        out = self.out_conv_samiese(out)
        return out

