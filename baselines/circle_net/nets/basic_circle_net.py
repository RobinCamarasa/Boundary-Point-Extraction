from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep
from monai.networks.nets.basic_unet import UpCat
from monai.networks.nets import BasicUNet


class BasicCircleNet(BasicUNet):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels_heatmap: int = 1,
        out_channels_radius: int = 1,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = (
            "LeakyReLU", {"negative_slope": 0.1, "inplace": True}
            ),
        norm: Union[str, tuple] = (
            "instance", {"affine": True}
            ),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv"
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels_heatmap,
            features=features,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            upsample=upsample
        )
        fea = ensure_tuple_rep(features, 6)

        # Define radius path
        self.upcat_1_radius = UpCat(
            spatial_dims, fea[1], fea[0], fea[5],
            act, norm, bias, dropout, upsample, halves=False
            )
        self.final_conv_radius = Conv["conv", spatial_dims](
            fea[5], out_channels_radius, kernel_size=1
            )


    def forward(self, x: torch.Tensor):
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)

        # Define heatmap path
        u1_heatmaps = self.upcat_1(u2, x0)
        logits_heatmaps = self.final_conv(u1_heatmaps)

        # Define radius path
        u1_radius = self.upcat_1_radius(u2, x0)
        logits_radius = self.final_conv_radius(u1_radius)

        return {
            "heatmap": logits_heatmaps,
            "radius": logits_radius,
        }
