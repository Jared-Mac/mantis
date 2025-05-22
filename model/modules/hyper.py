"""
This module defines hyper-network architectures, which are used in
hyperprior-based learned compression models.

Hyper-networks (like `SimpleResidualHyperNetwork`, `SimpleHyperNetwork`) are
components of certain `CompressionModule` subclasses (e.g.,
`ScaleHyperpriorModule`, `MeanScaleHyperpriorModule` in
`model.modules.compressor.py`). They consist of a hyper-analysis (encoder)
part that processes the main latent representation (y) to produce a hyper-latent
(z), and a hyper-synthesis (decoder) part that transforms the quantized
hyper-latent (z_hat) back into parameters (e.g., scale, mean) for the entropy
model of the main latents.

Key classes provided by the module:
    - HyperNetwork: Base class for hyper-networks, providing `hyper_analysis` (g_a)
      and `hyper_synthesis` (g_s) properties.
    - SimpleResidualHyperNetwork: A hyper-network built using `ResidualBlockWithStride`.
    - SimpleHyperNetwork: A hyper-network built using `ConvBlock3x3`.
"""
from torch import nn
from timm.models.layers import trunc_normal_, to_2tuple

from model.modules.layers.conv import ConvBlock3x3, ResidualBlockWithStride
from model.modules.module_registry import register_hyper_network
from abc import ABC, abstractmethod


class HyperNetwork(nn.Module):
    def __init__(self):
        super(HyperNetwork, self).__init__()

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def hyper_analysis(self):
        return self.g_a

    @property
    def hyper_synthesis(self):
        return self.g_s


@register_hyper_network
class SimpleResidualHyperNetwork(HyperNetwork):

    def __init__(self,
                 ha_in_ch1,
                 ha_in_ch2,
                 hs_in_ch1,
                 hs_in_ch2,
                 target_ch):
        super(SimpleResidualHyperNetwork, self).__init__()

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(in_ch=ha_in_ch1,
                                    out_ch=ha_in_ch2,
                                    stride=2,
                                    activation=nn.LeakyReLU),
            ResidualBlockWithStride(in_ch=ha_in_ch2,
                                    out_ch=hs_in_ch1,
                                    activation=nn.LeakyReLU,
                                    stride=2)
        )
        self.g_s = nn.Sequential(
            ResidualBlockWithStride(in_ch=hs_in_ch1,
                                    out_ch=hs_in_ch2,
                                    activation=nn.LeakyReLU,
                                    stride=2,
                                    upsample=True),
            ResidualBlockWithStride(in_ch=hs_in_ch2,
                                    out_ch=target_ch,
                                    activation=nn.LeakyReLU,
                                    stride=2,
                                    upsample=True)
        )

    def forward(self, x):
        raise NotImplementedError()


@register_hyper_network
class SimpleHyperNetwork(HyperNetwork):

    def __init__(self,
                 ha_in_ch1,
                 ha_in_ch2,
                 hs_in_ch1,
                 hs_in_ch2,
                 target_ch):
        super(SimpleHyperNetwork, self).__init__()

        self.g_a = nn.Sequential(
            ConvBlock3x3(in_ch=ha_in_ch1,
                         out_ch=ha_in_ch2,
                         stride=2,
                         activation=nn.LeakyReLU),
            ConvBlock3x3(in_ch=ha_in_ch2,
                         out_ch=hs_in_ch1,
                         activation=nn.LeakyReLU,
                         stride=2)
        )
        self.g_s = nn.Sequential(
            ConvBlock3x3(in_ch=hs_in_ch1,
                         out_ch=hs_in_ch2,
                         activation=nn.LeakyReLU,
                         stride=2,
                         upsample=True),
            ConvBlock3x3(in_ch=hs_in_ch2,
                         out_ch=target_ch,
                         activation=nn.LeakyReLU,
                         stride=2,
                         upsample=True)
        )

    def forward(self, x):
        raise NotImplementedError()
