import torch
import torch.nn as nn
from compressai.layers import GDN
from film_layer import FiLMLayer

class ConvGDN(nn.Module):
    """Conv2d + GDN."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.gdn = GDN(out_channels)

    def forward(self, x):
        return self.gdn(self.conv(x))

class ResGDNBlock(nn.Module):
    """Residual block with GDN."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvGDN(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvGDN(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.shortcut = None
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
            
        out += identity
        return out

class FiLMedGDNResidualBlock(nn.Module):
    """FiLMed Residual block with GDN."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.res_gdn_block = ResGDNBlock(in_channels, out_channels, stride=stride)
        self.film_layer = FiLMLayer()

    def forward(self, x, gamma=None, beta=None):
        residual_out = self.res_gdn_block(x)
        if gamma is not None and beta is not None:
            return self.film_layer(residual_out, gamma, beta)
        return residual_out 