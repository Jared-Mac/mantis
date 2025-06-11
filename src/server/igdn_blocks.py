import torch.nn as nn
from compressai.layers import GDN

class DeconvIGDN(nn.Module):
    """ConvTranspose2d + IGDN."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        )
        self.igdn = GDN(out_channels, inverse=True)

    def forward(self, x):
        return self.igdn(self.deconv(x)) 