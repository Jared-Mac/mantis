import torch
import torch.nn as nn
from compressai.layers import GDN


class ConvGDNBlock(nn.Module):
    """Conv2d + GDN block from FrankenSplit."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.gdn = GDN(out_channels)

    def forward(self, x):
        return self.gdn(self.conv(x))


class SharedStem(nn.Module):
    """
    Shared stem component - First FrankenSplit ConvGDN block.
    
    This is the task-agnostic feature extraction component that processes
    input images into initial feature representations. Follows FrankenSplit
    AnalysisNetworkCNN: first block with increased channels (3→96).
    """
    
    def __init__(self, input_channels=3, output_channels=96):
        super().__init__()
        
        # First FrankenSplit block: (3, 96, 5, 2, 2) - increased from 64 channels
        self.conv_gdn = ConvGDNBlock(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=False
        )
        
    def forward(self, x):
        """
        Forward pass through the shared stem.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            
        Returns:
            f_stem: Stem features (B, 96, H//2, W//2)
        """
        return self.conv_gdn(x) 