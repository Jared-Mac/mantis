import torch
import torch.nn as nn
from compressai.layers import ResidualBlockWithStride, ResidualBlock


class SharedStem(nn.Module):
    """
    Shared stem component (Block 1) of the MANTiS client encoder.
    
    This is the task-agnostic feature extraction component that processes
    input images into initial feature representations.
    """
    
    def __init__(self, input_channels=3, output_channels=128, num_blocks=2):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv2d(input_channels, output_channels // 2, 
                                kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_in = nn.BatchNorm2d(output_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks with stride for downsampling
        self.res_block1 = ResidualBlockWithStride(
            output_channels // 2, output_channels, stride=2
        )
        
        # Additional residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(output_channels, output_channels)
            for _ in range(num_blocks - 1)
        ])
        
    def forward(self, x):
        """
        Forward pass through the shared stem.
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            f_stem: Stem features (B, output_channels, H//4, W//4)
        """
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu(x)
        
        x = self.res_block1(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
            
        return x 