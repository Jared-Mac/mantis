import torch
import torch.nn as nn
from compressai.layers import ResidualBlock, ResidualBlockWithStride
from film_layer import FiLMLayer


class FiLMedResidualBlock(nn.Module):
    """
    Residual block with FiLM conditioning.
    
    Integrates FiLM transformations into a standard residual block.
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.film_layer = FiLMLayer()
        
        # Use CompressAI's residual block as base
        if stride > 1:
            self.res_block = ResidualBlockWithStride(in_channels, out_channels, stride=stride)
        else:
            self.res_block = ResidualBlock(in_channels, out_channels)
            
    def forward(self, x, gamma=None, beta=None):
        """
        Forward pass with optional FiLM conditioning.
        
        Args:
            x: Input features (B, C, H, W)
            gamma: FiLM gamma parameter (B, C) or None for identity
            beta: FiLM beta parameter (B, C) or None for identity
            
        Returns:
            Output features after residual connection and FiLM
        """
        # Apply residual block
        residual_out = self.res_block(x)
        
        # Apply FiLM if parameters provided
        if gamma is not None and beta is not None:
            residual_out = self.film_layer(residual_out, gamma, beta)
            
        return residual_out


class FiLMedEncoder(nn.Module):
    """
    FiLMed encoder component (Blocks 2 and 3) of the MANTiS client.
    
    Contains multiple FiLMed residual blocks that can be conditioned
    on task-specific FiLM parameters.
    """
    
    def __init__(self, input_channels, output_channels, num_blocks=3, film_bypass=False):
        """
        Initialize FiLMed encoder.
        
        Args:
            input_channels: Number of input channels from stem
            output_channels: Number of output channels for latent representation
            num_blocks: Number of FiLMed residual blocks
            film_bypass: If True, FiLM is bypassed (identity transformation)
        """
        super().__init__()
        
        self.film_bypass = film_bypass
        self.num_blocks = num_blocks
        
        # First block with stride for downsampling
        self.blocks = nn.ModuleList([
            FiLMedResidualBlock(input_channels, output_channels, stride=2)
        ])
        
        # Additional blocks without stride
        for _ in range(num_blocks - 1):
            self.blocks.append(
                FiLMedResidualBlock(output_channels, output_channels, stride=1)
            )
            
    def forward(self, x, film_params_list=None):
        """
        Forward pass through FiLMed encoder.
        
        Args:
            x: Input features from stem (B, C, H, W)
            film_params_list: List of (gamma, beta) tuples for each block, or None
            
        Returns:
            z: Latent representation (B, output_channels, H//2, W//2)
        """
        for i, block in enumerate(self.blocks):
            if self.film_bypass or film_params_list is None:
                # No FiLM conditioning (identity transformation)
                x = block(x, gamma=None, beta=None)
            else:
                # Apply FiLM conditioning
                if i < len(film_params_list):
                    gamma, beta = film_params_list[i]
                    x = block(x, gamma=gamma, beta=beta)
                else:
                    # No FiLM parameters for this block
                    x = block(x, gamma=None, beta=None)
                    
        return x 