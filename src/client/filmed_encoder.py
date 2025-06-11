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


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for FrankenSplit."""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
    def forward(self, x, film_params):
        """
        Apply FiLM to input features.
        
        Args:
            x: Input features (B, C, H, W)
            film_params: FiLM parameters (B, 2*C) - first C for scale, next C for shift
            
        Returns:
            Modulated features (B, C, H, W)
        """
        batch_size = x.size(0)
        
        # Split FiLM parameters into scale and shift
        film_params = film_params.view(batch_size, 2, self.channels)
        scale = film_params[:, 0, :].view(batch_size, self.channels, 1, 1)  # (B, C, 1, 1)
        shift = film_params[:, 1, :].view(batch_size, self.channels, 1, 1)  # (B, C, 1, 1)
        
        # Apply FiLM: y = scale * x + shift
        return scale * x + shift


class FiLMedBlock(nn.Module):
    """ConvGDN block with optional FiLM modulation."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_film=False, bias=False):
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
        
        self.use_film = use_film
        if use_film:
            self.film = FiLMLayer(out_channels)
            
    def forward(self, x, film_params=None):
        """
        Forward pass with optional FiLM modulation.
        
        Args:
            x: Input features
            film_params: FiLM parameters (required if use_film=True)
            
        Returns:
            Output features
        """
        x = self.conv(x)
        x = self.gdn(x)
        
        if self.use_film:
            if film_params is None:
                raise ValueError("FiLM parameters required for FiLMed block")
            x = self.film(x, film_params)
            
        return x


class FiLMedEncoder(nn.Module):
    """
    FiLMed encoder component from FrankenSplit.
    
    Implements the remaining FrankenSplit ConvGDN blocks after the stem:
    - Block 2: 96→48 channels, 5×5, stride=2 (no FiLM)  
    - Block 3: 48→48 channels, 2×2, stride=2 (with FiLM)
    """
    
    def __init__(self, input_channels=96, latent_channels=48):
        super().__init__()
        
        # Block 2: (96, 48, 5, 2, 3) - no FiLM
        self.block2 = FiLMedBlock(
            in_channels=input_channels,
            out_channels=latent_channels,     # 48
            kernel_size=5,
            stride=2,
            padding=3,
            use_film=False,
            bias=False
        )
        
        # Block 3: (48, 48, 2, 2, 0) - with FiLM  
        self.block3 = FiLMedBlock(
            in_channels=latent_channels,      # 48
            out_channels=latent_channels,     # 48
            kernel_size=2,
            stride=2,
            padding=0,
            use_film=True,
            bias=False
            )
            
    def forward(self, f_stem, film_params):
        """
        Forward pass through the FiLMed encoder.
        
        Args:
            f_stem: Stem features (B, 96, H//2, W//2)
            film_params: FiLM parameters from generator (B, 2*latent_channels)
            
        Returns:
            z: Encoded latent representation (B, latent_channels, H//8, W//8)
        """
        # Block 2: 96→48, no FiLM
        x = self.block2(f_stem)
        
        # Block 3: 48→48, with FiLM
        z = self.block3(x, film_params)
                    
        return z 