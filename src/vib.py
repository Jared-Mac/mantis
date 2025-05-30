import torch
import torch.nn as nn
import math
from compressai.entropy_models import EntropyBottleneck


def add_quantization_noise(z, training=True):
    """
    Add uniform noise for training simulation of quantization.
    
    Args:
        z: Input tensor to add noise to
        training: Whether in training mode (adds noise) or inference mode (rounds)
        
    Returns:
        Tensor with noise added (training) or rounded (inference)
    """
    if training:
        # Uniform noise U(-0.5, 0.5) for training simulation
        noise = torch.empty_like(z).uniform_(-0.5, 0.5)
        return z + noise
    else:
        # Simple rounding for inference
        return torch.round(z)


class VIBBottleneck(nn.Module):
    """
    Variational Information Bottleneck using CompressAI's entropy model.
    """
    
    def __init__(self, channels):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(channels)
    
    def forward(self, z, training=True):
        """
        Forward pass through VIB bottleneck.
        
        Args:
            z: Input latent representation
            training: Whether in training mode
            
        Returns:
            z_hat: Quantized representation
            z_likelihoods: Likelihoods for rate computation
        """
        # Add quantization noise during training
        z_noisy = add_quantization_noise(z, training)
        
        # Pass through entropy bottleneck
        z_hat, z_likelihoods = self.entropy_bottleneck(z_noisy)
        
        return z_hat, z_likelihoods
    
    def compress(self, z):
        """Compress quantized representation to bitstream."""
        z_quantized = torch.round(z)
        return self.entropy_bottleneck.compress(z_quantized)
    
    def decompress(self, strings, shape):
        """Decompress bitstream back to representation."""
        return self.entropy_bottleneck.decompress(strings, shape) 