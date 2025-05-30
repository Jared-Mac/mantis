import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Applies gamma * x + beta transformation to input feature maps.
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        """
        Apply FiLM transformation to input features.
        
        Args:
            x: Input feature map (B, C, H, W)
            gamma: Scaling parameter (B, C) or (B, C, 1, 1)
            beta: Shifting parameter (B, C) or (B, C, 1, 1)
            
        Returns:
            Transformed feature map: gamma * x + beta
        """
        # Ensure gamma and beta are broadcastable to x
        if gamma.ndim == 2:  # (B, C) -> (B, C, 1, 1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        return gamma * x + beta 