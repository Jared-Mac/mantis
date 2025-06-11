import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """
    FiLM generator that produces conditioning parameters for the encoder.
    
    Takes task probabilities and generates FiLM parameters (scale and shift)
    for the final ConvGDN block (48 channels) in the FrankenSplit architecture.
    """
    
    def __init__(self, num_tasks=3, film_channels=48, hidden_dim=64):
        """
        Initialize FiLM generator.
        
        Args:
            num_tasks: Number of tasks (input dimension)
            film_channels: Number of channels to generate FiLM params for (48)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.film_channels = film_channels
        
        # MLP to generate FiLM parameters
        # Output: 2 * film_channels (scale + shift parameters)
        self.mlp = nn.Sequential(
            nn.Linear(num_tasks, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * film_channels)  # 2 * 48 = 96 outputs
        )

    def forward(self, task_probs):
        """
        Generate FiLM parameters from task probabilities.
        
        Args:
            task_probs: Task probabilities (B, num_tasks)
            
        Returns:
            film_params: FiLM parameters (B, 2*film_channels)
                        First film_channels values are scale, next are shift
        """
        film_params = self.mlp(task_probs)  # (B, 2*film_channels)
        return film_params 