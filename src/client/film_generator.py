import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """
    FiLM Generator that produces gamma and beta parameters for FiLMed layers.
    
    Takes task probabilities P_task and generates FiLM parameters for each
    convolutional layer in the FiLMed encoder.
    """
    
    def __init__(self, num_tasks, num_filmed_layers, channels_per_layer, hidden_dim=64):
        """
        Initialize FiLM generator.
        
        Args:
            num_tasks: Number of tasks
            num_filmed_layers: Number of layers that will receive FiLM parameters
            channels_per_layer: List of channel counts for each FiLMed layer
            hidden_dim: Hidden dimension for MLP
        """
        super().__init__()
        
        self.num_filmed_layers = num_filmed_layers
        self.channels_per_layer = channels_per_layer
        
        # Total number of FiLM parameters (gamma and beta for each channel)
        total_film_params = sum(c * 2 for c in channels_per_layer)
        
        # MLP to generate FiLM parameters from task probabilities
        self.fc_layers = nn.Sequential(
            nn.Linear(num_tasks, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, total_film_params)
        )
        
        # Initialize with identity transformation (gamma=1, beta=0)
        self._init_identity()
        
    def _init_identity(self):
        """Initialize the final layer to produce identity FiLM parameters."""
        with torch.no_grad():
            # Set final layer weights and biases for identity initialization
            final_layer = self.fc_layers[-1]
            nn.init.zeros_(final_layer.weight)
            
            # Initialize bias to [1, 0, 1, 0, ...] pattern for gamma=1, beta=0
            bias_init = []
            for num_channels in self.channels_per_layer:
                bias_init.extend([1.0] * num_channels)  # gamma = 1
                bias_init.extend([0.0] * num_channels)  # beta = 0
            final_layer.bias.copy_(torch.tensor(bias_init))

    def forward(self, p_task):
        """
        Generate FiLM parameters from task probabilities.
        
        Args:
            p_task: Task probabilities (B, num_tasks)
            
        Returns:
            List of (gamma, beta) tuples for each FiLMed layer
        """
        # Generate flattened FiLM parameters
        film_params_flat = self.fc_layers(p_task)  # (B, total_film_params)
        
        # Reshape into list of (gamma, beta) tuples for each layer
        output_params = []
        current_idx = 0
        
        for num_channels in self.channels_per_layer:
            # Extract gamma and beta for this layer
            gamma = film_params_flat[:, current_idx : current_idx + num_channels]
            beta = film_params_flat[:, current_idx + num_channels : current_idx + 2 * num_channels]
            
            output_params.append((gamma, beta))
            current_idx += 2 * num_channels
            
        return output_params 