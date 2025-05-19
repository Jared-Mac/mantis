import torch
from torch import nn

class TaskProbabilityModel(nn.Module):
    def __init__(self, 
                 input_channels_from_stem: int,
                 # input_spatial_side_dim: int, # Assuming square feature map after stem for simplicity in flattening
                 output_cond_signal_dim: int,
                 hidden_dims: list = None, # e.g. [128, 64]
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # Use AdaptiveAvgPool2d to handle variable spatial sizes from stem and reduce to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # After pooling, the number of features is just input_channels_from_stem
        flattened_input_dim = input_channels_from_stem
        
        layers = []
        current_dim = flattened_input_dim
        
        if hidden_dims:
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU(inplace=True))
                if dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, output_cond_signal_dim))
        # Typically, the direct output of the linear layer is used as conditioning.
        # If you need probabilities (e.g. for a classification-like task detector),
        # you might add a nn.Sigmoid() or nn.Softmax(dim=-1) here,
        # but FiLMGenerator might expect more general features.
        
        self.predictor = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
        self._output_dim = output_cond_signal_dim

    def forward(self, x_from_stem: torch.Tensor) -> torch.Tensor:
        # x_from_stem is expected to be (B, C, H, W)
        x = self.pool(x_from_stem)  # (B, C, 1, 1)
        x = torch.flatten(x, 1)     # (B, C)
        conditioning_signal = self.predictor(x) # (B, output_cond_signal_dim)
        return self.sigmoid(conditioning_signal)

    def get_output_dim(self) -> int:
        return self._output_dim