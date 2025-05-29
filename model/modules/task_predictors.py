# model/modules/task_predictors.py
import torch
from torch import nn

class TaskProbabilityModel(nn.Module):
    def __init__(self, 
                 input_channels_from_stem: int,
                 output_cond_signal_dim: int,
                 hidden_dims: list = None, # e.g. [128, 64]
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
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
        # Output raw logits, let loss function handle sigmoid/softmax if needed (e.g. BCEWithLogitsLoss)
        
        self.predictor = nn.Sequential(*layers)
        self._output_dim = output_cond_signal_dim

    def forward(self, x_from_stem: torch.Tensor) -> torch.Tensor:
        # x_from_stem is expected to be (B, C, H, W)
        x = self.pool(x_from_stem)   # (B, C, 1, 1)
        x = torch.flatten(x, 1)      # (B, C)
        conditioning_signal = self.predictor(x) # (B, output_cond_signal_dim)
        return conditioning_signal 

    def get_output_dim(self) -> int:
        return self._output_dim