import torch
import torch.nn as nn


class TaskDetector(nn.Module):
    """
    Task detector component that predicts task probabilities from stem features.
    
    Takes stem features f_stem and outputs task probabilities P_task.
    """
    
    def __init__(self, input_feat_dim, num_tasks, hidden_dim=64):
        super().__init__()
        
        self.num_tasks = num_tasks
        
        # Global average pooling to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Multi-layer perceptron for task prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(input_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tasks),
            nn.Sigmoid()  # Output probabilities for each task
        )
        
    def forward(self, f_stem):
        """
        Predict task probabilities from stem features.
        
        Args:
            f_stem: Stem features (B, C, H, W)
            
        Returns:
            p_task: Task probabilities (B, num_tasks)
        """
        # Pool spatial dimensions
        x = self.pool(f_stem)  # (B, C, 1, 1)
        x = self.flatten(x)    # (B, C)
        
        # Predict task probabilities
        p_task = self.fc_layers(x)  # (B, num_tasks)
        
        return p_task 