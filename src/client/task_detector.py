import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskDetector(nn.Module):
    """
    Task detector that predicts task ID from stem features.
    
    Branches from the shared stem to predict which task the input image
    corresponds to. Uses 96-channel features from increased FrankenSplit stem.
    """
    
    def __init__(self, input_channels=96, num_tasks=3):
        """
        Initialize task detector.
        
        Args:
            input_channels: Number of input channels from stem (96)
            num_tasks: Number of tasks to predict
        """
        super().__init__()
        
        # Global average pooling + classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(input_channels, num_tasks)
        
    def forward(self, f_stem):
        """
        Forward pass to predict task ID.
        
        Args:
            f_stem: Features from shared stem (B, 96, H, W)
            
        Returns:
            task_logits: Task prediction logits (B, num_tasks)
        """
        # Global average pooling: (B, 96, H, W) -> (B, 96, 1, 1)
        pooled = self.global_avg_pool(f_stem)
        
        # Flatten: (B, 96, 1, 1) -> (B, 96)
        flattened = pooled.view(pooled.size(0), -1)
        
        # Classify: (B, 96) -> (B, num_tasks)
        task_logits = self.classifier(flattened)
        
        return task_logits