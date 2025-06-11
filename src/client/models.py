import torch
import torch.nn as nn
import torch.nn.functional as F

from .stem import SharedStem
from .filmed_encoder import FiLMedEncoder  
from .task_detector import TaskDetector
from .film_generator import FiLMGenerator


class MANTiSClient(nn.Module):
    """
    MANTiS Client model implementing FrankenSplit architecture.
    
    The client processes input images through:
    1. SharedStem: 3→96 channels (first ConvGDN block)
    2. TaskDetector: branches from stem to predict task
    3. FiLMGenerator: generates conditioning from task prediction  
    4. FiLMedEncoder: 96→48→48 channels (blocks 2-3, FiLM on final block)
    
    Architecture matches FrankenSplit AnalysisNetworkCNN with ConvGDN blocks.
    """
    
    def __init__(self, num_tasks=3, latent_channels=48):
        """
        Initialize MANTiS client.
        
        Args:
            num_tasks: Number of tasks for task detection
            latent_channels: Number of channels in final latent representation (48)
        """
        super().__init__()
        
        # FrankenSplit architecture components (increased stem channels)
        self.stem = SharedStem(input_channels=3, output_channels=96)
        self.task_detector = TaskDetector(input_channels=96, num_tasks=num_tasks)
        self.film_generator = FiLMGenerator(num_tasks=num_tasks, film_channels=latent_channels)
        self.filmed_encoder = FiLMedEncoder(input_channels=96, latent_channels=latent_channels)
        
    def forward(self, x):
        """
        Forward pass through MANTiS client.
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            z: Latent representation (B, latent_channels, H//8, W//8)
            task_logits: Task prediction logits (B, num_tasks)
        """
        # Shared stem: 3→96 channels
        f_stem = self.stem(x)
        
        # Task detection from stem features
        task_logits = self.task_detector(f_stem)
        task_probs = F.softmax(task_logits, dim=1)
        
        # Generate FiLM parameters from task prediction
        film_params = self.film_generator(task_probs)
        
        # FiLMed encoding: 96→48→48 channels
        z = self.filmed_encoder(f_stem, film_params)
        
        return z, task_logits 