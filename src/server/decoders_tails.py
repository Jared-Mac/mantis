import torch
import torch.nn as nn
from compressai.layers import ResidualBlockUpsample, ResidualBlock


class GenericDecoder(nn.Module):
    """
    Generic decoder for Stage 1 that reconstructs features for head distillation.
    
    Mirrors the encoder's upsampling path using CompressAI layers.
    """
    
    def __init__(self, input_channels, output_channels, num_blocks=3):
        super().__init__()
        
        # Upsampling blocks to mirror encoder downsampling
        self.blocks = nn.ModuleList()
        
        # First upsampling block
        self.blocks.append(
            ResidualBlockUpsample(input_channels, output_channels, upsample=2)
        )
        
        # Additional blocks without upsampling
        for _ in range(num_blocks - 1):
            self.blocks.append(
                ResidualBlock(output_channels, output_channels)
            )
            
        # Final upsampling to match teacher feature dimensions
        self.final_upsample = ResidualBlockUpsample(output_channels, output_channels, upsample=2)
        
    def forward(self, z):
        """
        Decode latent representation to reconstructed features.
        
        Args:
            z: Latent representation (B, input_channels, H, W)
            
        Returns:
            Reconstructed features for head distillation
        """
        x = z
        for block in self.blocks:
            x = block(x)
        x = self.final_upsample(x)
        return x


class TaskSpecificDecoder(nn.Module):
    """
    Task-specific decoder for Stage 2.
    
    Can be shared across tasks or specialized per task.
    """
    
    def __init__(self, input_channels, output_channels, num_blocks=2):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        
        # Upsampling blocks
        current_channels = input_channels
        for i in range(num_blocks):
            if i == 0:
                # First block with upsampling
                self.blocks.append(
                    ResidualBlockUpsample(current_channels, output_channels, upsample=2)
                )
            else:
                # Additional blocks
                self.blocks.append(
                    ResidualBlock(output_channels, output_channels)
                )
            current_channels = output_channels
            
    def forward(self, z):
        """
        Decode latent representation for task-specific processing.
        
        Args:
            z: Latent representation (B, input_channels, H, W)
            
        Returns:
            Task-specific features
        """
        x = z
        for block in self.blocks:
            x = block(x)
        return x


class ClassificationTail(nn.Module):
    """
    Classification head for image classification tasks.
    """
    
    def __init__(self, input_channels, num_classes, hidden_dim=None):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        if hidden_dim is not None:
            # Two-layer classifier
            self.classifier = nn.Sequential(
                nn.Linear(input_channels, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            # Single-layer classifier
            self.classifier = nn.Linear(input_channels, num_classes)
            
    def forward(self, x):
        """
        Classify input features.
        
        Args:
            x: Feature map (B, C, H, W)
            
        Returns:
            Class logits (B, num_classes)
        """
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)


class SegmentationTail(nn.Module):
    """
    Segmentation head for pixel-level tasks.
    """
    
    def __init__(self, input_channels, num_classes):
        super().__init__()
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, 3, padding=1),
            nn.BatchNorm2d(input_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 2, num_classes, 1)
        )
        
    def forward(self, x):
        """
        Generate segmentation map.
        
        Args:
            x: Feature map (B, C, H, W)
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        return self.segmentation_head(x)


class TaskSpecificTail(nn.Module):
    """
    Wrapper for different types of task-specific heads.
    """
    
    def __init__(self, task_type, input_channels, num_classes, **kwargs):
        super().__init__()
        
        if task_type == 'classification':
            self.head = ClassificationTail(input_channels, num_classes, **kwargs)
        elif task_type == 'segmentation':
            self.head = SegmentationTail(input_channels, num_classes, **kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
            
    def forward(self, x):
        return self.head(x) 