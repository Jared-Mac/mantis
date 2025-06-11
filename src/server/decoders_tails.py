import torch
import torch.nn as nn
from compressai.layers import ResidualBlockUpsample, ResidualBlock, GDN
from .igdn_blocks import DeconvIGDN
from compressai.layers import ResidualBlockWithStride
from vib import VIBBottleneck
import torch.utils.checkpoint as checkpoint


class ResNetBPBlock(nn.Module):
    """ResNet Blueprint Block: 1x1 Conv S=1 -> 3x3 Conv S=1 -> 1x1 Conv S=1"""
    
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection if input/output channels match
        self.use_residual = (in_channels == out_channels)
        
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.use_residual:
            out = out + identity
            
        out = self.relu(out)
        return out

class RestorationBlock(nn.Module):
    """Restoration Block for upsampling features"""
    
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=4, stride=scale_factor, padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.upsample(x)))

class FrankenSplitDecoder(nn.Module):
    """
    FrankenSplit decoder following the blueprint:
    Restoration Block -> ResNet BP Block (9) -> Restoration Block -> ResNet BP Block (4)
    
    Uses smaller channel counts to match paper's 2.06M parameter target.
    """
    
    def __init__(self, input_channels, output_channels):
        super().__init__()
        
        # Channel configuration for ~2.06M parameters
        c1, c2 = 128, 256  # Intermediate channel counts
        mid1, mid2 = 64, 128  # Middle channels for ResNet BP blocks
        
        # First restoration block: upsample from 28x28 to 56x56
        self.restoration1 = RestorationBlock(input_channels, c1, scale_factor=2)
        
        # 9 ResNet BP blocks
        self.resnet_blocks1 = nn.Sequential(*[
            ResNetBPBlock(c1, mid1, c1) for _ in range(9)
        ])
        
        # Second restoration block: upsample from 56x56 to 112x112
        self.restoration2 = RestorationBlock(c1, c2, scale_factor=2)
        
        # 4 ResNet BP blocks
        self.resnet_blocks2 = nn.Sequential(*[
            ResNetBPBlock(c2, mid2, c2) for _ in range(4)
        ])
        
        # Final layer to match teacher feature dimensions
        self.final_conv = nn.Conv2d(
            c2, output_channels, kernel_size=1) if output_channels != c2 else nn.Identity()
        self.out_relu = nn.ReLU(inplace=True)
            
    def forward(self, x):
        # x shape: (B, input_channels, 28, 28)
        x = self.restoration1(x)        # (B, 128, 56, 56)
        x = self.resnet_blocks1(x)      # (B, 128, 56, 56)
        x = self.restoration2(x)        # (B, 256, 112, 112)
        x = self.resnet_blocks2(x)      # (B, 256, 112, 112)
        x = self.out_relu(self.final_conv(x))
        return x


class GenericDecoder(nn.Module):
    """
    Generic decoder for Stage 1 that reconstructs features for head distillation.
    
    Mirrors the encoder's upsampling path using CompressAI layers.
    """
    
    def __init__(self, input_channels, output_channels, num_blocks=3): # Original default
        super().__init__()
        self.blocks = nn.ModuleList()
        # First upsampling block (e.g., z: H/8 -> H/4)
        self.blocks.append(
            ResidualBlockUpsample(input_channels, output_channels, upsample=2)
        )
        # Additional blocks without upsampling
        for _ in range(num_blocks - 1): # This loop might change behavior based on num_blocks
            self.blocks.append(
                ResidualBlock(output_channels, output_channels)
            )
        # Final upsampling (e.g., H/4 -> H/2)
        self.final_upsample = ResidualBlockUpsample(output_channels, output_channels, upsample=2)

    def forward(self, z):
        x = z
        for block in self.blocks:
            x = block(x)
        # If num_blocks=1, self.blocks has one upsampler. Then final_upsample applies again.
        # This results in z (H/8) -> block_out (H/4) -> final_out (H/2). This is the 112x112 output.
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


class GenericDecoderStage1(nn.Module):
    """
    Synthesis Network for Stage 1 Head Distillation, following FrankenSplit pattern.
    Uses transposed convolution + IGDN blocks for proper upsampling and feature synthesis.
    """
    def __init__(self, input_channels, output_channels, num_processing_blocks=2):
        super().__init__()
        
        # Define block parameters similar to FrankenSplit synthesis network
        # Start with input_channels, progressively upsample to output_channels
        intermediate_channels = max(output_channels * 2, input_channels // 2)
        
        block_params = [
            # (in_channels, out_channels, kernel_size, stride, padding, output_padding)
            (input_channels, intermediate_channels, 3, 2, 1, 1),  # First upsample
            (intermediate_channels, output_channels, 3, 2, 1, 1), # Second upsample  
        ]
        
        # Add processing blocks if needed (no upsampling, just refinement)
        for _ in range(num_processing_blocks):
            block_params.append(
                (output_channels, output_channels, 3, 1, 1, 0)  # Processing block
            )
        
        # Build the synthesis layers
        igdn_blocks = []
        for in_channels, out_channels, kernel_size, stride, padding, output_padding in block_params:
            igdn_blocks.append(DeconvIGDN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            ))
        
        self.layers = nn.Sequential(*igdn_blocks)
        
        # Final layer to ensure proper output channels (like in FrankenSplit)
        self.final_layer = nn.Conv2d(output_channels, output_channels, 
                                   kernel_size=3, stride=1, padding=1, bias=False)
            
    def forward(self, z_hat):
        """
        Args:
            z_hat: Latent representation (B, input_channels, H_latent, W_latent)
        Returns:
            Synthesized features (B, output_channels, H_upsampled, W_upsampled)
        """
        x = self.layers(z_hat)
        x = self.final_layer(x)
        return x


class ResNetCompatibleTail(nn.Module):
    """
    ResNet-compatible tail that mimics ResNet's layer3, layer4, and fc layer.
    
    Takes 512-channel features (like ResNet layer2 output), passes them through
    pre-trained ResNet layer3 and layer4, then applies global pooling and a task-specific
    classifier. This allows for maximum use of pre-trained weights.
    """
    
    def __init__(self, input_channels=512, num_classes=1000, use_pretrained_layers=True):
        super().__init__()
        
        if input_channels != 512:
            raise ValueError(f"ResNetCompatibleTail expects input_channels=512 to match ResNet layer3, but got {input_channels}")

        if use_pretrained_layers:
            import torchvision.models as models
            full_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
            self.layer3 = full_resnet.layer3
            self.layer4 = full_resnet.layer4
            self.avgpool = full_resnet.avgpool
            
            # Task-specific classifier, initialized with correct output dimension.
            # Weights are loaded later via load_task_weights().
            self.fc = nn.Linear(2048, num_classes)
        else:
            raise NotImplementedError("Building ResNet layers from scratch is not supported. Please use pre-trained layers.")
        
    def forward(self, x):
        """
        Forward pass through ResNet-compatible tail.
        
        Args:
            x: Features from decoder (B, 512, H, W) - like ResNet layer2 output
            
        Returns:
            Class logits (B, num_classes)
        """
        x = self.layer3(x)      # (B, 512, H, W) -> (B, 1024, H, W)
        
        # Use gradient checkpointing for memory-intensive layer4
        if self.training:
            x = checkpoint.checkpoint(self.layer4, x, use_reentrant=False)
        else:
            x = self.layer4(x)      # (B, 1024, H, W) -> (B, 2048, H/2, W/2)
            
        x = self.avgpool(x)     # (B, 2048, H/2, W/2) -> (B, 2048, 1, 1)
        x = torch.flatten(x, 1) # (B, 2048, 1, 1) -> (B, 2048)
        x = self.fc(x)          # (B, 2048) -> (B, num_classes)
        return x
        
    def load_task_weights(self, class_indices, teacher_fc_weight, teacher_fc_bias):
        """Load task-specific weights into the final classifier."""
        task_weights = teacher_fc_weight[class_indices, :]  # (num_task_classes, 2048)
        task_biases = teacher_fc_bias[class_indices]        # (num_task_classes,)
        
        self.fc.weight.data.copy_(task_weights)
        self.fc.bias.data.copy_(task_biases)
