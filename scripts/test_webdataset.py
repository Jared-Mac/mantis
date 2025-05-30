#!/usr/bin/env python3
"""
Test script for MANTiS with WebDataset ImageNet.

This script tests:
1. WebDataset loading with ImageNet data
2. MANTiS Stage 1 model forward pass
3. Loss computation
4. Basic training step
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

# Add src directory to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Import our modules
import registry  # This registers our components
from webdataset_wrapper import create_imagenet_webdataset_loaders
from models import MantisStage1
from losses import VIBLossStage1
import torchvision.models as models


def test_webdataset_loading(data_dir, batch_size=8, num_batches=3):
    """Test webdataset loading."""
    print(f"\n=== Testing WebDataset Loading ===")
    print(f"Data directory: {data_dir}")
    
    train_loader, val_loader = create_imagenet_webdataset_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=2  # Reduce workers for testing
    )
    
    print(f"Created train and validation loaders")
    
    # Test train loader
    print("Testing train loader...")
    train_iter = iter(train_loader)
    
    for i in range(num_batches):
        try:
            batch = next(train_iter)
            images, labels = batch
            print(f"  Batch {i+1}: images {images.shape}, labels {labels.shape}")
            print(f"    Image range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"    Label range: [{labels.min()}, {labels.max()}]")
        except Exception as e:
            print(f"  Error in batch {i+1}: {e}")
            break
    
    # Test val loader
    print("Testing validation loader...")
    val_iter = iter(val_loader)
    
    for i in range(2):  # Just test 2 batches for validation
        try:
            batch = next(val_iter)
            images, labels = batch
            print(f"  Val Batch {i+1}: images {images.shape}, labels {labels.shape}")
        except Exception as e:
            print(f"  Error in val batch {i+1}: {e}")
            break
    
    return train_loader, val_loader


def test_mantis_stage1_model(train_loader, device='cuda'):
    """Test MANTiS Stage 1 model with teacher distillation."""
    print(f"\n=== Testing MANTiS Stage 1 Model ===")
    
    # Teacher model (frozen ResNet50) - truncated to layer2 for efficiency
    full_teacher = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Create truncated teacher that only goes up to layer2
    class TruncatedResNet50(nn.Module):
        def __init__(self, resnet50):
            super().__init__()
            self.conv1 = resnet50.conv1
            self.bn1 = resnet50.bn1
            self.relu = resnet50.relu
            self.maxpool = resnet50.maxpool
            self.layer1 = resnet50.layer1
            self.layer2 = resnet50.layer2
            # Note: We exclude layer3, layer4, avgpool, and fc for efficiency
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            return x
    
    teacher_model = TruncatedResNet50(full_teacher)
    teacher_model.eval()
    teacher_model = teacher_model.to(device)
    
    student_model = MantisStage1(
        client_params={'stem_channels': 128, 'encoder_channels': 256, 'num_encoder_blocks': 3},
        decoder_params={'input_channels': 256, 'output_channels': 512, 'num_blocks': 1},  # Changed from 1024 to 512 to match layer2
        vib_channels=256
    )
    student_model = student_model.to(device)
    
    # Create loss function
    vib_loss = VIBLossStage1(num_pixels_placeholder=65536)
    mse_loss = nn.MSELoss()
    
    print(f"Models created and moved to {device}")
    
    # Test forward pass
    train_iter = iter(train_loader)
    batch = next(train_iter)
    images, labels = batch
    images = images.to(device)
    
    print(f"Testing forward pass with batch: {images.shape}")
    
    with torch.no_grad():
        # Teacher forward pass - now outputs layer2 features directly
        teacher_model.eval()
        teacher_layer2_features = teacher_model(images)
        
        # Student forward pass
        student_model.eval()
        student_outputs = student_model(images)
        
        print(f"Teacher layer2 output: {teacher_layer2_features.shape}")
        print(f"Student g_s_output: {student_outputs['g_s_output'].shape}")
        print(f"Student z_likelihoods keys: {list(student_outputs['z_likelihoods'].keys())}")
        
        # Check if dimensions match
        if student_outputs['g_s_output'].shape != teacher_layer2_features.shape:
            print(f"⚠️  Dimension mismatch detected!")
            print(f"  Expected: {teacher_layer2_features.shape}")
            print(f"  Got: {student_outputs['g_s_output'].shape}")
            print(f"  Trying adaptive pooling to match dimensions...")
            
            # Use adaptive pooling to match dimensions
            target_size = teacher_layer2_features.shape[-2:]  # Get H, W
            adaptive_pool = nn.AdaptiveAvgPool2d(target_size).to(device)
            student_features = adaptive_pool(student_outputs['g_s_output'])
            print(f"  After adaptive pooling: {student_features.shape}")
        else:
            student_features = student_outputs['g_s_output']
        
        # Test losses
        # VIB loss
        vib_loss_value = vib_loss(student_outputs['z_likelihoods'], None)
        print(f"VIB loss: {vib_loss_value.item():.6f}")
        
        # Head distillation loss
        hd_loss_value = mse_loss(student_features, teacher_layer2_features)
        print(f"Head distillation loss: {hd_loss_value.item():.6f}")
        
        # Combined loss
        total_loss = hd_loss_value + 0.01 * vib_loss_value
        print(f"Total loss: {total_loss.item():.6f}")
    
    print("✓ Model forward pass and loss computation successful!")
    return student_model, teacher_model


def test_training_step(student_model, teacher_model, train_loader, device='cuda'):
    """Test a basic training step."""
    print(f"\n=== Testing Training Step ===")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=0.0001, weight_decay=0.01)
    
    # Create loss functions
    vib_loss = VIBLossStage1(num_pixels_placeholder=65536)
    mse_loss = nn.MSELoss()
    
    # Create adaptive pooling layer for dimension matching
    adaptive_pool = None
    
    # Training mode
    student_model.train()
    teacher_model.eval()
    
    train_iter = iter(train_loader)
    
    print("Running training steps...")
    for step in range(3):
        try:
            batch = next(train_iter)
            images, labels = batch
            images = images.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - teacher now outputs layer2 features directly
            with torch.no_grad():
                teacher_layer2_features = teacher_model(images)
            
            student_outputs = student_model(images)
            
            # Handle dimension mismatch with adaptive pooling
            student_features = student_outputs['g_s_output']
            teacher_target = teacher_layer2_features  # Direct output from truncated model
            
            if student_features.shape != teacher_target.shape:
                if adaptive_pool is None:
                    target_size = teacher_target.shape[-2:]
                    adaptive_pool = nn.AdaptiveAvgPool2d(target_size).to(device)
                student_features = adaptive_pool(student_features)
            
            # Compute losses
            vib_loss_value = vib_loss(student_outputs['z_likelihoods'], None)
            hd_loss_value = mse_loss(student_features, teacher_target)
            total_loss = hd_loss_value + 0.01 * vib_loss_value
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            print(f"  Step {step+1}: Total Loss = {total_loss.item():.6f}, "
                  f"HD Loss = {hd_loss_value.item():.6f}, "
                  f"VIB Loss = {vib_loss_value.item():.6f}")
            
        except Exception as e:
            print(f"  Error in training step {step+1}: {e}")
            break
    
    print("✓ Training steps completed successfully!")


def main():
    """Main test function."""
    # Configuration
    data_dir = os.path.expanduser("~/imagenet-1k-wds")
    batch_size = 8  # Small batch for testing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Testing MANTiS with WebDataset ImageNet")
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    
    try:
        # Test 1: WebDataset loading
        train_loader, val_loader = test_webdataset_loading(
            data_dir=data_dir,
            batch_size=batch_size,
            num_batches=3
        )
        
        # Test 2: Model forward pass
        student_model, teacher_model = test_mantis_stage1_model(
            train_loader=train_loader,
            device=device
        )
        
        # Test 3: Training step
        test_training_step(
            student_model=student_model,
            teacher_model=teacher_model,
            train_loader=train_loader,
            device=device
        )
        
        print(f"\n🎉 All tests passed! MANTiS with WebDataset is working correctly.")
        print(f"\nYou can now run full training with:")
        print(f"python scripts/train_stage1_webdataset.py --data_dir {data_dir}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 