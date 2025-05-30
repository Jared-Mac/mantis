#!/usr/bin/env python3
"""
MANTiS Stage 1 Training with WebDataset ImageNet.

This script trains Stage 1 of MANTiS using:
1. WebDataset ImageNet format
2. ResNet50 teacher for head distillation  
3. VIB rate loss for compression
"""

import sys
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import wandb

# Add src directory to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Import our modules
import registry  # This registers our components
from webdataset_wrapper import create_imagenet_webdataset_loaders
from models import MantisStage1
from losses import VIBLossStage1
import torchvision.models as models


def get_argparser():
    parser = argparse.ArgumentParser(description='MANTiS Stage 1 Training with WebDataset')
    parser.add_argument('--data_dir', type=str, default='~/imagenet-1k-wds',
                        help='Directory containing ImageNet webdataset .tar files')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--beta_stage1', type=float, default=0.01,
                        help='Weight for VIB rate loss')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--save_dir', type=str, default='./saved_checkpoints/stage1/',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Logging frequency (batches)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Checkpoint save frequency (epochs)')
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='mantis-stage1',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[],
                        help='Wandb tags')
    
    # Memory management arguments
    parser.add_argument('--grad_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps to reduce memory usage')
    parser.add_argument('--use_checkpointing', action='store_true',
                        help='Use gradient checkpointing to save memory')
    parser.add_argument('--monitor_memory', action='store_true',
                        help='Monitor and log GPU memory usage')
    
    return parser


def setup_models(device):
    """Setup teacher and student models."""
    print("Setting up models...")
    
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
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model = teacher_model.to(device)
    
    # Student model (MANTiS Stage 1) - updated for layer2 teacher (512 channels)
    student_model = MantisStage1(
        client_params={'stem_channels': 128, 'encoder_channels': 256, 'num_encoder_blocks': 3},
        decoder_params={'input_channels': 256, 'output_channels': 512, 'num_blocks': 1},  # Changed from 1024 to 512 to match layer2
        vib_channels=256
    )
    student_model = student_model.to(device)
    
    # Check for multi-GPU and wrap student model with DataParallel
    if torch.cuda.device_count() > 1 and device == 'cuda':
        print(f"Using {torch.cuda.device_count()} GPUs!")
        student_model = nn.DataParallel(student_model)
    
    # Since we truncated the model, we get layer2 features directly from forward()
    # No need for hooks anymore, but keeping the structure for compatibility
    teacher_features = {}
    
    print(f"✓ Models setup complete")
    print(f"  Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()):,} (truncated to layer2)")
    print(f"  Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    
    return teacher_model, student_model, teacher_features


def setup_training(student_model, lr, weight_decay):
    """Setup optimizer, scheduler, and loss functions."""
    print("Setting up training components...")
    
    # Optimizer
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=lr/10)
    
    # Loss functions
    vib_loss = VIBLossStage1(num_pixels_placeholder=65536)
    mse_loss = nn.MSELoss()
    
    print("✓ Training components setup complete")
    return optimizer, scheduler, vib_loss, mse_loss


def train_epoch(student_model, teacher_model, teacher_features, train_loader, 
                optimizer, vib_loss, mse_loss, beta_stage1, epoch, device, log_freq):
    """Train for one epoch."""
    student_model.train()
    teacher_model.eval()
    
    running_loss = 0.0
    running_hd_loss = 0.0
    running_vib_loss = 0.0
    
    # Get gradient accumulation steps from args
    grad_accumulation_steps = getattr(args, 'grad_accumulation_steps', 1)
    monitor_memory = getattr(args, 'monitor_memory', False)
    
    # Create adaptive pooling layer for dimension matching
    adaptive_pool = None
    
    # Reset gradient accumulation
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]', unit="batch")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        
        # Teacher forward pass (no gradients) - now outputs layer2 features directly
        with torch.no_grad():
            teacher_layer2_features = teacher_model(images)
        
        # Student forward pass
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
        total_loss_value = (hd_loss_value + beta_stage1 * vib_loss_value) / grad_accumulation_steps
        
        # Backward pass
        total_loss_value.backward()
        
        # Update weights every grad_accumulation_steps batches
        if (batch_idx + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Accumulate losses for epoch average (use original loss values)
        actual_total_loss = total_loss_value.item() * grad_accumulation_steps
        running_loss += actual_total_loss
        running_hd_loss += hd_loss_value.item()
        running_vib_loss += vib_loss_value.item()
        
        # Memory monitoring
        memory_info = {}
        if monitor_memory:
            memory_info = get_gpu_memory_info()
        
        # Update progress bar with instantaneous and running average losses
        progress_info = {
            'Loss': f'{actual_total_loss:.4f}',
            'HD': f'{hd_loss_value.item():.4f}',
            'VIB': f'{vib_loss_value.item():.2f}',
            'AvgLoss': f'{running_loss / (batch_idx + 1):.4f}'
        }
        
        if monitor_memory and memory_info:
            progress_info['GPU'] = f'{memory_info.get("memory_allocated_gb", 0):.1f}GB'
        
        pbar.set_postfix(progress_info)
        
        # Logging (less frequent as tqdm provides continuous feedback)
        if (batch_idx + 1) % log_freq == 0 and log_freq > 0:
            avg_loss = running_loss / (batch_idx + 1)
            avg_hd = running_hd_loss / (batch_idx + 1)
            avg_vib = running_vib_loss / (batch_idx + 1)
            
            # Log to wandb if enabled
            if hasattr(train_epoch, 'use_wandb') and train_epoch.use_wandb:
                step = epoch * len(train_loader) + batch_idx + 1
                log_dict = {
                    'train/batch_loss': actual_total_loss,
                    'train/batch_hd_loss': hd_loss_value.item(),
                    'train/batch_vib_loss': vib_loss_value.item(),
                    'train/running_avg_loss': avg_loss,
                    'train/running_avg_hd_loss': avg_hd,
                    'train/running_avg_vib_loss': avg_vib,
                    'train/learning_rate': optimizer.param_groups[0]['lr']
                }
                
                # Add memory info if monitoring
                if monitor_memory and memory_info:
                    for key, value in memory_info.items():
                        log_dict[f'memory/{key}'] = value
                
                wandb.log(log_dict, step=step)
            
            memory_str = ""
            if monitor_memory and memory_info:
                memory_str = f", GPU: {memory_info.get('memory_allocated_gb', 0):.1f}/{memory_info.get('total_memory_gb', 0):.1f}GB"
            
            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}: Avg Loss = {avg_loss:.4f}, "
                  f"HD = {avg_hd:.4f}, VIB = {avg_vib:.2f}{memory_str}")
        
        # Clean up GPU cache periodically to prevent memory accumulation
        if monitor_memory and (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    # Final optimizer step if using gradient accumulation
    if len(train_loader) % grad_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    avg_epoch_loss = running_loss / len(train_loader)
    avg_epoch_hd_loss = running_hd_loss / len(train_loader)
    avg_epoch_vib_loss = running_vib_loss / len(train_loader)
    
    return avg_epoch_loss, avg_epoch_hd_loss, avg_epoch_vib_loss


def validate(student_model, teacher_model, teacher_features, val_loader, 
            vib_loss, mse_loss, beta_stage1, device, epoch): # Added epoch for tqdm
    """Validate the model."""
    student_model.eval()
    teacher_model.eval()
    
    total_loss = 0.0
    total_hd_loss = 0.0  
    total_vib_loss = 0.0
    
    # Create adaptive pooling layer for dimension matching
    adaptive_pool = None
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Val]', unit="batch") # Added epoch
        for images, labels in pbar: # Changed from tqdm(val_loader, desc='Validation')
            images = images.to(device)
            
            # Teacher forward pass
            teacher_layer2_features = teacher_model(images)
            
            # Student forward pass
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
            total_loss_value = hd_loss_value + beta_stage1 * vib_loss_value
            
            total_loss += total_loss_value.item()
            total_hd_loss += hd_loss_value.item()
            total_vib_loss += vib_loss_value.item()
            
            pbar.set_postfix({
                'Loss': f'{total_loss_value.item():.4f}',
                'HD': f'{hd_loss_value.item():.4f}',
                'VIB': f'{vib_loss_value.item():.2f}',
                'AvgLoss': f'{total_loss / (pbar.n + 1):.4f}' # Use pbar.n for current iteration
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_hd_loss = total_hd_loss / len(val_loader)
    avg_vib_loss = total_vib_loss / len(val_loader)

    return avg_loss, avg_hd_loss, avg_vib_loss


def get_gpu_memory_info():
    """Get GPU memory usage information."""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        return {
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'max_memory_allocated_gb': max_memory_allocated,
            'total_memory_gb': total_memory,
            'memory_free_gb': total_memory - memory_allocated
        }
    return {}


def main():
    """Main training function."""
    global args # Make args global for tqdm descriptions
    parser = get_argparser()
    args = parser.parse_args()
    
    # Expand paths
    args.data_dir = os.path.expanduser(args.data_dir)
    args.save_dir = os.path.expanduser(args.save_dir)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb_config = {
            'data_dir': args.data_dir,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'beta_stage1': args.beta_stage1,
            'num_workers': args.num_workers,
            'device': args.device,
            'save_dir': args.save_dir,
            'log_freq': args.log_freq,
            'save_freq': args.save_freq,
            'grad_accumulation_steps': args.grad_accumulation_steps,
            'use_checkpointing': args.use_checkpointing,
            'monitor_memory': args.monitor_memory
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            gpu_info = get_gpu_memory_info()
            wandb_config.update({f'gpu_{k}': v for k, v in gpu_info.items()})
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            config=wandb_config
        )
        # Set flag for functions to know wandb is enabled
        train_epoch.use_wandb = True
        validate.use_wandb = True
        print("✓ Wandb initialized")
    else:
        train_epoch.use_wandb = False 
        validate.use_wandb = False
    
    print(f"MANTiS Stage 1 Training")
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Beta (VIB weight): {args.beta_stage1}")
    
    # Memory optimization info
    if args.grad_accumulation_steps > 1:
        print(f"Using gradient accumulation: {args.grad_accumulation_steps} steps")
        print(f"Effective batch size: {args.batch_size * args.grad_accumulation_steps}")
    
    if args.monitor_memory:
        print("GPU memory monitoring enabled")
        if torch.cuda.is_available():
            gpu_info = get_gpu_memory_info()
            print(f"GPU memory: {gpu_info.get('memory_allocated_gb', 0):.1f}GB allocated, "
                  f"{gpu_info.get('total_memory_gb', 0):.1f}GB total")
    
    if args.use_checkpointing:
        print("Gradient checkpointing enabled (memory-efficient training)")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup data loaders
    print("\nLoading data...")
    train_loader, val_loader = create_imagenet_webdataset_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Setup models
    teacher_model, student_model, teacher_features = setup_models(args.device)
    
    # Setup training
    optimizer, scheduler, vib_loss, mse_loss = setup_training(
        student_model, args.lr, args.weight_decay
    )
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_hd, train_vib = train_epoch(
            student_model, teacher_model, teacher_features, train_loader,
            optimizer, vib_loss, mse_loss, args.beta_stage1, epoch, 
            args.device, args.log_freq
        )
        
        # Validate  
        val_loss, val_hd, val_vib = validate(
            student_model, teacher_model, teacher_features, val_loader,
            vib_loss, mse_loss, args.beta_stage1, args.device, epoch # Pass epoch
        )
        
        # Scheduler step
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f}, HD: {train_hd:.4f}, VIB: {train_vib:.2f}")
        print(f"  Val   - Loss: {val_loss:.4f}, HD: {val_hd:.4f}, VIB: {val_vib:.2f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log epoch-level metrics to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_loss': train_loss,
                'train/epoch_hd_loss': train_hd,
                'train/epoch_vib_loss': train_vib,
                'val/epoch_loss': val_loss,
                'val/epoch_hd_loss': val_hd,
                'val/epoch_vib_loss': val_vib,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'time/epoch_time': epoch_time,
                'model/best_val_loss': best_val_loss if val_loss >= best_val_loss else val_loss
            }, step=(epoch + 1) * len(train_loader))
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or val_loss < best_val_loss:
            # Handle DataParallel model state_dict
            model_state_dict_to_save = student_model.module.state_dict() \
                if isinstance(student_model, nn.DataParallel) else student_model.state_dict()

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model_state_dict_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': args
            }
            
            # Save regular checkpoint
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, save_path)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)
                print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    
    # Finish wandb run
    if args.use_wandb:
        wandb.log({'model/final_best_val_loss': best_val_loss})
        wandb.finish()
        print("✓ Wandb run completed")


if __name__ == '__main__':
    main() 