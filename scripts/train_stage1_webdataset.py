#!/usr/bin/env python3
"""
MANTiS Stage 1 Training with WebDataset ImageNet.

This script trains Stage 1 of MANTiS using:
1. A configuration file for all parameters.
2. WebDataset for efficient data loading.
3. A truncated ResNet50 teacher for head distillation.
4. VIB rate loss for compression.
"""

import sys
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler        # autocast -> torch.amp.autocast
from torch import amp
from tqdm import tqdm
import time
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
import json
from types import SimpleNamespace
import torchvision.models as models
import torch.nn.functional as F                           

# Add src directory to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Import our modules
import registry
from webdataset_wrapper import create_imagenet_webdataset_loaders
from client.models import MANTiSClient
from server.decoders_tails import FrankenSplitDecoder
from losses import VIBLossStage1
from vib import VIBBottleneck

class Stage1MANTiSWrapper(nn.Module):
    def __init__(self, client_params, decoder_params, vib_channels):
        super().__init__()
        self.latent_channels = vib_channels          # <-- remember it

        # client / bottleneck / decoder
        self.client   = MANTiSClient(
            num_tasks=client_params.get('num_tasks', 3),
            latent_channels=vib_channels,
        )
        self.vib      = VIBBottleneck(vib_channels)
        self.decoder  = FrankenSplitDecoder(
            input_channels=vib_channels,
            output_channels=decoder_params['output_channels'],
        )

        # identity FiLM (γ = 1, β = 0) kept as a buffer
        self.register_buffer(
            "identity_film_params",
            torch.cat(
                [torch.ones(vib_channels), torch.zeros(vib_channels)]
            ),
        )

    def forward(self, x):
        b = x.size(0)

        # stem → encoder (identity FiLM)
        f_stem = self.client.stem(x)
        identity_film = self.identity_film_params.unsqueeze(0).expand(b, -1)
        z_raw = self.client.filmed_encoder(f_stem, identity_film)

        # VIB bottleneck
        z_hat, z_liks = self.vib(z_raw, training=self.training)

        # decoder for head‑distillation
        g_s = self.decoder(z_hat)

        return {
            "g_s_output": g_s,
            "z_likelihoods": {"z": z_liks},
            "debug_z_raw_mean":  z_raw.mean(),
            "debug_z_hat_mean":  z_hat.mean(),
            "debug_reconstructed_features_mean": g_s.mean(),
        }

class BppLossStage1(nn.Module):
    """
    BppLoss for Stage 1 following the BppLossOrig pattern.
    Computes rate loss as -log2(likelihoods) normalized by latent spatial dimensions.
    """
    
    def __init__(self, latent_height=28, latent_width=28, reduction='mean'):
        super().__init__()
        self.latent_h = latent_height
        self.latent_w = latent_width
        self.reduction = reduction

    def forward(self, z_likelihoods_dict, *args, **kwargs):
        """
        Compute BPP loss from likelihoods.
        
        Args:
            z_likelihoods_dict: Dictionary containing 'z' key with likelihoods tensor
            
        Returns:
            BPP loss (bits per latent pixel)
        """
        if isinstance(z_likelihoods_dict, dict) and 'z' in z_likelihoods_dict:
            likelihoods = z_likelihoods_dict['z']
        else:
            likelihoods = z_likelihoods_dict
            
        n = likelihoods.size(0)  # batch size
        
        if self.reduction == 'sum':
            bpp = -likelihoods.log2().sum()
        elif self.reduction == 'batchmean':
            bpp = -likelihoods.log2().sum() / n
        elif self.reduction == 'mean':
            bpp = -likelihoods.log2().sum() / (n * self.latent_h * self.latent_w)
        else:
            raise Exception(f"Reduction: {self.reduction} does not exist")
            
        return bpp

def setup_models(config, device, rank=0):
    """Setup teacher and student models based on the config."""
    print("Setting up models...")
    model_cfg = config.model.config

    # Teacher model (frozen ResNet50) - truncated to layer2 for distillation
    full_teacher = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    class TruncatedResNet50(nn.Module):
        def __init__(self, resnet50):
            super().__init__()
            self.conv1, self.bn1, self.relu, self.maxpool = resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool
            self.layer1, self.layer2 = resnet50.layer1, resnet50.layer2
        def forward(self, x):
            # Add debug info for teacher model
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self.layer2(x)
            return x

    teacher_model = TruncatedResNet50(full_teacher).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Test teacher model output range with a dummy input
    if rank == 0:
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 224, 224).to(device)
            teacher_test = teacher_model(dummy_input)
            print(f"\n=== TEACHER MODEL DEBUG ===")
            print(f"Teacher output shape: {teacher_test.shape}")
            print(f"Teacher output: min={teacher_test.min().item():.6f}, max={teacher_test.max().item():.6f}, mean={teacher_test.mean().item():.6f}, std={teacher_test.std().item():.6f}")
            print("=" * 30)

    # Student model - using new config structure
    student_model = Stage1MANTiSWrapper(
        client_params=vars(model_cfg.client_params),
        decoder_params=vars(model_cfg.decoder_params),
        vib_channels=model_cfg.vib_channels
    ).to(device)

    print("✓ Models setup complete.")
    print(f"  Teacher params (frozen): {sum(p.numel() for p in teacher_model.parameters()):,}")
    print(f"  Student params: {sum(p.numel() for p in student_model.parameters()):,}")
    return teacher_model, student_model

def setup_training(student_model, config):
    """Setup optimizer, scheduler, and loss functions."""
    print("Setting up training components...")
    train_cfg = config.training
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=train_cfg.optimizer.lr, weight_decay=train_cfg.optimizer.weight_decay)
    
    scheduler_cfg = train_cfg.scheduler
    SchedulerClass = getattr(torch.optim.lr_scheduler, scheduler_cfg.type)
    scheduler = SchedulerClass(optimizer, **vars(scheduler_cfg.params))

    # For BPP-style loss, normalize by latent dimensions
    vib_loss_fn = VIBLossStage1(num_pixels_placeholder=28*28)
    mse_loss_fn = nn.MSELoss()
    lambda_cos = config.training.loss_weights.lambda_cos  # Read from config
    
    # Beta warm-up configuration
    beta_warmup_epochs = getattr(config.training, 'beta_warmup_epochs', 5)  # Default 5 epochs
    beta_target = config.training.loss_weights.beta_stage1
    
    print(f"✓ Training components setup. Beta warm-up: {beta_warmup_epochs} epochs to reach {beta_target}")
    print(f"  Cosine loss weight: {lambda_cos}")
    return optimizer, scheduler, vib_loss_fn, mse_loss_fn, lambda_cos, beta_warmup_epochs, beta_target

def train_epoch(student_model, teacher_model, train_loader, optimizer, vib_loss_fn, mse_loss_fn, lambda_cos, beta_warmup_epochs, beta_target, config, epoch, device, rank, world_size, scaler):
    student_model.train()
    
    # Calculate current beta value with warm-up
    if epoch < beta_warmup_epochs:
        current_beta = beta_target * (epoch + 1) / beta_warmup_epochs  # Linear warm-up
    else:
        current_beta = beta_target
    
    running_loss, running_hd_loss, running_vib_loss = 0.0, 0.0, 0.0
    adaptive_pool = None
    
    if rank == 0:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.num_epochs} [Train]', unit="batch")
    else:
        pbar = train_loader

    for batch_idx, (images, _) in enumerate(pbar):
        optimizer.zero_grad()  # Zero gradients at the start of each batch
        images = images.to(device, non_blocking=True)

        # Debug input statistics (only first batch to avoid spam)
        if batch_idx == 0 and rank == 0:
            print(f"\n=== INPUT DEBUG (Epoch {epoch+1}) ===")
            print(f"Input images shape: {images.shape}")
            print(f"Input: min={images.min().item():.6f}, max={images.max().item():.6f}, mean={images.mean().item():.6f}, std={images.std().item():.6f}")
            print("=" * 30)

        with torch.no_grad():
            teacher_features = teacher_model(images)

        with amp.autocast(device_type="cuda", dtype=torch.float16,
                          enabled=config.training.use_amp):
            student_outputs = student_model(images)
            student_features = student_outputs['g_s_output']
            
            if student_features.shape != teacher_features.shape:
                if adaptive_pool is None:
                    adaptive_pool = nn.AdaptiveAvgPool2d(teacher_features.shape[2:]).to(device)
                student_features = adaptive_pool(student_features)

            # Ensure teacher & student tensors share the same dtype for fp16-safe gradients
            teacher_features = teacher_features.to(student_features.dtype)   # keeps gradients fp16‑safe

            # Debug feature statistics (only print occasionally to avoid spam)
            if batch_idx == 0 and rank == 0:
                print(f"\n=== FEATURE DEBUG (Epoch {epoch+1}, Batch {batch_idx+1}) ===")
                print(f"Teacher features shape: {teacher_features.shape}")
                print(f"Student features shape: {student_features.shape}")
                print(f"Teacher: min={teacher_features.min().item():.6f}, max={teacher_features.max().item():.6f}, mean={teacher_features.mean().item():.6f}, std={teacher_features.std().item():.6f}")
                print(f"Student: min={student_features.min().item():.6f}, max={student_features.max().item():.6f}, mean={student_features.mean().item():.6f}, std={student_features.std().item():.6f}")
                print(f"MSE before loss: {torch.nn.functional.mse_loss(student_features, teacher_features).item():.6f}")
                
                # Check HD loss computation outside AMP
                hd_loss_outside_amp = mse_loss_fn(student_features, teacher_features)
                print(f"HD loss outside AMP: {hd_loss_outside_amp.item():.6f}")
                print("=" * 50)

            # ── COSINE + MSE DISTORTION ───────────────────────────────
            cos_sim = F.cosine_similarity(
                          student_features.flatten(1),
                          teacher_features.flatten(1),
                          dim=1
                      ).mean()
            cos_loss = 1.0 - cos_sim                           

            hd_loss  = mse_loss_fn(student_features, teacher_features) \
                       + lambda_cos * cos_loss                
                
            # Debug HD loss inside AMP  
            if batch_idx == 0 and rank == 0:
                print(f"MSE loss: {mse_loss_fn(student_features, teacher_features).item():.6f}")
                print(f"Cosine loss: {cos_loss.item():.6f}")
                print(f"Combined HD loss: {hd_loss.item():.6f}")
                
            vib_loss = vib_loss_fn(student_outputs['z_likelihoods'])
            
            # Debug VIB loss computation
            if batch_idx == 0 and rank == 0:
                print(f"\n=== VIB LOSS DEBUG ===")
                z_liks = student_outputs['z_likelihoods']
                print(f"z_likelihoods type: {type(z_liks)}")
                print(f"z_likelihoods keys: {z_liks.keys() if isinstance(z_liks, dict) else 'Not a dict'}")
                if isinstance(z_liks, dict) and 'z' in z_liks:
                    z_tensor = z_liks['z']
                    print(f"z_tensor shape: {z_tensor.shape}")
                    print(f"z_tensor stats (densities): min={z_tensor.min().item():.6f}, max={z_tensor.max().item():.6f}, mean={z_tensor.mean().item():.6f}")
                    # Note: VIBLossStage1 will clamp these densities at 1.0 for the loss calculation
                    print(f"VIB normalization pixels: {28*28}")
                print(f"VIB loss (BPP): {vib_loss.item():.6f}")
                print(f"Beta weight: {current_beta}")
                print(f"Weighted VIB: {(current_beta * vib_loss).item():.6f}")
                print("=" * 30)
                
            total_loss = hd_loss + current_beta * vib_loss
            
            # Debug total loss composition
            if batch_idx == 0 and rank == 0:
                print(f"Total loss: {total_loss.item():.6f} = {hd_loss.item():.6f} + {(current_beta * vib_loss).item():.6f}")
                print("=" * 50)
        
        # Check for NaN loss
        if torch.isnan(total_loss):
            print(f"ERROR: NaN loss detected at epoch {epoch+1}, batch {batch_idx}")
            print(f"  total_loss: {total_loss.item()}, hd_loss: {hd_loss.item()}, vib_loss: {vib_loss.item()}")
            print(f"  z_raw_mean: {student_outputs['debug_z_raw_mean'].item()}")
            print(f"  z_hat_mean: {student_outputs['debug_z_hat_mean'].item()}")
            print(f"  reconstructed_features_mean: {student_outputs['debug_reconstructed_features_mean'].item()}")
            
            # Additional debugging: check model parameters for NaN
            for name, param in student_model.named_parameters():
                if torch.isnan(param).any():
                    print(f"  NaN detected in parameter: {name}")
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"  NaN detected in gradient: {name}")
            
            #Gracefully exit without saving corrupted checkpoints
            if dist.is_initialized():
                dist.destroy_process_group()
            sys.exit(1)
        
        # Additional safety check for extreme loss values
        if total_loss.item() > 1000 or torch.isinf(total_loss):
            print(f"WARNING: Extreme loss detected at epoch {epoch+1}, batch {batch_idx}: {total_loss.item()}")
            print(f"  Skipping backward pass for this batch")
            continue
        
        scaler.scale(total_loss).backward()
        
        # Clip gradients to prevent exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.training.grad_clip_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += total_loss.item()
        running_hd_loss += hd_loss.item()
        running_vib_loss += vib_loss.item()

        if rank == 0:
            # Debug loss accumulation (only occasionally)
            if batch_idx == 0:
                print(f"\n=== LOSS ACCUMULATION DEBUG ===")
                print(f"Current HD loss: {hd_loss.item():.6f}")
                print(f"Current VIB loss: {vib_loss.item():.6f}")
                print(f"Running HD loss: {running_hd_loss:.6f}")
                print(f"Running VIB loss: {running_vib_loss:.6f}")
                print("=" * 30)
                
            # Debug what's actually being displayed
            if batch_idx < 5:  # Show first few batches
                print(f"About to display - HD: {hd_loss.item():.6f}, VIB: {vib_loss.item():.6f}")
                
            pbar.set_postfix({'Loss': f'{total_loss.item():.4f}', 'HD': f'{hd_loss.item():.4f}', 'VIB': f'{vib_loss.item():.4f}'})

            # Log to wandb every log_freq batches
            if config.project.use_wandb and (batch_idx + 1) % config.logging_and_saving.log_freq == 0:
                wandb.log({
                    'batch': epoch * len(train_loader) + batch_idx + 1,
                    'train/batch_total_loss': total_loss.item(),
                    'train/batch_hd_loss': hd_loss.item(),
                    'train/batch_cos_loss': cos_loss.item(),         
                    'train/batch_vib_loss': vib_loss.item(),
                    'train/current_beta': current_beta,  # Beta warm-up tracking
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'debug/z_raw_mean': student_outputs['debug_z_raw_mean'].item(),
                    'debug/z_hat_mean': student_outputs['debug_z_hat_mean'].item(),
                    'debug/reconstructed_features_mean': student_outputs['debug_reconstructed_features_mean'].item(),
                })

    avg_losses = {
        'total': torch.tensor(running_loss / len(train_loader), device=device),
        'hd': torch.tensor(running_hd_loss / len(train_loader), device=device),
        'vib': torch.tensor(running_vib_loss / len(train_loader), device=device)
    }
    
    for key in avg_losses:
        dist.all_reduce(avg_losses[key], op=dist.ReduceOp.AVG)
        avg_losses[key] = avg_losses[key].item()
    return avg_losses

def validate_epoch(student_model, teacher_model, val_loader, vib_loss_fn, mse_loss_fn, beta_warmup_epochs, beta_target, config, epoch, device, rank):
    student_model.eval()
    
    # Calculate current beta value (same as training)
    if epoch < beta_warmup_epochs:
        current_beta = beta_target * (epoch + 1) / beta_warmup_epochs
    else:
        current_beta = beta_target
        
    total_loss, total_hd_loss, total_vib_loss = 0.0, 0.0, 0.0
    adaptive_pool = None

    if rank == 0:
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.training.num_epochs} [Val]', unit="batch")
    else:
        pbar = val_loader

    with torch.no_grad():
        for images, _ in pbar:
            images = images.to(device, non_blocking=True)
            teacher_features = teacher_model(images)
            
            with amp.autocast(device_type="cuda", dtype=torch.float16,
                              enabled=config.training.use_amp):
                student_outputs = student_model(images)
                student_features = student_outputs['g_s_output']

                if student_features.shape != teacher_features.shape:
                    if adaptive_pool is None:
                        adaptive_pool = nn.AdaptiveAvgPool2d(teacher_features.shape[2:]).to(device)
                    student_features = adaptive_pool(student_features)
                
                # Ensure teacher & student tensors share the same dtype for fp16-safe gradients
                teacher_features = teacher_features.to(student_features.dtype)   # keeps gradients fp16‑safe
                
                hd_loss = mse_loss_fn(student_features, teacher_features)
                vib_loss = vib_loss_fn(student_outputs['z_likelihoods'])
                total_loss_batch = hd_loss + current_beta * vib_loss
            
            if torch.isnan(total_loss_batch):
                print(f"WARNING: NaN loss detected during validation at epoch {epoch+1}")
                print(f"  hd_loss: {hd_loss.item()}, vib_loss: {vib_loss.item()}")
                # Log debug info if available
                if 'debug_z_raw_mean' in student_outputs:
                     print(f"  z_raw_mean: {student_outputs['debug_z_raw_mean'].item()}")
                     print(f"  z_hat_mean: {student_outputs['debug_z_hat_mean'].item()}")
                     print(f"  reconstructed_features_mean: {student_outputs['debug_reconstructed_features_mean'].item()}")
            else:
            total_loss += total_loss_batch.item()
            total_hd_loss += hd_loss.item()
            total_vib_loss += vib_loss.item()

            if rank == 0:
                pbar.set_postfix({'Loss': f'{total_loss / (pbar.n + 1):.4f}'})

    avg_losses = {
        'total': torch.tensor(total_loss / len(val_loader), device=device),
        'hd': torch.tensor(total_hd_loss / len(val_loader), device=device),
        'vib': torch.tensor(total_vib_loss / len(val_loader), device=device)
    }
    
    for key in avg_losses:
        dist.all_reduce(avg_losses[key], op=dist.ReduceOp.AVG)
        avg_losses[key] = avg_losses[key].item()
    return avg_losses

def main():
    parser = argparse.ArgumentParser(description='MANTiS Stage 1 Training from config file')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    cli_args = parser.parse_args()

    with open(cli_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = json.loads(json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d))

    # DDP Setup
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = config.training.device if torch.cuda.is_available() else "cpu"
    
    if rank == 0:
        print(f"Starting Stage 1 training. DDP: {is_ddp}, World Size: {world_size}, Device: {device}")
        save_dir = Path(config.project.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / 'config.yaml', 'w') as f:
            yaml.dump(config_dict, f)

    if config.project.use_wandb and rank == 0:
        wandb_run_name = config.project.wandb_run_name or f"s1_{time.strftime('%Y%m%d-%H%M%S')}"
        wandb.init(project=config.project.wandb_project, name=wandb_run_name, tags=config.project.wandb_tags, config=config_dict)
        print("✓ Wandb initialized.")

    scaler = torch.amp.GradScaler('cuda', enabled=config.training.use_amp)
    
    config.data.data_dir = os.path.expanduser(config.data.data_dir)
    train_loader, val_loader = create_imagenet_webdataset_loaders(
        data_dir=config.data.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        image_size=config.data.image_size
    )
    
    teacher_model, student_model = setup_models(config, device, rank)
    if is_ddp:
        student_model = DDP(student_model, device_ids=[local_rank], find_unused_parameters=True)
        
    optimizer, scheduler, vib_loss_fn, mse_loss_fn, lambda_cos, beta_warmup_epochs, beta_target = setup_training(student_model, config)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if hasattr(config.model, 'resume_checkpoint') and config.model.resume_checkpoint:
        print(f"Resuming training from {config.model.resume_checkpoint}")
        ckpt_path = os.path.expanduser(config.model.resume_checkpoint)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model_to_load = student_model.module if is_ddp else student_model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        if rank == 0: print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    for epoch in range(start_epoch, config.training.num_epochs):
        if is_ddp:
            # WebDataset samplers don't have set_epoch method
            if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_losses = train_epoch(student_model, teacher_model, train_loader, optimizer, vib_loss_fn, mse_loss_fn, lambda_cos, beta_warmup_epochs, beta_target, config, epoch, device, rank, world_size, scaler)
        val_losses = validate_epoch(student_model, teacher_model, val_loader, vib_loss_fn, mse_loss_fn, beta_warmup_epochs, beta_target, config, epoch, device, rank)
        
        if rank == 0:
            scheduler.step()
            print(f"\nEpoch {epoch+1}/{config.training.num_epochs}")
            print(f"  Train - Loss: {train_losses['total']:.4f} (HD: {train_losses['hd']:.4f}, VIB: {train_losses['vib']:.4f})")
            print(f"  Val   - Loss: {val_losses['total']:.4f} (HD: {val_losses['hd']:.4f}, VIB: {val_losses['vib']:.4f})")
            
            # Calculate current beta for logging (same logic as in train_epoch)
            if epoch < beta_warmup_epochs:
                current_beta_epoch = beta_target * (epoch + 1) / beta_warmup_epochs
            else:
                current_beta_epoch = beta_target
            print(f"  Beta: {current_beta_epoch:.6f} (target: {beta_target}, warmup: {epoch+1}/{beta_warmup_epochs})")
            
            if config.project.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_total_loss': train_losses['total'], 'train/epoch_hd_loss': train_losses['hd'], 'train/epoch_vib_loss': train_losses['vib'],
                    'val/epoch_total_loss': val_losses['total'], 'val/epoch_hd_loss': val_losses['hd'], 'val/epoch_vib_loss': val_losses['vib'],
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/epoch_beta': current_beta_epoch,  # Beta warm-up tracking
                })

            is_best = val_losses['total'] < best_val_loss
            if is_best: best_val_loss = val_losses['total']
            
            save_dir = Path(config.project.save_dir)
            model_to_save = student_model.module if is_ddp else student_model
            checkpoint = {
                'epoch': epoch + 1, 'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_losses['total'], 'config': config_dict
            }
            
            if (epoch + 1) % config.logging_and_saving.save_freq == 0:
                save_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save(checkpoint, save_path)
                print(f"  Checkpoint saved to {save_path}")
            
            if is_best:
                best_path = save_dir / 'best_model.pth'
                torch.save(checkpoint, best_path)
                print(f"  ✓ New best model saved (val_loss: {best_val_loss:.4f}) at {best_path}")

    if rank == 0:
        print("\n🎉 Stage 1 Training completed!")
        if config.project.use_wandb: wandb.finish()
    if is_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()