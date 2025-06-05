# file_path: scripts/train_stage1_webdataset.py
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
from torch.cuda.amp import GradScaler, autocast # For Mixed Precision
from tqdm import tqdm
import time
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
torch.autograd.set_detect_anomaly(True)
# Add src directory to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Import our modules
import registry  # This registers our components
from webdataset_wrapper import create_imagenet_webdataset_loaders
from models import MantisStage1
from losses import VIBLossStage1
import torchvision.models as models

# For profiling
# import torch.profiler


def get_argparser():
    parser = argparse.ArgumentParser(description='MANTiS Stage 1 Training with WebDataset')
    parser.add_argument('--data_dir', type=str, default='~/imagenet-1k-wds',
                        help='Directory containing ImageNet webdataset .tar files. Consider staging to local SSD if on a cluster.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--beta_stage1', type=float, default=0.01,
                        help='Weight for VIB rate loss')
    parser.add_argument('--num_workers', type=int, default=16, # Adjusted default, tune based on system
                        help='Number of data loading workers. Tune based on CPU cores and I/O.')
    parser.add_argument('--prefetch_factor', type=int, default=8, # PyTorch default, can be tuned
                        help='Number of batches loaded in advance by each worker.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--save_dir', type=str, default='./saved_checkpoints/stage1/',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_freq', type=int, default=200, # Increased logging frequency
                        help='Logging frequency (batches)')
    parser.add_argument('--save_freq', type=int, default=1,
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
    parser.add_argument('--use_checkpointing', action='store_true', # torch.utils.checkpoint
                        help='Use gradient checkpointing to save memory')
    parser.add_argument('--monitor_memory', action='store_true',
                        help='Monitor and log GPU memory usage')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision (AMP)')
    parser.add_argument('--profile_batches', type=int, default=0,
                        help='Number of initial batches to profile with torch.profiler. 0 to disable.')

    # Checkpoint resuming
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training from')

    return parser


def setup_models(device, use_checkpointing=False): # Added use_checkpointing
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

    student_model = MantisStage1(
        client_params={'stem_channels': 128, 'encoder_channels': 256, 'num_encoder_blocks': 3},
        # Adjusted decoder_params for GenericDecoderStage1:
        # input_channels is the output of VIBBottleneck (vib_channels = 256)
        # output_channels is the channel depth of teacher's layer2 (512 for ResNet50)
        decoder_params={'input_channels': 256, 'output_channels': 512, 'num_processing_blocks': 2}, # Example: 2 processing blocks
        vib_channels=256
    )

    if use_checkpointing: # Apply gradient checkpointing if enabled
        print("Applying gradient checkpointing to student model...")
        if hasattr(student_model.client_encoder.encoder, 'blocks'): # Check if 'blocks' attribute exists
            # Assuming student_model.client_encoder.encoder.blocks is a ModuleList of checkpointable modules
            # This is a general example; you might need to adjust based on your model structure
            # and which specific parts benefit most from checkpointing.
            # The example below shows a conceptual application.
            # A more common pattern is to call torch.utils.checkpoint.checkpoint directly in the forward pass
            # of the module you want to checkpoint.
            # For this example, let's assume we are wrapping the module if it's not already a DataParallel instance.
            # This part needs careful implementation based on how MantisStage1 is structured.
            # The original `checkpoint_wrapper` is less common now.
            # The current approach is typically:
            #
            # from torch.utils.checkpoint import checkpoint
            #
            # class MyModule(nn.Module):
            #     def __init__(self):
            #         super().__init__()
            #         self.layer = nn.Linear(10,10)
            #     def forward(self, x):
            #          # if self.training and use_checkpointing:
            #          #    return checkpoint(self.layer, x, use_reentrant=False)
            #          # else:
            #          return self.layer(x)
            #
            # Given the current structure, we might enable checkpointing within the MantisStage1 model's forward pass
            # or its submodules' forward passes if `use_checkpointing` is true.
            # For now, this is a placeholder for where you'd integrate it more deeply.
            # If MantisStage1 itself or parts of it are single large nn.Sequential,
            # checkpointing sections of it would be done like:
            # student_model.client_encoder.encoder = torch.utils.checkpoint.checkpoint_sequential(
            #    student_model.client_encoder.encoder.blocks,
            #    segments=len(student_model.client_encoder.encoder.blocks),
            #    input=dummy_input_for_shape_tracing # this is tricky with dynamic inputs
            # )
            # This specific part of checkpointing requires careful thought on model architecture.
            # The previous `checkpoint_wrapper` is not available in newer torch.
            # For now, we will assume checkpointing is handled inside the model if args.use_checkpointing is true.
            pass # Placeholder for more specific checkpointing logic within model definition

    student_model = student_model.to(device)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=lr/10) # T_max to num_epochs

    # Loss functions
    # Assuming input 224x224, stem out H/4 (56x56), encoder out H/8 (28x28).
    # VIB loss is on 'z_likelihoods', which come from z_raw (output of client_encoder).
    # z_raw is the latent representation before VIBBottleneck.
    # In MantisStage1, client_encoder (MantisStage1Client) outputs z.
    # MantisStage1Client: self.stem -> f_stem (B, stem_channels, H/4, W/4)
    #                     self.encoder(f_stem) -> z (B, encoder_channels, H/8, W/8)
    # So, if input is 224x224, H/8 = 28. num_pixels_placeholder should be 28*28 = 784.
    vib_loss_fn = VIBLossStage1(num_pixels_placeholder=28*28)
    mse_loss_fn = nn.MSELoss()

    print("✓ Training components setup complete")
    return optimizer, scheduler, vib_loss_fn, mse_loss_fn


def train_epoch(student_model, teacher_model, teacher_features, train_loader,
                optimizer, vib_loss_fn, mse_loss_fn, beta_stage1, epoch, device, log_freq,
                grad_accumulation_steps, use_amp, scaler, monitor_memory, is_ddp, rank):
    """Train for one epoch."""
    student_model.train()
    teacher_model.eval()

    running_loss = 0.0
    running_hd_loss = 0.0
    running_vib_loss = 0.0

    adaptive_pool = None
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]', unit="batch", disable=args.profile_batches > 0 and epoch==0)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True) # non_blocking for pinned memory

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_layer2_features = teacher_model(images)

        # Student forward pass with AMP context
        with autocast(enabled=use_amp):
            student_outputs = student_model(images) # This might need args.use_checkpointing if handled inside
            student_features = student_outputs['g_s_output']
            teacher_target = teacher_layer2_features

            if student_features.shape != teacher_target.shape:
                if adaptive_pool is None: # Initialize adaptive_pool only once if needed
                    target_spatial_size = teacher_target.shape[-2:]
                    print(f"Adapting student features from {student_features.shape} to {teacher_target.shape} using target size {target_spatial_size}")
                    adaptive_pool = nn.AdaptiveAvgPool2d(target_spatial_size).to(device)
                student_features_adapted = adaptive_pool(student_features)
            else:
                student_features_adapted = student_features

            vib_loss_value = vib_loss_fn(student_outputs['z_likelihoods'], None)
            hd_loss_value = mse_loss_fn(student_features_adapted, teacher_target)
            # Loss for this batch, to be divided by grad_accumulation_steps before backward
            current_loss_value = (hd_loss_value + beta_stage1 * vib_loss_value)

        # Scale loss and backward pass
        loss_to_backward = current_loss_value / grad_accumulation_steps
        if use_amp:
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        if (batch_idx + 1) % grad_accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True) # set_to_none can save memory

        # Accumulate losses for epoch average (use original non-accumulated loss for logging)
        actual_total_loss = current_loss_value.item() # Log the loss for this step
        running_loss += actual_total_loss
        running_hd_loss += hd_loss_value.item()
        running_vib_loss += vib_loss_value.item()

        memory_info = {}
        if monitor_memory and torch.cuda.is_available():
            memory_info = get_gpu_memory_info()

        progress_info = {
            'Loss': f'{actual_total_loss:.4f}', # current batch loss
            'HD': f'{hd_loss_value.item():.4f}',
            'VIB': f'{vib_loss_value.item():.2f}',
            'AvgLoss': f'{running_loss / (batch_idx + 1):.4f}' # running average
        }
        if monitor_memory and memory_info:
            # Example: pick one metric or summarize; memory_info can be large
            allocated_gb = memory_info.get(f'gpu{torch.cuda.current_device()}_allocated_gb', 0)
            progress_info['GPU'] = f'{allocated_gb:.1f}GB'
        pbar.set_postfix(progress_info)

        if (batch_idx + 1) % log_freq == 0 and log_freq > 0 and hasattr(train_epoch, 'use_wandb') and train_epoch.use_wandb and rank == 0:
            step = epoch * len(train_loader) + batch_idx + 1
            log_dict_train = {
                'train/batch_loss': actual_total_loss,
                'train/batch_hd_loss': hd_loss_value.item(),
                'train/batch_vib_loss': vib_loss_value.item(),
                'train/learning_rate': optimizer.param_groups[0]['lr']
            }
            if monitor_memory and memory_info:
                log_dict_train.update(memory_info)
            wandb.log(log_dict_train, step=step)

        if monitor_memory and (batch_idx + 1) % 200 == 0: # Less frequent cache clearing
            torch.cuda.empty_cache()
        
        # For profiler step
        if args.profile_batches > 0 and epoch == 0 and batch_idx < args.profile_batches :
            if 'prof' in globals() and prof is not None: # Check if prof is defined
                 prof.step()


    # Final optimizer step if the number of batches is not perfectly divisible by accumulation steps
    if len(train_loader) % grad_accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_epoch_loss = running_loss / len(train_loader)
    avg_epoch_hd_loss = running_hd_loss / len(train_loader)
    avg_epoch_vib_loss = running_vib_loss / len(train_loader)

    if is_ddp:
        train_loss_tensor = torch.tensor(avg_epoch_loss, device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        avg_epoch_loss = train_loss_tensor.item()

        train_hd_tensor = torch.tensor(avg_epoch_hd_loss, device=device)
        dist.all_reduce(train_hd_tensor, op=dist.ReduceOp.AVG)
        avg_epoch_hd_loss = train_hd_tensor.item()

        train_vib_tensor = torch.tensor(avg_epoch_vib_loss, device=device)
        dist.all_reduce(train_vib_tensor, op=dist.ReduceOp.AVG)
        avg_epoch_vib_loss = train_vib_tensor.item()

    return avg_epoch_loss, avg_epoch_hd_loss, avg_epoch_vib_loss


def validate(student_model, teacher_model, teacher_features, val_loader,
             vib_loss_fn, mse_loss_fn, beta_stage1, device, epoch, use_amp, is_ddp):
    """Validate the model."""
    student_model.eval()
    teacher_model.eval()

    total_loss = 0.0
    total_hd_loss = 0.0
    total_vib_loss = 0.0
    adaptive_pool_val = None

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Val]', unit="batch")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)

            teacher_layer2_features = teacher_model(images)

            with autocast(enabled=use_amp):
                student_outputs = student_model(images)
                student_features = student_outputs['g_s_output']
                teacher_target = teacher_layer2_features

                if student_features.shape != teacher_target.shape:
                    if adaptive_pool_val is None:
                        target_spatial_size_val = teacher_target.shape[-2:]
                        adaptive_pool_val = nn.AdaptiveAvgPool2d(target_spatial_size_val).to(device)
                    student_features_adapted = adaptive_pool_val(student_features)
                else:
                    student_features_adapted = student_features

                vib_loss_value = vib_loss_fn(student_outputs['z_likelihoods'], None)
                hd_loss_value = mse_loss_fn(student_features_adapted, teacher_target)
                total_loss_value = hd_loss_value + beta_stage1 * vib_loss_value

            total_loss += total_loss_value.item()
            total_hd_loss += hd_loss_value.item()
            total_vib_loss += vib_loss_value.item()

            pbar.set_postfix({
                'Loss': f'{total_loss_value.item():.4f}',
                'HD': f'{hd_loss_value.item():.4f}',
                'VIB': f'{vib_loss_value.item():.2f}',
                'AvgLoss': f'{total_loss / (pbar.n + 1):.4f}'
            })

    avg_loss = total_loss / len(val_loader)
    avg_hd_loss = total_hd_loss / len(val_loader)
    avg_vib_loss = total_vib_loss / len(val_loader)

    if is_ddp:
        val_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = val_loss_tensor.item()

        val_hd_tensor = torch.tensor(avg_hd_loss, device=device)
        dist.all_reduce(val_hd_tensor, op=dist.ReduceOp.AVG)
        avg_hd_loss = val_hd_tensor.item()

        val_vib_tensor = torch.tensor(avg_vib_loss, device=device)
        dist.all_reduce(val_vib_tensor, op=dist.ReduceOp.AVG)
        avg_vib_loss = val_vib_tensor.item()

    return avg_loss, avg_hd_loss, avg_vib_loss


def get_gpu_memory_info():
    """Get GPU memory usage information for the current device."""
    if not torch.cuda.is_available():
        return {}
    
    current_device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(current_device) / 1024**3
    reserved = torch.cuda.memory_reserved(current_device) / 1024**3 # Total memory reserved by allocator
    max_allocated = torch.cuda.max_memory_allocated(current_device) / 1024**3 # Peak allocated
    # max_reserved = torch.cuda.max_memory_reserved(current_device) / 1024**3 # Peak reserved
    
    props = torch.cuda.get_device_properties(current_device)
    total_memory = props.total_memory / 1024**3

    # Reset peak stats for next measurement interval if desired, but usually not needed per call
    # torch.cuda.reset_peak_memory_stats(current_device)

    return {
        f'gpu{current_device}_allocated_gb': allocated,
        f'gpu{current_device}_reserved_gb': reserved,
        f'gpu{current_device}_max_allocated_gb': max_allocated,
        # f'gpu{current_device}_max_reserved_gb': max_reserved,
        f'gpu{current_device}_total_memory_gb': total_memory,
        f'gpu{current_device}_free_approx_gb': total_memory - reserved # Free in PyTorch's view
    }

def main():
    """Main training function."""
    global args, prof
    prof = None
    parser = get_argparser()
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        is_ddp = True
        print(f"DDP enabled: Rank {rank}/{world_size} on GPU {local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        is_ddp = False
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, switching to CPU")
            args.device = 'cpu'
        elif args.device == 'cuda':
             torch.cuda.set_device(0)
             args.device = 'cuda:0'

    torch.backends.cudnn.benchmark = True

    args.data_dir = os.path.expanduser(args.data_dir)
    args.save_dir = os.path.expanduser(args.save_dir)

    if args.use_amp and args.device == 'cpu':
        if rank == 0:
            print("Warning: AMP is requested but device is CPU. AMP will not be used.")
        args.use_amp = False

    scaler = GradScaler(enabled=args.use_amp) # Moved scaler init here

    if args.use_wandb and rank == 0:
        wandb_config = vars(args).copy()
        if torch.cuda.is_available():
             gpu_info_initial = get_gpu_memory_info()
             wandb_config.update(gpu_info_initial)

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            config=wandb_config
        )
        train_epoch.use_wandb = True
        print("✓ Wandb initialized")
    else:
        train_epoch.use_wandb = False

    if rank == 0:
        print(f"MANTiS Stage 1 Training - PID: {os.getpid()}")
        print(f"Torch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Using device: {args.device}")
        print(f"Data directory: {args.data_dir}")
        print(f"Batch size: {args.batch_size}, Grad Acc Steps: {args.grad_accumulation_steps}, Effective BS: {args.batch_size * args.grad_accumulation_steps}")
        print(f"Num workers: {args.num_workers}, Prefetch factor: {args.prefetch_factor}")

        if args.use_amp:
            print("Using Automatic Mixed Precision (AMP)")
        if args.use_checkpointing:
            print("Gradient Checkpointing requested (ensure model internals support it or are wrapped).")


    os.makedirs(args.save_dir, exist_ok=True)

    if rank == 0:
        print("\nLoading data...")
    train_loader, val_loader = create_imagenet_webdataset_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        image_size=224
    )
    if rank == 0:
        print(f"Train loader approx batches (per GPU): {len(train_loader)}, Val loader approx batches (per GPU): {len(val_loader)}")
        print(f"Global batch size: {args.batch_size * world_size}")

    teacher_model, student_model, teacher_features = setup_models(args.device, args.use_checkpointing)

    # Add checkpoint loading logic here if resuming from checkpoint
    # This should be done before DDP wrapping
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_checkpoint:
        if rank == 0:
            print(f"Loading checkpoint from {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device)
        
        # Handle 'module.' prefix if checkpoint was saved from DDP model
        state_dict = checkpoint['model_state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
            new_state_dict[name] = v
        
        student_model.load_state_dict(new_state_dict)  # Load into model before DDP wrapping
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        
        if is_ddp:
            dist.barrier()  # Ensure all processes load before proceeding

    # Debug: Print parameter info to identify unused parameters
    if rank == 0:
        print("Model parameters:")
        for i, (name, param) in enumerate(student_model.named_parameters()):
            if i == 45:  # The problematic index from error message
                print(f"----> Index {i}: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
            # Uncomment below to see all parameters for context
            # else:
            #     print(f"      Index {i}: {name}")

    if is_ddp:
        student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer, scheduler, vib_loss_fn, mse_loss_fn = setup_training(
        student_model.module if isinstance(student_model, DDP) else student_model,
        args.lr, args.weight_decay
    )

    # Load optimizer and scheduler state if resuming
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if args.use_amp and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if rank == 0:
            print("Optimizer, scheduler, and scaler states loaded")

    if rank == 0:
        print(f"\nStarting training for {args.num_epochs} epochs...")
    best_val_loss = float('inf')

    if args.profile_batches > 0 and args.device == 'cuda' and rank == 0:
        print(f"Profiling the first {args.profile_batches} batches of epoch 0...")
        profiler_log_dir = os.path.join(args.save_dir, 'profiler_traces')
        os.makedirs(profiler_log_dir, exist_ok=True)

        prof = torch.profiler.profile(
             activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
             schedule=torch.profiler.schedule(wait=1, warmup=1, active=args.profile_batches, repeat=1),
             on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_log_dir),
             record_shapes=True,
             profile_memory=True,
             with_stack=True
        )
        prof.start()
        print(f"Profiler started. Traces will be saved to {profiler_log_dir}")


    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()

        train_loss, train_hd, train_vib = train_epoch(
            student_model, teacher_model, teacher_features, train_loader,
            optimizer, vib_loss_fn, mse_loss_fn, args.beta_stage1, epoch,
            args.device, args.log_freq, args.grad_accumulation_steps,
            args.use_amp, scaler, args.monitor_memory, is_ddp, rank
        )

        val_loss, val_hd, val_vib = validate(
            student_model, teacher_model, teacher_features, val_loader,
            vib_loss_fn, mse_loss_fn, args.beta_stage1, args.device, epoch, args.use_amp, is_ddp
        )

        scheduler.step()
        epoch_time_taken = time.time() - epoch_start_time

        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.num_epochs} completed in {epoch_time_taken:.1f}s")
            print(f"  Train - Loss: {train_loss:.4f}, HD: {train_hd:.4f}, VIB: {train_vib:.2f}")
            print(f"  Val   - Loss: {val_loss:.4f}, HD: {val_hd:.4f}, VIB: {val_vib:.2f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6e}")

        if args.use_wandb and rank == 0:
            log_dict_epoch = {
                'epoch': epoch + 1,
                'train/epoch_loss': train_loss,
                'train/epoch_hd_loss': train_hd,
                'train/epoch_vib_loss': train_vib,
                'val/epoch_loss': val_loss,
                'val/epoch_hd_loss': val_hd,
                'val/epoch_vib_loss': val_vib,
                'train/learning_rate_epoch': optimizer.param_groups[0]['lr'],
                'time/epoch_time_seconds': epoch_time_taken,
            }
            if args.monitor_memory and torch.cuda.is_available():
                log_dict_epoch.update(get_gpu_memory_info())
            wandb.log(log_dict_epoch)

        if (epoch + 1) % args.save_freq == 0 or val_loss < best_val_loss:
            if rank == 0:
                model_to_save = student_model.module if isinstance(student_model, DDP) else student_model
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'args': args,
                    'scaler_state_dict': scaler.state_dict() if args.use_amp else None,
                }
                
                save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save(checkpoint, save_path)
                print(f"  Checkpoint saved to {save_path}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(args.save_dir, 'best_model.pth')
                    torch.save(checkpoint, best_path)
                    print(f"  ✓ New best model saved (val_loss: {val_loss:.4f}) at {best_path}")

        if prof and epoch == 0 and args.profile_batches > 0 and rank == 0:
            if args.profile_batches > 0 and epoch == 0:
                 print("Exiting after profiling epoch 0 as per configuration (remove if multi-epoch profiling needed).")


    if prof and prof.profiler is not None and rank == 0:
        prof.stop()
        print(f"Profiling finished. Traces saved to {os.path.join(args.save_dir, 'profiler_traces')}")

    if rank == 0:
        print(f"\n🎉 Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {args.save_dir}")

    if args.use_wandb and rank == 0:
        wandb.log({'model/final_best_val_loss': best_val_loss})
        wandb.finish()
        print("✓ Wandb run completed")

    if is_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    # Note: If using torch.utils.checkpoint, DataParallel can sometimes interact unexpectedly.
    # It's often recommended to apply DataParallel as the outermost wrapper.
    # Ensure checkpointing logic is compatible with how DataParallel splits batches across GPUs.
    main()