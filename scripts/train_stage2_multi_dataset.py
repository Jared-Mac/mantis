#!/usr/bin/env python3
"""
MANTiS Stage 2 Training with Multi-Dataset (CIFAR-100, STL-10, Flowers-102).
"""

import sys
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import time
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import yaml
from types import SimpleNamespace
import json

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import registry
from multi_dataset_wrapper import (
    create_multi_task_datasets, 
    create_task_definitions_multi_dataset,
    multi_dataset_collate_fn
)
from models import MantisStage2
from losses import CombinedMantisLoss


def setup_models_and_load_checkpoint(config, device, task_definitions, rank=0):
    if rank == 0:
        print("Setting up Stage 2 model from config...")
    model_cfg = config.model.config
    num_tasks = model_cfg.num_tasks
    task_names = list(task_definitions.keys())
    
    # --- Client Parameters ---
    client_params = {
        'stem_params': {
            'input_channels': 3,
            'output_channels': model_cfg.stem_channels,
        },
        'task_detector_params': {
            'input_channels': model_cfg.stem_channels,
            'num_tasks': num_tasks
        },
        'film_gen_params': {
            'num_tasks': num_tasks,
            'film_channels': model_cfg.encoder_output_channels,
            'hidden_dim': model_cfg.film_gen_hidden_dim
        },
        'filmed_encoder_params': {
            'input_channels': model_cfg.stem_channels,
            'latent_channels': model_cfg.encoder_output_channels,
        }
    }

    # Task-specific decoder parameters (same for all tasks)
    decoder_params_list = [
        {
            'input_channels': model_cfg.vib_channels, 
            'output_channels': model_cfg.decoder_output_channels
        }
        for _ in range(num_tasks)
    ]

    # Task-specific tail parameters
    tail_params_list = [
        {
            'input_channels': model_cfg.decoder_output_channels,
            'num_classes': len(task_definitions[task_name])
        }
        for task_name in task_names
    ]

    # Create the model
    model = MantisStage2(
        client_params=client_params,
        num_tasks=num_tasks,
        decoder_params_list=decoder_params_list,
        tail_params_list=tail_params_list,
        vib_channels=model_cfg.vib_channels
    )

    if rank == 0:
        print(f"Loading Stage 1 checkpoint from: {config.model.stage1_checkpoint_path}")
    model.load_stage1_weights(config.model.stage1_checkpoint_path, rank)
    
    model.load_stage1_decoder_weights(config.model.stage1_checkpoint_path, rank)
    
    # Note: For multi-dataset, we don't load teacher tail weights since each dataset
    # has its own label space. The tails will be randomly initialized.
    if rank == 0:
        print("Note: Tail weights are randomly initialized for multi-dataset training.")

    if config.model.freeze_stem:
        if rank == 0:
            print("Freezing stem parameters.")
        for param in model.client.stem.parameters():
            param.requires_grad = False
    
    model = model.to(device)
    if rank == 0:
        print(f"✓ MantisStage2 model setup complete. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def setup_optimizer(model, config):
    main_lr = config.training.optimizer.main_lr
    backbone_lr = config.training.optimizer.backbone_lr
    weight_decay = config.training.optimizer.weight_decay
    freeze_stem = config.model.freeze_stem
    
    param_groups = [
        {'params': [], 'lr': main_lr, 'name': 'client_stem'},
        {'params': [], 'lr': main_lr, 'name': 'client_encoder_convs'},
        {'params': [], 'lr': main_lr, 'name': 'client_new_modules'},
        {'params': [], 'lr': main_lr, 'name': 'server_modules'},
        {'params': [], 'lr': main_lr, 'name': 'vib_bottleneck_film'},
        {'params': [], 'lr': main_lr, 'name': 'default_group'}
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if name.startswith('client.stem.'):
            if not freeze_stem: 
                param_groups[0]['params'].append(param)
        elif name.startswith('client.filmed_encoder.'):
            param_groups[1]['params'].append(param)
        elif name.startswith('client.task_detector.') or name.startswith('client.film_generator.'):
            param_groups[2]['params'].append(param)
        elif name.startswith('server_decoders.') or name.startswith('server_tails.'):
            param_groups[3]['params'].append(param)
        elif name.startswith('vib_bottleneck_film.'):
            param_groups[4]['params'].append(param)
        else:
            param_groups[5]['params'].append(param) 

    final_param_groups = [pg for pg in param_groups if len(pg['params']) > 0]
    
    if not final_param_groups:
        raise ValueError("No parameters to optimize. Check model structure and freeze_stem flag.")

    if dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized():
        for pg in final_param_groups:
            print(f"Optimizer group: {pg['name']}, LR: {pg['lr']}, Num params: {sum(p.numel() for p in pg['params'])}")

    optimizer = torch.optim.AdamW(final_param_groups, weight_decay=weight_decay)
    return optimizer


def setup_training_components(student_model, config, num_tasks, task_loss_configs):
    print("Setting up training components (optimizer, scheduler, loss)...")
    
    optimizer = setup_optimizer(
        student_model.module if isinstance(student_model, DDP) else student_model,
        config
    )
    
    scheduler = None
    if hasattr(config.training, 'scheduler') and config.training.scheduler:
        scheduler_config = config.training.scheduler
        scheduler_type = scheduler_config.type
        
        try:
            scheduler_params = vars(scheduler_config.params)
        except TypeError:
            scheduler_params = {}

        if scheduler_type == 'CosineAnnealingLR' and 'T_max' not in scheduler_params:
            scheduler_params['T_max'] = config.training.num_epochs
            print(f"Scheduler T_max not specified, defaulting to num_epochs: {config.training.num_epochs}")

        try:
            SchedulerClass = getattr(torch.optim.lr_scheduler, scheduler_type)
            scheduler = SchedulerClass(optimizer, **scheduler_params)
            print(f"✓ Scheduler '{scheduler_type}' initialized with params: {scheduler_params}")
        except AttributeError:
            print(f"Error: Scheduler type '{scheduler_type}' not found in torch.optim.lr_scheduler.")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)
            print("Defaulting to CosineAnnealingLR with T_max=num_epochs.")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.num_epochs, eta_min=config.training.optimizer.main_lr / 100)
        print("✓ Scheduler section not found in config, using default CosineAnnealingLR.")

    loss_weights = config.training.loss_weights
    combined_loss_fn = CombinedMantisLoss(
        task_detector_loss_weight=loss_weights.lambda_task,
        downstream_loss_weight=loss_weights.lambda_downstream,
        rate_loss_weight=loss_weights.beta_prime,
        num_tasks=num_tasks,
        task_loss_configs=task_loss_configs
    )
    print("✓ Training components setup complete.")
    return optimizer, scheduler, combined_loss_fn


def train_epoch(student_model, train_loader, optimizer, combined_loss_fn, epoch_idx, device, log_freq,
                grad_accumulation_steps, use_amp, scaler, monitor_memory, is_ddp, rank, num_epochs_total, profile_batches, gradient_clip_val):
    student_model.train()
    
    running_total_loss = 0.0
    running_task_loss = 0.0
    running_downstream_loss = 0.0
    running_rate_loss = 0.0
    
    optimizer.zero_grad()

    global prof
    pbar_desc = f'Epoch {epoch_idx+1}/{num_epochs_total} [Train]'
    pbar_disabled = (rank != 0) or (profile_batches > 0 and epoch_idx == 0 and rank == 0)
    pbar = tqdm(train_loader, desc=pbar_desc, unit="batch", disable=pbar_disabled)

    for batch_idx, (images, targets_dict) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        
        targets_for_loss_fn = {
            'Y_task': targets_dict['Y_task'].to(device, non_blocking=True),
            'Y_downstream': [t.to(device, non_blocking=True) for t in targets_dict['Y_downstream']]
        }

        with autocast(device_type=device.split(':')[0], dtype=torch.float16, enabled=(use_amp and 'cuda' in device)): 
            student_outputs = student_model(images) 
            loss_components = combined_loss_fn(student_outputs, targets_for_loss_fn)
            current_total_loss = loss_components['total_loss']
        
        loss_to_backward = current_total_loss / grad_accumulation_steps
        if use_amp and device != 'cpu': 
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        if (batch_idx + 1) % grad_accumulation_steps == 0:
            if use_amp and device != 'cpu':
                scaler.unscale_(optimizer)
                if gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:
                if gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), gradient_clip_val)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        running_total_loss += current_total_loss.item()
        running_task_loss += loss_components['task_detector_loss'].item()
        running_downstream_loss += loss_components['downstream_loss'].item()
        running_rate_loss += loss_components['rate_loss'].item()
        
        if rank == 0: 
            pbar_postfix = {
                'Loss': f'{current_total_loss.item():.4f}',
                'Task': f'{loss_components["task_detector_loss"].item():.4f}',
                'Down': f'{loss_components["downstream_loss"].item():.4f}',
                'Rate': f'{loss_components["rate_loss"].item():.2f}'
            }
            if monitor_memory and torch.cuda.is_available():
                allocated_gb = get_gpu_memory_info().get(f'gpu{torch.cuda.current_device()}_allocated_gb',0)
                pbar_postfix['GPU'] = f'{allocated_gb:.1f}GB'
            pbar.set_postfix(pbar_postfix)

        if (batch_idx + 1) % log_freq == 0 and log_freq > 0 and hasattr(train_epoch, 'use_wandb') and train_epoch.use_wandb and rank == 0:
            step = epoch_idx * len(train_loader) + batch_idx + 1
            log_data_train = {
                'train/batch_total_loss': current_total_loss.item(),
                'train/batch_task_loss': loss_components['task_detector_loss'].item(),
                'train/batch_downstream_loss': loss_components['downstream_loss'].item(),
                'train/batch_rate_loss': loss_components['rate_loss'].item(),
                'train/learning_rate_group0': optimizer.param_groups[0]['lr'] 
            }
            if monitor_memory and torch.cuda.is_available(): log_data_train.update(get_gpu_memory_info())
            wandb.log(log_data_train, step=step)

        if monitor_memory and (batch_idx + 1) % 200 == 0 and rank == 0:
            if torch.cuda.is_available(): torch.cuda.empty_cache() 
        
        if profile_batches > 0 and epoch_idx == 0 and batch_idx < profile_batches and rank==0 :
             if prof is not None: prof.step()

    if len(train_loader) > 0 and len(train_loader) % grad_accumulation_steps != 0 : 
        if use_amp and device != 'cpu': scaler.step(optimizer); scaler.update()
        else: optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_losses = {
        'total': running_total_loss / len(train_loader) if len(train_loader) > 0 else 0,
        'task': running_task_loss / len(train_loader) if len(train_loader) > 0 else 0,
        'downstream': running_downstream_loss / len(train_loader) if len(train_loader) > 0 else 0,
        'rate': running_rate_loss / len(train_loader) if len(train_loader) > 0 else 0
    }
    
    if is_ddp: 
        for key in avg_losses:
            loss_tensor = torch.tensor(avg_losses[key], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_losses[key] = loss_tensor.item()
            
    return avg_losses


def validate_epoch(student_model, val_loader, combined_loss_fn, epoch_idx, device, use_amp, is_ddp, rank, num_epochs_total):
    student_model.eval()
    
    total_loss_val = 0.0
    total_task_loss_val = 0.0
    total_downstream_loss_val = 0.0
    total_rate_loss_val = 0.0
    
    with torch.no_grad():
        pbar_desc = f'Epoch {epoch_idx+1}/{num_epochs_total} [Val]'
        pbar = tqdm(val_loader, desc=pbar_desc, unit="batch", disable=(rank!=0))
        for images, targets_dict in pbar:
            images = images.to(device, non_blocking=True)
            
            targets_for_loss_fn_val = {
                'Y_task': targets_dict['Y_task'].to(device, non_blocking=True),
                'Y_downstream': [t.to(device, non_blocking=True) for t in targets_dict['Y_downstream']]
            }
            
            with autocast(device_type=device.split(':')[0], dtype=torch.float16, enabled=(use_amp and 'cuda' in device)): 
                student_outputs = student_model(images)
                loss_components = combined_loss_fn(student_outputs, targets_for_loss_fn_val)
            
            total_loss_val += loss_components['total_loss'].item()
            total_task_loss_val += loss_components['task_detector_loss'].item()
            total_downstream_loss_val += loss_components['downstream_loss'].item()
            total_rate_loss_val += loss_components['rate_loss'].item()

            if rank == 0:
                pbar.set_postfix({
                    'Loss': f'{loss_components["total_loss"].item():.4f}',
                    'AvgLoss': f'{total_loss_val / (pbar.n + 1):.4f}'
                })

    avg_losses_val = {
        'total': total_loss_val / len(val_loader) if len(val_loader) > 0 else 0,
        'task': total_task_loss_val / len(val_loader) if len(val_loader) > 0 else 0,
        'downstream': total_downstream_loss_val / len(val_loader) if len(val_loader) > 0 else 0,
        'rate': total_rate_loss_val / len(val_loader) if len(val_loader) > 0 else 0
    }

    if is_ddp: 
        for key in avg_losses_val:
            loss_tensor = torch.tensor(avg_losses_val[key], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_losses_val[key] = loss_tensor.item()
            
    return avg_losses_val


def get_gpu_memory_info(): 
    if not torch.cuda.is_available(): return {}
    current_device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(current_device) / 1024**3
    reserved = torch.cuda.memory_reserved(current_device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(current_device) / 1024**3
    props = torch.cuda.get_device_properties(current_device)
    total_memory = props.total_memory / 1024**3
    return {
        f'gpu{current_device}_allocated_gb': allocated,
        f'gpu{current_device}_reserved_gb': reserved,
        f'gpu{current_device}_max_allocated_gb': max_allocated,
        f'gpu{current_device}_total_memory_gb': total_memory,
    }


def main():
    global prof 
    prof = None
    parser = argparse.ArgumentParser(description='MANTiS Stage 2 Multi-Dataset Training')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    cli_args = parser.parse_args()

    with open(cli_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = json.loads(json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d))

    # DDP Setup
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        is_ddp = True
        if rank == 0: print(f"DDP enabled: Rank {rank}/{world_size} on GPU {local_rank}")
    else:
        rank = 0; world_size = 1; local_rank = 0; is_ddp = False
        device = config.training.device
        if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'
        elif device == 'cuda': torch.cuda.set_device(0); device = 'cuda:0'
    
    use_amp = config.training.use_amp
    scaler_enabled = use_amp and (device != 'cpu') 
    scaler = GradScaler(enabled=scaler_enabled)

    torch.backends.cudnn.benchmark = True
    config.data.data_root = os.path.expanduser(config.data.data_root)
    config.project.save_dir = os.path.expanduser(config.project.save_dir)
    if use_amp and device == 'cpu': 
        use_amp = False 
        if rank == 0: print("AMP disabled for CPU.")
    
    if config.project.use_wandb and rank == 0:
        wandb_config = config_dict
        if torch.cuda.is_available(): wandb_config.update(get_gpu_memory_info())
        wandb_run_name = config.project.wandb_run_name if config.project.wandb_run_name else f"stage2_multi_dataset_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=config.project.wandb_project, name=wandb_run_name, tags=config.project.wandb_tags, config=wandb_config)
        if not hasattr(train_epoch, 'use_wandb'): 
             train_epoch.use_wandb = True
        print("✓ Wandb initialized")
    else:
        if not hasattr(train_epoch, 'use_wandb'):
            train_epoch.use_wandb = False

    if rank == 0:
        print(f"MANTiS Stage 2 Multi-Dataset Training - PID: {os.getpid()}")
        print(f"Loaded configuration from: {cli_args.config}")
        print(f"Torch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}") 
            print(f"CuDNN version: {torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, 'version') else 'N/A'}") 
            print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'N/A'}")
        print(f"Using device: {device}")
        print(f"Data root: {config.data.data_root}")
        print(f"Batch size (per GPU): {config.training.batch_size}, Grad Acc Steps: {config.training.grad_accumulation_steps}")
        print(f"Effective global batch size: {config.training.batch_size * world_size * config.training.grad_accumulation_steps}")
        print(f"Num workers: {config.data.num_workers}, Prefetch factor: {config.data.prefetch_factor}")
        if use_amp and device != 'cpu': print("Using Automatic Mixed Precision (AMP) on CUDA")
        elif use_amp and device == 'cpu': print("AMP requested for CPU but will be disabled.")
        if config.model.freeze_stem: print("Stem parameters will be frozen.")

    os.makedirs(config.project.save_dir, exist_ok=True)

    # Load multi-dataset
    task_definitions = create_task_definitions_multi_dataset()
    num_tasks = len(task_definitions)
    task_names = list(task_definitions.keys())
    if rank == 0: print(f"Defined {num_tasks} tasks: {task_names}")

    if rank == 0: print("\nLoading multi-dataset...")
    
    datasets_info = create_multi_task_datasets(
        data_root=config.data.data_root,
        image_size=config.data.image_size,
        download=config.data.download_datasets
    )
    
    train_loader = DataLoader(
        datasets_info['train'], 
        batch_size=config.training.batch_size, 
        num_workers=config.data.num_workers,
        persistent_workers=True if config.data.num_workers > 0 else False, 
        pin_memory=True,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
        drop_last=True, 
        shuffle=True,
        collate_fn=multi_dataset_collate_fn 
    )
    
    val_loader = DataLoader(
        datasets_info['val'], 
        batch_size=config.training.batch_size, 
        num_workers=config.data.num_workers,
        persistent_workers=True if config.data.num_workers > 0 else False, 
        pin_memory=True,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
        drop_last=False, 
        shuffle=False,
        collate_fn=multi_dataset_collate_fn 
    )

    if rank == 0: 
        print(f"Train/Val loaders created. Batches per GPU: {len(train_loader)}/{len(val_loader)}")
        print(f"Classes per task: {datasets_info['num_classes']}")

    student_model = setup_models_and_load_checkpoint(config, device, task_definitions, rank)
    
    start_epoch = 0
    best_val_loss = float('inf')

    if config.model.resume_stage2_checkpoint:
        if rank == 0: print(f"Resuming Stage 2 training from {config.model.resume_stage2_checkpoint}")
        checkpoint = torch.load(config.model.resume_stage2_checkpoint, map_location=device, weights_only=False)
        
        state_dict = checkpoint['model_state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        student_model.load_state_dict(new_state_dict)
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        if rank == 0: print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        if is_ddp: dist.barrier()

    if is_ddp:
        student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    task_loss_configs = [{'type': 'CrossEntropyLoss', 'params': {'ignore_index': -100}} for _ in range(num_tasks)]
    
    optimizer, scheduler, combined_loss_fn = setup_training_components(
         student_model, config, num_tasks, task_loss_configs
    )

    if config.model.resume_stage2_checkpoint: 
        checkpoint = torch.load(config.model.resume_stage2_checkpoint, map_location=device, weights_only=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if use_amp and scaler.is_enabled() and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if rank == 0: print("Optimizer, scheduler, and scaler states loaded for Stage 2 resumption.")

    if rank == 0: print(f"\nStarting Stage 2 multi-dataset training for {config.training.num_epochs} epochs...")
    
    log_saving_cfg = config.logging_and_saving
    if log_saving_cfg.profile_batches > 0 and device.startswith('cuda') and rank == 0: 
        profiler_log_dir = os.path.join(config.project.save_dir, 'profiler_traces_stage2')
        os.makedirs(profiler_log_dir, exist_ok=True)
        prof = torch.profiler.profile(
             activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
             schedule=torch.profiler.schedule(wait=1, warmup=1, active=log_saving_cfg.profile_batches, repeat=1),
             on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_log_dir),
             record_shapes=True, profile_memory=True, with_stack=True)
        if prof: prof.start() 
        print(f"Profiler started. Traces will be saved to {profiler_log_dir}")

    for epoch in range(start_epoch, config.training.num_epochs):
        epoch_start_time = time.time()

        gradient_clip_val = config.training.optimizer.gradient_clip_val if hasattr(config.training.optimizer, 'gradient_clip_val') else 0.0

        train_losses = train_epoch(
            student_model, train_loader, optimizer, combined_loss_fn, epoch, device, log_saving_cfg.log_freq,
            config.training.grad_accumulation_steps, use_amp, scaler, log_saving_cfg.monitor_memory, is_ddp, rank, config.training.num_epochs, log_saving_cfg.profile_batches,
            gradient_clip_val
        )
        val_losses = validate_epoch(
            student_model, val_loader, combined_loss_fn, epoch, device, use_amp, is_ddp, rank, config.training.num_epochs
        )
        scheduler.step()
        epoch_time_taken = time.time() - epoch_start_time

        if rank == 0:
            print(f"\nEpoch {epoch+1}/{config.training.num_epochs} completed in {epoch_time_taken:.1f}s")
            print(f"  Train - Loss: {train_losses['total']:.4f} (Task: {train_losses['task']:.4f}, Down: {train_losses['downstream']:.4f}, Rate: {train_losses['rate']:.2f})")
            print(f"  Val   - Loss: {val_losses['total']:.4f} (Task: {val_losses['task']:.4f}, Down: {val_losses['downstream']:.4f}, Rate: {val_losses['rate']:.2f})")
            current_lr_example = optimizer.param_groups[0]['lr'] 
            print(f"  LR (group 0): {current_lr_example:.6e}")

            if hasattr(train_epoch, 'use_wandb') and train_epoch.use_wandb:
                log_dict_epoch = {
                    'epoch': epoch + 1,
                    'train/epoch_total_loss': train_losses['total'], 'train/epoch_task_loss': train_losses['task'],
                    'train/epoch_downstream_loss': train_losses['downstream'], 'train/epoch_rate_loss': train_losses['rate'],
                    'val/epoch_total_loss': val_losses['total'], 'val/epoch_task_loss': val_losses['task'],
                    'val/epoch_downstream_loss': val_losses['downstream'], 'val/epoch_rate_loss': val_losses['rate'],
                    'train/learning_rate_group0_epoch': current_lr_example,
                    'time/epoch_time_seconds': epoch_time_taken,
                }
                if log_saving_cfg.monitor_memory and torch.cuda.is_available(): log_dict_epoch.update(get_gpu_memory_info())
                wandb.log(log_dict_epoch)

            is_best = val_losses['total'] < best_val_loss
            if is_best:
                best_val_loss = val_losses['total']

            if (epoch + 1) % log_saving_cfg.save_freq == 0 or is_best:
                model_to_save = student_model.module if isinstance(student_model, DDP) else student_model
                checkpoint = {
                    'epoch': epoch + 1, 'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_losses['total'], 'config': config_dict, 
                    'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
                }

                latest_save_path = os.path.join(config.project.save_dir, 'latest_checkpoint.pth')
                torch.save(checkpoint, latest_save_path)
                print(f"  Saved latest checkpoint to {latest_save_path}")

                if (epoch + 1) % log_saving_cfg.save_freq == 0:
                    save_path = os.path.join(config.project.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                    torch.save(checkpoint, save_path)
                    print(f"  Checkpoint saved to {save_path}")
                
                if is_best:
                    best_path = os.path.join(config.project.save_dir, 'best_model_stage2.pth')
                    torch.save(checkpoint, best_path)
                    print(f"  ✓ New best model saved (val_loss: {best_val_loss:.4f}) at {best_path}")
        
        if prof and epoch == 0 and log_saving_cfg.profile_batches > 0 and rank==0: 
            print("Exiting after profiling first epoch as per configuration.")
            break 

    if prof and hasattr(prof, 'profiler') and prof.profiler is not None and rank == 0: 
        prof.stop()
        print(f"Profiling finished. Traces saved to {os.path.join(config.project.save_dir, 'profiler_traces_stage2')}")

    if rank == 0:
        print(f"\n🎉 Stage 2 Multi-Dataset Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {config.project.save_dir}")
        if hasattr(train_epoch, 'use_wandb') and train_epoch.use_wandb:
            wandb.log({'model/final_best_val_loss_stage2': best_val_loss})
            wandb.finish()
            print("✓ Wandb run completed")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main() 