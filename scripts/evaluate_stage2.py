#!/usr/bin/env python3
"""
MANTiS Stage 2 Evaluation Script.

This script loads a trained Stage 2 model and evaluates its performance on the
ImageNet validation set, measuring per-task accuracy and average BPP.
"""

import sys
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import yaml
from types import SimpleNamespace
import json
import numpy as np

# Add src directory to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from webdataset_wrapper import ImageNetWebDataset
from datasets import create_imagenet_task_definitions
from models import MantisStage2


# --- Helper Functions (from train_stage2_webdataset.py) ---

def create_task_processor_fn(task_definitions_dict):
    """Creates a function to process raw class labels into MANTiS multi-task targets."""
    task_names_list = list(task_definitions_dict.keys())
    num_tasks = len(task_names_list)
    
    class_to_tasks_map = {}
    for task_idx, (task_name, class_indices) in enumerate(task_definitions_dict.items()):
        for class_idx in class_indices:
            if class_idx not in class_to_tasks_map:
                class_to_tasks_map[class_idx] = []
            class_to_tasks_map[class_idx].append(task_idx)
            
    task_class_mappings_map = {} 
    for task_idx, (task_name, class_indices) in enumerate(task_definitions_dict.items()):
        task_class_mappings_map[task_idx] = {
            orig_idx: new_idx for new_idx, orig_idx in enumerate(class_indices)
        }

    def process_target(original_class_str):
        original_class = int(original_class_str)
        y_task = torch.zeros(num_tasks, dtype=torch.float32)
        y_downstream = [None] * num_tasks 

        if original_class in class_to_tasks_map:
            active_tasks_for_class = class_to_tasks_map[original_class]
            for task_idx in active_tasks_for_class:
                y_task[task_idx] = 1.0
                if task_idx in task_class_mappings_map:
                    task_specific_class = task_class_mappings_map[task_idx].get(original_class)
                    if task_specific_class is not None:
                         y_downstream[task_idx] = task_specific_class 
        
        return {
            'Y_task': y_task,
            'Y_downstream': y_downstream, 
            'original_class': original_class
        }
    return process_target


def mantis_collate_fn(batch):
    """Custom collate_fn for MANTiS Stage 2, handles None in targets."""
    images = [item[0] for item in batch]
    targets_list = [item[1] for item in batch]

    collated_images = default_collate(images)

    collated_targets = {}
    if targets_list:
        collated_targets['Y_task'] = default_collate([t['Y_task'] for t in targets_list])
        collated_targets['original_class'] = default_collate([t['original_class'] for t in targets_list])

        num_tasks = len(targets_list[0]['Y_downstream'])
        y_downstream_collated_per_task = [[] for _ in range(num_tasks)]
        
        for sample_targets in targets_list:
            for task_idx in range(num_tasks):
                label = sample_targets['Y_downstream'][task_idx]
                y_downstream_collated_per_task[task_idx].append(label if label is not None else -100)
        
        collated_targets['Y_downstream'] = [
            torch.tensor(task_labels, dtype=torch.long) for task_labels in y_downstream_collated_per_task
        ]
    
    return collated_images, collated_targets


# --- Evaluation Function ---

def evaluate(model, val_loader, device, num_tasks, image_size):
    """
    Evaluate the Stage 2 model on the validation set.
    Calculates per-task accuracy and average BPP.
    """
    model.eval()
    
    total_bpp = 0.0
    correct_per_task = [0] * num_tasks
    total_per_task = [0] * num_tasks
    
    pbar = tqdm(val_loader, desc='Evaluating', unit='batch', leave=False)
    
    with torch.no_grad():
        for i, (images, targets_dict) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            y_downstream = [t.to(device, non_blocking=True) for t in targets_dict['Y_downstream']]
            
            # Use autocast for consistency with AMP-trained models
            with torch.autocast(device_type=device.split(':')[0], enabled=(device.startswith('cuda'))):
                model_output = model(images)
            
            # 1. Calculate BPP
            z_film_likelihoods = model_output['z_film_likelihoods']['z_film']
            num_pixels = image_size * image_size
            # The rate is sum(-log2(P(z_film_hat))) over the batch
            # BPP = rate / (batch_size * num_pixels_in_image)
            bpp = torch.sum(-torch.log2(z_film_likelihoods)) / (images.size(0) * num_pixels)
            total_bpp += bpp.item()
            
            # 2. Calculate accuracy
            downstream_outputs = model_output['downstream_outputs']
            
            for k in range(num_tasks):
                task_k_outputs = downstream_outputs[k]
                task_k_targets = y_downstream[k]
                
                # Mask for samples active for this task (ignore_index is -100)
                active_mask = (task_k_targets != -100)
                
                if active_mask.sum() > 0:
                    active_outputs = task_k_outputs[active_mask]
                    active_targets = task_k_targets[active_mask]
                    
                    _, predicted = torch.max(active_outputs, 1)
                    correct_per_task[k] += (predicted == active_targets).sum().item()
                    total_per_task[k] += active_targets.size(0)

            pbar.set_postfix({'Avg BPP': f'{total_bpp / (i + 1):.4f}'})

    avg_bpp = total_bpp / len(val_loader) if len(val_loader) > 0 else 0
    
    accuracies = []
    for k in range(num_tasks):
        if total_per_task[k] > 0:
            acc = 100 * correct_per_task[k] / total_per_task[k]
            accuracies.append(acc)
        else:
            accuracies.append(float('nan')) # No samples for this task
            
    return avg_bpp, accuracies


# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(description='MANTiS Stage 2 Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file used for training.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the trained Stage 2 model checkpoint. If not provided, uses best_model_stage2.pth from the config save_dir.')
    parser.add_argument('--data_dir', type=str, default=None, help='Override data directory from config.')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size from config.')
    cli_args = parser.parse_args()

    with open(cli_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = json.loads(json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d))

    # Override config with CLI args if provided
    if cli_args.data_dir:
        config.data.data_dir = cli_args.data_dir
    if cli_args.batch_size:
        config.training.batch_size = cli_args.batch_size

    # Determine checkpoint path
    checkpoint_path = cli_args.checkpoint
    if checkpoint_path is None:
        save_dir = os.path.expanduser(config.project.save_dir)
        checkpoint_path = os.path.join(save_dir, 'best_model_stage2.pth')
        print(f"--> --checkpoint not provided, defaulting to: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Data Loading ---
    task_definitions = create_imagenet_task_definitions()
    num_tasks = len(task_definitions)
    task_names = list(task_definitions.keys())
    target_processor_func = create_task_processor_fn(task_definitions)
    
    print("Loading validation data...")
    val_wds_dataset_obj = ImageNetWebDataset(
        data_dir=config.data.data_dir,
        split='val',
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        image_size=config.data.image_size,
        target_processor=target_processor_func,
        resampled=False
    )
    _val_wds_dataset = val_wds_dataset_obj.create_dataset()
    val_loader = DataLoader(
        _val_wds_dataset, batch_size=config.training.batch_size, num_workers=config.data.num_workers,
        persistent_workers=True if config.data.num_workers > 0 else False,
        collate_fn=mantis_collate_fn
    )

    # --- Model Creation ---
    print("Setting up Stage 2 model from config...")
    model_cfg = config.model.config
    
    client_params = {
        'stem_params': {'input_channels': 3, 'output_channels': model_cfg.stem_channels},
        'task_detector_params': {'input_channels': model_cfg.stem_channels, 'num_tasks': num_tasks},
        'film_gen_params': {'num_tasks': num_tasks, 'film_channels': model_cfg.encoder_output_channels, 'hidden_dim': model_cfg.film_gen_hidden_dim},
        'filmed_encoder_params': {'input_channels': model_cfg.stem_channels, 'latent_channels': model_cfg.encoder_output_channels}
    }
    decoder_params_list = [{'input_channels': model_cfg.vib_channels, 'output_channels': model_cfg.decoder_output_channels} for _ in range(num_tasks)]
    tail_params_list = [{'input_channels': model_cfg.decoder_output_channels, 'num_classes': len(task_definitions[task_name])} for task_name in task_names]

    model = MantisStage2(
        client_params=client_params,
        num_tasks=num_tasks,
        decoder_params_list=decoder_params_list,
        tail_params_list=tail_params_list,
        vib_channels=model_cfg.vib_channels
    )

    # --- Load Checkpoint ---
    print(f"Loading Stage 2 checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint['model_state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    print("✓ Model loaded successfully.")

    # --- Run Evaluation ---
    avg_bpp, accuracies = evaluate(model, val_loader, device, num_tasks, config.data.image_size)

    # --- Print Results ---
    print("\n--- MANTiS Stage 2 Evaluation Results ---")
    print(f"  > Checkpoint: {checkpoint_path}")
    print(f"  > Average BPP on validation set: {avg_bpp:.4f}")
    print("\n  Per-Task Top-1 Accuracy:")
    
    valid_accuracies = []
    for i, acc in enumerate(accuracies):
        task_name = task_names[i]
        if not np.isnan(acc):
            print(f"    - Task '{task_name}': {acc:.2f}%")
            valid_accuracies.append(acc)
        else:
            print(f"    - Task '{task_name}': N/A (no validation samples)")
    
    if valid_accuracies:
        macro_avg_acc = np.mean(valid_accuracies)
        print(f"\n  > Macro-Average Accuracy (over {len(valid_accuracies)} tasks): {macro_avg_acc:.2f}%")
    
    print("\nEvaluation complete.")


if __name__ == '__main__':
    main() 