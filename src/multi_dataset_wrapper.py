#!/usr/bin/env python3
"""
Multi-Dataset wrapper for MANTiS Stage 2 training with distinct datasets as tasks.

This module combines CIFAR-100, STL-10, and Flowers-102 as separate tasks:
- Task 0: CIFAR-100 (100 classes)
- Task 1: STL-10 (10 classes) 
- Task 2: Flowers-102 (102 classes)
"""

import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import os
import numpy as np
from PIL import Image


class MultiTaskDataset(data.Dataset):
    """
    Multi-task dataset that combines multiple datasets as separate tasks.
    Each sample belongs to exactly one task.
    """
    
    def __init__(self, datasets_list, task_names, image_size=224):
        """
        Initialize multi-task dataset.
        
        Args:
            datasets_list: List of (dataset, task_id) tuples
            task_names: List of task names
            image_size: Target image size
        """
        self.datasets_list = datasets_list
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.image_size = image_size
        
        # Calculate dataset sizes and cumulative indices
        self.dataset_sizes = [len(dataset) for dataset, _ in datasets_list]
        self.cumulative_sizes = np.cumsum([0] + self.dataset_sizes)
        self.total_size = sum(self.dataset_sizes)
        
        print(f"Multi-task dataset created:")
        for i, (task_name, size) in enumerate(zip(task_names, self.dataset_sizes)):
            print(f"  Task {i}: {task_name} - {size:,} samples")
        print(f"  Total: {self.total_size:,} samples")
        
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        """Get item with multi-task format."""
        # Find which dataset this index belongs to
        dataset_idx = 0
        while dataset_idx < len(self.datasets_list) - 1:
            if index < self.cumulative_sizes[dataset_idx + 1]:
                break
            dataset_idx += 1
        
        # Get local index within the dataset
        local_index = index - self.cumulative_sizes[dataset_idx]
        
        # Get item from dataset
        dataset, task_id = self.datasets_list[dataset_idx]
        image, class_label = dataset[local_index]
        
        # Create multi-task target
        y_task = torch.zeros(self.num_tasks, dtype=torch.float32)
        y_downstream = [None] * self.num_tasks
        
        # Set active task
        y_task[task_id] = 1.0
        y_downstream[task_id] = class_label
        
        targets = {
            'Y_task': y_task,
            'Y_downstream': y_downstream,
            'original_class': class_label,
            'task_id': task_id
        }
        
        return image, targets


def get_transforms(is_training=True, image_size=224):
    """Get transforms for multi-dataset training."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_multi_task_datasets(data_root, image_size=224, download=True):
    """
    Create multi-task datasets with CIFAR-100, STL-10, and Flowers-102.
    
    Args:
        data_root: Root directory for datasets
        image_size: Target image size
        download: Whether to download datasets if not present
        
    Returns:
        Dictionary with train/val datasets and task info
    """
    
    # Ensure data directory exists
    os.makedirs(data_root, exist_ok=True)
    
    # Define task names and corresponding number of classes
    task_names = ['CIFAR100', 'STL10', 'Flowers102']
    num_classes = [100, 10, 102]
    
    print(f"Creating multi-task datasets in: {data_root}")
    
    # Get transforms
    train_transform = get_transforms(is_training=True, image_size=image_size)
    val_transform = get_transforms(is_training=False, image_size=image_size)
    
    # --- Training Datasets ---
    print("Loading training datasets...")
    
    # CIFAR-100 (Task 0)
    cifar100_train = datasets.CIFAR100(
        root=os.path.join(data_root, 'cifar100'),
        train=True,
        transform=train_transform,
        download=download
    )
    
    # STL-10 (Task 1)
    stl10_train = datasets.STL10(
        root=os.path.join(data_root, 'stl10'),
        split='train',
        transform=train_transform,
        download=download
    )
    
    # Flowers-102 (Task 2)
    flowers102_train = datasets.Flowers102(
        root=os.path.join(data_root, 'flowers102'),
        split='train',
        transform=train_transform,
        download=download
    )
    
    # --- Validation Datasets ---
    print("Loading validation datasets...")
    
    # CIFAR-100 validation
    cifar100_val = datasets.CIFAR100(
        root=os.path.join(data_root, 'cifar100'),
        train=False,
        transform=val_transform,
        download=False  # Already downloaded
    )
    
    # STL-10 validation
    stl10_val = datasets.STL10(
        root=os.path.join(data_root, 'stl10'),
        split='test',  # STL-10 uses 'test' for validation
        transform=val_transform,
        download=False
    )
    
    # Flowers-102 validation
    flowers102_val = datasets.Flowers102(
        root=os.path.join(data_root, 'flowers102'),
        split='test',  # Flowers-102 uses 'test' for validation
        transform=val_transform,
        download=False
    )
    
    # Create multi-task datasets
    train_datasets_list = [
        (cifar100_train, 0),
        (stl10_train, 1),
        (flowers102_train, 2)
    ]
    
    val_datasets_list = [
        (cifar100_val, 0),
        (stl10_val, 1),
        (flowers102_val, 2)
    ]
    
    train_dataset = MultiTaskDataset(train_datasets_list, task_names, image_size)
    val_dataset = MultiTaskDataset(val_datasets_list, task_names, image_size)
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'num_tasks': len(task_names),
        'task_names': task_names,
        'num_classes': num_classes
    }


def create_task_definitions_multi_dataset():
    """
    Create task definitions for multi-dataset training.
    
    Returns:
        Dictionary mapping task names to class counts
    """
    task_definitions = {
        'CIFAR100': list(range(100)),     # CIFAR-100: 100 classes
        'STL10': list(range(10)),         # STL-10: 10 classes  
        'Flowers102': list(range(102))    # Flowers-102: 102 classes
    }
    
    return task_definitions


def create_task_processor_fn_multi_dataset():
    """
    Create a simple task processor for multi-dataset that passes through the targets.
    This is simpler than ImageNet since each dataset is already a separate task.
    """
    def process_target(target_dict):
        # Target dict already contains the multi-task format from MultiTaskDataset
        return target_dict
    
    return process_target


def multi_dataset_collate_fn(batch):
    """
    Custom collate function for multi-dataset training.
    """
    images = [item[0] for item in batch]
    targets_list = [item[1] for item in batch]
    
    # Collate images
    collated_images = torch.stack(images)
    
    # Collate targets
    collated_targets = {}
    if targets_list:
        collated_targets['Y_task'] = torch.stack([t['Y_task'] for t in targets_list])
        collated_targets['original_class'] = torch.tensor([t['original_class'] for t in targets_list])
        collated_targets['task_id'] = torch.tensor([t['task_id'] for t in targets_list])
        
        # Handle Y_downstream
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


if __name__ == "__main__":
    # Test the multi-dataset creation
    print("Testing multi-dataset creation...")
    
    data_root = "./data/multi_task"
    datasets_info = create_multi_task_datasets(data_root, download=True)
    
    print(f"\nDatasets created successfully!")
    print(f"Tasks: {datasets_info['task_names']}")
    print(f"Classes per task: {datasets_info['num_classes']}")
    print(f"Train samples: {len(datasets_info['train'])}")
    print(f"Val samples: {len(datasets_info['val'])}")
    
    # Test a few samples
    print(f"\nTesting samples:")
    for i in range(min(5, len(datasets_info['train']))):
        image, targets = datasets_info['train'][i]
        print(f"Sample {i}: Task {targets['task_id']}, Class {targets['original_class']}, Y_task: {targets['Y_task']}") 