import torch
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import json
from PIL import Image
import numpy as np


class ImageNetSubgroupsDataset(ImageFolder):
    """
    ImageNet dataset organized into task subgroups.
    
    Each image can belong to multiple tasks based on semantic groupings.
    """
    
    def __init__(self, root, task_definitions, transform=None, target_transform=None):
        """
        Initialize ImageNet subgroups dataset.
        
        Args:
            root: Root directory of ImageNet dataset
            task_definitions: Dictionary mapping task names to class indices
                Example: {
                    'Animals': [0, 1, 2, ...],  # ImageNet class indices for animals
                    'Vehicles': [10, 11, 12, ...],
                    'Food': [20, 21, 22, ...]
                }
            transform: Image transformations
            target_transform: Target transformations
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self.task_definitions = task_definitions
        self.task_names = list(task_definitions.keys())
        self.num_tasks = len(self.task_names)
        
        # Create mappings between original class indices and task assignments
        self._create_task_mappings()
        
    def _create_task_mappings(self):
        """Create mappings for multi-task labels."""
        # Map original class index to list of task indices it belongs to
        self.class_to_tasks = {}
        
        for task_idx, (task_name, class_indices) in enumerate(self.task_definitions.items()):
            for class_idx in class_indices:
                if class_idx not in self.class_to_tasks:
                    self.class_to_tasks[class_idx] = []
                self.class_to_tasks[class_idx].append(task_idx)
                
        # Create task-specific class mappings (for downstream classification)
        self.task_class_mappings = {}
        for task_idx, (task_name, class_indices) in enumerate(self.task_definitions.items()):
            # Map original class index to task-specific class index
            self.task_class_mappings[task_idx] = {
                orig_idx: new_idx for new_idx, orig_idx in enumerate(class_indices)
            }
    
    def __getitem__(self, index):
        """
        Get item with multi-task labels.
        
        Returns:
            image: PIL Image or transformed image
            targets: Dictionary containing:
                - Y_task: Multi-hot vector indicating which tasks are active
                - Y_downstream: List of task-specific class labels
                - original_class: Original ImageNet class index
        """
        image, original_class = super().__getitem__(index)
        
        # Create multi-task target
        y_task = torch.zeros(self.num_tasks, dtype=torch.float32)
        y_downstream = [None] * self.num_tasks
        
        # Check which tasks this class belongs to
        if original_class in self.class_to_tasks:
            active_tasks = self.class_to_tasks[original_class]
            
            for task_idx in active_tasks:
                y_task[task_idx] = 1.0
                # Map to task-specific class index
                if task_idx in self.task_class_mappings:
                    task_specific_class = self.task_class_mappings[task_idx].get(original_class, 0)
                    y_downstream[task_idx] = task_specific_class
        
        targets = {
            'Y_task': y_task,
            'Y_downstream': y_downstream,
            'original_class': original_class
        }
        
        return image, targets


class MultiDatasetWrapper(data.Dataset):
    """
    Wrapper for combining multiple datasets with different task structures.
    
    Useful for combining datasets like ImageNet, CIFAR-100, Food-101, etc.
    """
    
    def __init__(self, datasets, task_mappings):
        """
        Initialize multi-dataset wrapper.
        
        Args:
            datasets: List of dataset objects
            task_mappings: List of dictionaries mapping dataset classes to global tasks
        """
        self.datasets = datasets
        self.task_mappings = task_mappings
        
        # Calculate dataset sizes and cumulative indices
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        self.cumulative_sizes = np.cumsum([0] + self.dataset_sizes)
        self.total_size = sum(self.dataset_sizes)
        
        # Determine total number of tasks
        all_tasks = set()
        for mapping in task_mappings:
            for task_list in mapping.values():
                all_tasks.update(task_list)
        self.num_tasks = len(all_tasks)
        
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        """Get item from appropriate dataset with unified task format."""
        # Find which dataset this index belongs to
        dataset_idx = 0
        while dataset_idx < len(self.datasets) - 1:
            if index < self.cumulative_sizes[dataset_idx + 1]:
                break
            dataset_idx += 1
        
        # Get local index within the dataset
        local_index = index - self.cumulative_sizes[dataset_idx]
        
        # Get item from dataset
        image, target = self.datasets[dataset_idx][local_index]
        
        # Convert to unified task format
        if isinstance(target, dict):
            # Already in multi-task format
            return image, target
        else:
            # Convert single-class target to multi-task format
            y_task = torch.zeros(self.num_tasks, dtype=torch.float32)
            y_downstream = [None] * self.num_tasks
            
            task_mapping = self.task_mappings[dataset_idx]
            if target in task_mapping:
                active_tasks = task_mapping[target]
                for task_idx in active_tasks:
                    y_task[task_idx] = 1.0
                    y_downstream[task_idx] = target  # Use original class as task-specific class
            
            targets = {
                'Y_task': y_task,
                'Y_downstream': y_downstream,
                'original_class': target
            }
            
            return image, targets


def create_imagenet_task_definitions():
    """
    Create example task definitions for ImageNet subgroups.
    
    Returns:
        Dictionary mapping task names to ImageNet class indices
    """
    # Example task definitions (simplified)
    # In practice, these would be more comprehensive and based on WordNet hierarchy
    task_definitions = {
        'Animals': list(range(0, 398)),  # ImageNet classes 0-397 are mostly animals
        'Vehicles': list(range(398, 450)),  # Classes for various vehicles
        'Food': list(range(450, 500)),  # Food-related classes
        'Plants': list(range(500, 600)),  # Plant-related classes
        'Objects': list(range(600, 1000))  # Other objects
    }
    
    return task_definitions


def get_imagenet_transforms(is_training=True, image_size=224):
    """
    Get standard ImageNet transformations.
    
    Args:
        is_training: Whether for training (applies augmentation)
        image_size: Target image size
        
    Returns:
        Compose transform
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])


def create_mantis_datasets(data_root, image_size=224, use_multidataset=False):
    """
    Create datasets for MANTiS training.
    
    Args:
        data_root: Root directory containing datasets
        image_size: Target image size
        use_multidataset: Whether to use multiple datasets or just ImageNet
        
    Returns:
        Dictionary containing train and validation datasets
    """
    if use_multidataset:
        # Implementation for combining multiple datasets
        # This would require additional dataset implementations
        raise NotImplementedError("Multi-dataset support not yet implemented")
    else:
        # Single ImageNet with task subgroups
        task_definitions = create_imagenet_task_definitions()
        
        train_transform = get_imagenet_transforms(is_training=True, image_size=image_size)
        val_transform = get_imagenet_transforms(is_training=False, image_size=image_size)
        
        train_dataset = ImageNetSubgroupsDataset(
            root=os.path.join(data_root, 'imagenet/train'),
            task_definitions=task_definitions,
            transform=train_transform
        )
        
        val_dataset = ImageNetSubgroupsDataset(
            root=os.path.join(data_root, 'imagenet/val'),
            task_definitions=task_definitions,
            transform=val_transform
        )
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'num_tasks': len(task_definitions),
            'task_names': list(task_definitions.keys())
        } 