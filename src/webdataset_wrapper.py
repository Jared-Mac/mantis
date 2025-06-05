# file_path: src/webdataset_wrapper.py
"""
WebDataset wrapper for ImageNet data.

Provides a simple interface for loading ImageNet webdataset format data.
"""

import glob
import webdataset as wds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os # For expanding user path
import torch.distributed as dist
import torch # For torch.tensor in target_processor

class ImageNetWebDataset:
    """WebDataset wrapper for ImageNet training data."""

    def __init__(self,
                 data_dir,
                 split='train',
                 batch_size=64,
                 num_workers=4,
                 image_size=224,
                 shuffle_buffer=5000,
                 prefetch_factor=2,
                 resampled=False,
                 target_processor=None): # Added target_processor
        """
        Initialize ImageNet WebDataset.
        // ...
        Args:
        // ...
            prefetch_factor: Number of batches loaded in advance by each worker.
            resampled: If True, resample shards with replacement for better shuffling across epochs (for training).
            target_processor: A function to process the raw target (class index string) into the desired format.
        """
        self.data_dir = os.path.expanduser(data_dir)
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_factor = prefetch_factor
        self.resampled = resampled if split == 'train' else False
        self.target_processor = target_processor if target_processor else lambda x: int(x) # Default to int class

        # Get tar file URLs
        if split == 'train':
            pattern = f"{self.data_dir}/imagenet1k-train-*.tar"
            self.length = 1281167  # ImageNet train set size
        elif split == 'val':
            pattern = f"{self.data_dir}/imagenet1k-validation-*.tar"
            self.length = 50000   # ImageNet val set size
        else:
            raise ValueError(f"Unknown split: {split}")

        self.urls = sorted(glob.glob(pattern))
        if not self.urls:
            raise FileNotFoundError(f"No .tar files found for split '{split}' with pattern '{pattern}' in '{self.data_dir}'")
        print(f"Found {len(self.urls)} {split} shards from {self.data_dir}")

        # Setup transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else: # Validation or Test
            self.transform = transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])


    def create_dataset(self):
        """Create the webdataset."""
        
        base_urls = self.urls
        base_shardshuffle = (self.split == 'train')
        
        dataset = wds.WebDataset(base_urls, 
                                 shardshuffle=base_shardshuffle, 
                                 nodesplitter=wds.split_by_node, 
                                 resampled=self.resampled)

        if dist.is_available() and dist.is_initialized() and self.split == 'train':
            world_size = dist.get_world_size()
            effective_length_this_rank = (self.length + world_size - 1) // world_size 
        else:
            effective_length_this_rank = self.length

        if self.split == 'train':
            if self.resampled:
                dataset = dataset.shuffle(self.shuffle_buffer)
                dataset = dataset.with_epoch(effective_length_this_rank)
            else: 
                dataset = dataset.shuffle(self.shuffle_buffer)
        
        dataset = (dataset
                   .decode("pil")
                   .to_tuple("jpg;png;jpeg", "cls") # "cls" is the raw class index string
                   .map_tuple(self.transform, self.target_processor) # Apply target_processor
                  )
        
        dataset = dataset.with_length(effective_length_this_rank)
        return dataset


    def create_dataloader(self):
        """Create a DataLoader for the dataset."""
        dataset = self.create_dataset()

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=(self.split == 'train') 
        )


def create_imagenet_webdataset_loaders(data_dir, batch_size=64, num_workers=4, 
                                       prefetch_factor=2, image_size=224, 
                                       target_processor_train=None, target_processor_val=None): # Added target_processors
    """
    Create train and validation dataloaders for ImageNet webdataset.
    Allows custom target processors for train and validation.
    """
    train_dataset = ImageNetWebDataset(
        data_dir=data_dir,
        split='train',
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_buffer=10000, 
        prefetch_factor=prefetch_factor,
        image_size=image_size,
        resampled=True,
        target_processor=target_processor_train # Use custom processor
    )

    val_dataset = ImageNetWebDataset(
        data_dir=data_dir,
        split='val',
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_buffer=0, 
        prefetch_factor=prefetch_factor,
        image_size=image_size,
        resampled=False,
        target_processor=target_processor_val # Use custom processor
    )

    train_loader = train_dataset.create_dataloader()
    val_loader = val_dataset.create_dataloader()

    return train_loader, val_loader