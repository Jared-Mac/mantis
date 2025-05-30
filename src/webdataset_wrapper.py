"""
WebDataset wrapper for ImageNet data.

Provides a simple interface for loading ImageNet webdataset format data.
"""

import glob
import webdataset as wds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class ImageNetWebDataset:
    """WebDataset wrapper for ImageNet training data."""
    
    def __init__(self, 
                 data_dir, 
                 split='train', 
                 batch_size=64, 
                 num_workers=4, 
                 image_size=224,
                 shuffle_buffer=1000):
        """
        Initialize ImageNet WebDataset.
        
        Args:
            data_dir: Directory containing .tar files
            split: 'train' or 'val'
            batch_size: Batch size for training
            num_workers: Number of worker processes
            image_size: Input image size
            shuffle_buffer: Buffer size for shuffling (only for training)
        """
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        
        # Get tar file URLs
        if split == 'train':
            pattern = f"{data_dir}/imagenet1k-train-*.tar"
            self.length = 1281167  # ImageNet train set size
        else:
            pattern = f"{data_dir}/imagenet1k-validation-*.tar"
            self.length = 50000    # ImageNet val set size
            
        self.urls = sorted(glob.glob(pattern))
        print(f"Found {len(self.urls)} {split} shards")
        
        # Setup transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def create_dataset(self):
        """Create the webdataset."""
        dataset = wds.WebDataset(self.urls, shardshuffle=True, nodesplitter=wds.split_by_node)
        
        if self.split == 'train':
            dataset = dataset.shuffle(self.shuffle_buffer)
            
        dataset = (dataset
                  .decode("pil")
                  .to_tuple("jpg;png", "cls")
                  .map_tuple(self.transform, lambda x: int(x))
                  .with_length(self.length))
        
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
            prefetch_factor=4
        )


def create_imagenet_webdataset_loaders(data_dir, batch_size=64, num_workers=4):
    """
    Create train and validation dataloaders for ImageNet webdataset.
    
    Args:
        data_dir: Directory containing ImageNet webdataset .tar files
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        train_loader, val_loader
    """
    train_dataset = ImageNetWebDataset(
        data_dir=data_dir,
        split='train',
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    val_dataset = ImageNetWebDataset(
        data_dir=data_dir,
        split='val',
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_buffer=0  # No shuffling for validation
    )
    
    train_loader = train_dataset.create_dataloader()
    val_loader = val_dataset.create_dataloader()
    
    return train_loader, val_loader 