import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import os
from collections import Counter

def create_tiny_cifar100(num_samples_per_chunk=16, save_dir="testing"):
    """
    Creates a tiny subset of CIFAR-100 for testing.

    Args:
        num_samples_per_chunk (int): Number of samples to select from each chunk 
                                     (0-49, 50-99) for both train and val.
        save_dir (str): Directory to save the tiny dataset files.
    """
    print(f"Creating tiny CIFAR-100 dataset in '{save_dir}'...")
    os.makedirs(save_dir, exist_ok=True)

    # Define chunks
    chunk_ranges = {
        0: (0, 49),
        1: (50, 99)
    }
    
    # --- Training Dataset ---
    print("Processing training data...")
    cifar100_train_full = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=None # Save raw data
    )
    
    train_indices_chunk0 = []
    train_indices_chunk1 = []
    
    # Ensure we get diverse labels if possible, not just the first N
    # For simplicity here, we'll iterate and pick until we have enough.
    # A more robust way might be to group all indices by label first.
    
    labels_train = [cifar100_train_full.targets[i] for i in range(len(cifar100_train_full))]

    for i, label in enumerate(labels_train):
        if chunk_ranges[0][0] <= label <= chunk_ranges[0][1] and len(train_indices_chunk0) < num_samples_per_chunk:
            train_indices_chunk0.append(i)
        elif chunk_ranges[1][0] <= label <= chunk_ranges[1][1] and len(train_indices_chunk1) < num_samples_per_chunk:
            train_indices_chunk1.append(i)
        if len(train_indices_chunk0) == num_samples_per_chunk and len(train_indices_chunk1) == num_samples_per_chunk:
            break
            
    if len(train_indices_chunk0) < num_samples_per_chunk or len(train_indices_chunk1) < num_samples_per_chunk:
        print(f"Warning: Could not find enough diverse samples for training set. "
              f"Chunk0 got {len(train_indices_chunk0)}, Chunk1 got {len(train_indices_chunk1)}.")

    train_subset_indices = train_indices_chunk0 + train_indices_chunk1
    
    # Extract (image, label) tuples for saving, not the Subset object directly
    # This makes the saved data independent of the original torchvision.datasets.CIFAR100 structure.
    tiny_train_data = []
    for idx in train_subset_indices:
        img, label = cifar100_train_full[idx] # img is PIL Image
        tiny_train_data.append((img, label))

    train_save_path = os.path.join(save_dir, "tiny_cifar100_train.pt")
    torch.save(tiny_train_data, train_save_path)
    print(f"Saved {len(tiny_train_data)} training samples to {train_save_path}")
    train_labels_saved = [s[1] for s in tiny_train_data]
    print(f"Training labels distribution: {Counter(train_labels_saved)}")


    # --- Validation Dataset ---
    print("\nProcessing validation data...")
    cifar100_val_full = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=None # Save raw data
    )
    
    val_indices_chunk0 = []
    val_indices_chunk1 = []
    
    labels_val = [cifar100_val_full.targets[i] for i in range(len(cifar100_val_full))]

    for i, label in enumerate(labels_val):
        if chunk_ranges[0][0] <= label <= chunk_ranges[0][1] and len(val_indices_chunk0) < num_samples_per_chunk:
            val_indices_chunk0.append(i)
        elif chunk_ranges[1][0] <= label <= chunk_ranges[1][1] and len(val_indices_chunk1) < num_samples_per_chunk:
            val_indices_chunk1.append(i)
        if len(val_indices_chunk0) == num_samples_per_chunk and len(val_indices_chunk1) == num_samples_per_chunk:
            break

    if len(val_indices_chunk0) < num_samples_per_chunk or len(val_indices_chunk1) < num_samples_per_chunk:
        print(f"Warning: Could not find enough diverse samples for validation set. "
              f"Chunk0 got {len(val_indices_chunk0)}, Chunk1 got {len(val_indices_chunk1)}.")

    val_subset_indices = val_indices_chunk0 + val_indices_chunk1
    
    tiny_val_data = []
    for idx in val_subset_indices:
        img, label = cifar100_val_full[idx] # img is PIL Image
        tiny_val_data.append((img, label))
        
    val_save_path = os.path.join(save_dir, "tiny_cifar100_val.pt")
    torch.save(tiny_val_data, val_save_path)
    print(f"Saved {len(tiny_val_data)} validation samples to {val_save_path}")
    val_labels_saved = [s[1] for s in tiny_val_data]
    print(f"Validation labels distribution: {Counter(val_labels_saved)}")

    print("\nTiny CIFAR-100 dataset creation complete.")

if __name__ == "__main__":
    create_tiny_cifar100()
