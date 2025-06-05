#!/usr/bin/env python3
"""
Test script to verify WebDataset epoch termination fix.
This tests that epochs end properly when using resampled WebDatasets.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from webdataset_wrapper import ImageNetWebDataset

def test_epoch_termination(data_dir, max_batches_to_test=100):
    """Test that resampled WebDataset epochs terminate correctly."""
    print("=== Testing WebDataset Epoch Termination ===")
    
    # Create a small dataset for testing
    dataset = ImageNetWebDataset(
        data_dir=data_dir,
        split='train',
        batch_size=8,
        num_workers=0,  # Use 0 workers for simpler testing
        shuffle_buffer=100,
        resampled=True  # This should terminate after length samples
    )
    
    # Override length to a small number for testing
    dataset.length = 50  # Test with just 50 samples
    
    print(f"Dataset length set to: {dataset.length}")
    print(f"Expected batches: {dataset.length // dataset.batch_size}")
    
    dataloader = dataset.create_dataloader()
    print(f"DataLoader length: {len(dataloader)}")
    
    # Test epoch termination
    for epoch in range(2):  # Test 2 epochs
        print(f"\n--- Epoch {epoch + 1} ---")
        batch_count = 0
        sample_count = 0
        
        try:
            for batch_idx, (images, labels) in enumerate(dataloader):
                batch_count += 1
                sample_count += len(images)
                
                if batch_count % 5 == 0:
                    print(f"  Batch {batch_count}: {len(images)} samples (total: {sample_count})")
                
                # Safety check to prevent infinite loops during testing
                if batch_count > max_batches_to_test:
                    print(f"  ERROR: Exceeded max batches ({max_batches_to_test}), epoch did not terminate!")
                    return False
                    
        except Exception as e:
            print(f"  ERROR during iteration: {e}")
            return False
            
        print(f"  Epoch {epoch + 1} completed with {batch_count} batches, {sample_count} samples")
        
        # Check if epoch terminated at expected point
        expected_samples = dataset.length
        if abs(sample_count - expected_samples) > dataset.batch_size:
            print(f"  WARNING: Sample count ({sample_count}) differs significantly from expected ({expected_samples})")
    
    print("\n✓ Epoch termination test passed!")
    return True

if __name__ == '__main__':
    # You would replace this with your actual ImageNet WebDataset path
    data_dir = "~/imagenet-1k-wds"  # Adjust this path
    
    if not os.path.exists(os.path.expanduser(data_dir)):
        print(f"Data directory {data_dir} not found. Please adjust the path.")
        print("This test requires actual ImageNet WebDataset files to run.")
        sys.exit(1)
    
    success = test_epoch_termination(data_dir)
    if not success:
        sys.exit(1)
    print("All tests passed!") 