# File: test_loader_epoch.py
import sys
import os
from pathlib import Path
import argparse
import torch # For DataLoader, etc.

# Add src directory to Python path - adjust if your script is elsewhere
script_dir = Path(__file__).resolve().parent
src_path = script_dir.parent / 'src'
sys.path.insert(0, str(src_path))

# Import your webdataset wrapper
try:
    from webdataset_wrapper import create_imagenet_webdataset_loaders
except ImportError as e:
    print(f"Error importing webdataset_wrapper: {e}")
    print(f"Ensure src_path is correct: {src_path}")
    sys.exit(1)

def get_test_argparser():
    parser = argparse.ArgumentParser(description='Test DataLoader Epoch Length')
    parser.add_argument('--data_dir', type=str, default='~/imagenet-1k-wds',
                        help='Directory containing ImageNet webdataset .tar files.')
    parser.add_argument('--batch_size', type=int, default=128, # Set to your problematic batch size
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8, # Set to your problematic num_workers
                        help='Number of data loading workers.')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Prefetch factor for DataLoader.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size.')
    # Add a new argument to control how many extra batches to test for overrun
    parser.add_argument('--overrun_check_batches', type=int, default=300,
                        help='How many batches beyond the expected epoch length to iterate to check for overrun.')
    parser.add_argument('--test_num_workers_zero', action='store_true',
                        help='Specifically test with num_workers=0.')
    return parser

def test_dataloader_epoch(args):
    args.data_dir = os.path.expanduser(args.data_dir)
    
    current_num_workers = 0 if args.test_num_workers_zero else args.num_workers

    print("--- DataLoader Test Configuration ---")
    print(f"Data Directory: {args.data_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Workers: {current_num_workers}")
    print(f"Image Size: {args.image_size}")
    print(f"Overrun Check Margin: {args.overrun_check_batches} batches")
    print("-------------------------------------\n")

    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        return

    print("Initializing DataLoader...")
    try:
        train_loader, _ = create_imagenet_webdataset_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=current_num_workers,
            prefetch_factor=args.prefetch_factor if current_num_workers > 0 else None,
            image_size=args.image_size
        )
    except Exception as e:
        print(f"ERROR: Failed to create DataLoader: {e}")
        return

    expected_batches = len(train_loader)
    print(f"DataLoader initialized. Expected batches per epoch (len(train_loader)): {expected_batches}")

    if expected_batches == 0:
        print("ERROR: DataLoader reports 0 expected batches. Check dataset path and content.")
        return

    # Determine how many batches to iterate in total for this test
    # We want to see if it goes beyond 'expected_batches'
    max_batches_to_iterate = expected_batches + args.overrun_check_batches
    
    print(f"Starting iteration test: Will iterate up to a maximum of {max_batches_to_iterate} batches.")
    print(f"If the DataLoader is correct, it should stop after {expected_batches} batches.")

    actual_batches_yielded = 0
    first_batch_shape = None
    try:
        for batch_idx, batch_data in enumerate(train_loader):
            actual_batches_yielded += 1
            
            if first_batch_shape is None and isinstance(batch_data, (list, tuple)) and len(batch_data) > 0:
                if hasattr(batch_data[0], 'shape'):
                    first_batch_shape = batch_data[0].shape

            # Optional: Print progress less frequently to avoid spamming console
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == expected_batches:
                print(f"  Processed batch {batch_idx + 1}...")
                if (batch_idx + 1) == expected_batches:
                    print(f"  >>> Reached expected end of epoch ({expected_batches} batches). Continuing to check for overrun...")

            if actual_batches_yielded > max_batches_to_iterate:
                print(f"  WARNING: Reached test iteration limit of {max_batches_to_iterate} batches. Stopping test here.")
                break
        
        # This print occurs if the loop finishes naturally (DataLoader exhausted)
        print(f"\nDataLoader iteration finished naturally after {actual_batches_yielded} batches.")

    except Exception as e:
        print(f"\nERROR during DataLoader iteration at batch {actual_batches_yielded}: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Test Summary ---")
    if first_batch_shape:
        print(f"Shape of data from first batch (e.g., images): {first_batch_shape}")
    print(f"Expected batches per epoch (len(train_loader)): {expected_batches}")
    print(f"Actual batches yielded during this test run: {actual_batches_yielded}")

    if actual_batches_yielded == expected_batches:
        print("RESULT: SUCCESS! DataLoader yielded the expected number of batches and stopped correctly.")
    elif actual_batches_yielded < expected_batches and actual_batches_yielded < max_batches_to_iterate :
         print("RESULT: FAILURE! DataLoader stopped PREMATURELY before yielding all expected batches.")
    elif actual_batches_yielded > expected_batches:
        overrun_amount = actual_batches_yielded - expected_batches
        print(f"RESULT: FAILURE! DataLoader OVERRAN by {overrun_amount} batches.")
        if actual_batches_yielded >= max_batches_to_iterate:
             print(f"   (Note: Test was capped at {max_batches_to_iterate} total iterations. Overrun might be larger.)")
    else: # actual_batches_yielded == max_batches_to_iterate and not clearly overrunning
        print("RESULT: INCONCLUSIVE. Test reached iteration limit, but did not clearly overrun or stop at expected length.")
        print("          Consider increasing --overrun_check_batches if you still suspect an issue.")

if __name__ == '__main__':
    parser = get_test_argparser()
    args = parser.parse_args()
    test_dataloader_epoch(args)