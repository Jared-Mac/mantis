# testing.py (Continuing from your previous successful run)
import torch
from torch.utils.data import DataLoader # Add DataLoader import
from misc.datasets import get_dataset 
from torchdistill.common import yaml_util
import os

config_path = "config/mantis/imagenet/imagenet-resnet18.yaml" # 
config = yaml_util.load_yaml_file(os.path.expanduser(config_path))
datasets_config = config['datasets']

# Get train dataset config for your label-chunked imagenet
train_dataset_config_name = 'imagenet_label_chunked_for_tasks' # Key from your YAML
train_split_config = datasets_config[train_dataset_config_name]['splits']['train']
val_split_config = datasets_config[train_dataset_config_name]['splits']['val']

# Construct full config for get_dataset for train split
full_train_dataset_conf = {
    'type': datasets_config[train_dataset_config_name]['type'],
    **train_split_config 
}
print("Attempting to load training dataset...")
train_dataset = get_dataset(full_train_dataset_conf)
print(f"Successfully loaded training dataset: {train_dataset_config_name}")
print(f"Length of training dataset: {len(train_dataset)}")

# Construct full config for get_dataset for val split
full_val_dataset_conf = {
    'type': datasets_config[train_dataset_config_name]['type'],
    **val_split_config
}
print("Attempting to load validation dataset...")
val_dataset = get_dataset(full_val_dataset_conf)
print(f"Successfully loaded validation dataset: {train_dataset_config_name}")
print(f"Length of validation dataset: {len(val_dataset)}")

# --- NEW: Step 1 - Inspect __getitem__ output ---
if train_dataset and len(train_dataset) > 0:
    print("\n--- Inspecting train_dataset[0] ---")
    try:
        # Fetch the first sample
        image, targets_tuple, original_label, assigned_task_id = train_dataset[0]
        main_target, task_detector_target = targets_tuple

        print(f"Image shape: {image.shape}, type: {image.dtype}")
        print(f"Original Label: {original_label}")
        print(f"Assigned Task ID (user-defined): {assigned_task_id}") # This is the task_id from your YAML task_configs
        
        print(f"Targets Tuple:")
        print(f"  Main Target (task-specific label): {main_target}, shape: {main_target.shape}, type: {main_target.dtype}")
        print(f"  Task Detector Target (one-hot): {task_detector_target}, shape: {task_detector_target.shape}, type: {task_detector_target.dtype}")

        # You can add more assertions here based on your first sample's expected label and task
        # For example, if original_label 0 should map to task_id 0 (dense_idx 0) and task_specific_label 0:
        # if original_label == 0: 
        #     assert assigned_task_id == 0 # Or whatever task_id you expect for original_label 0
        #     assert main_target.item() == 0 # Assuming remapping starts from 0
        #     assert task_detector_target[train_dataset.task_id_to_dense_idx_map[0]].item() == 1.0 
        #     assert task_detector_target.sum().item() == 1.0 # Should be one-hot
            
        print(f"Number of distinct task chunks defined for detector: {train_dataset.num_defined_tasks}")
        assert task_detector_target.shape[0] == train_dataset.num_defined_tasks, \
            f"Task detector target dim ({task_detector_target.shape[0]}) != num_defined_tasks ({train_dataset.num_defined_tasks})"

    except Exception as e:
        print(f"Error inspecting dataset[0]: {e}")
        import traceback
        traceback.print_exc()

# --- NEW: Step 2 - Test DataLoader Iteration ---
if train_dataset and len(train_dataset) > 0:
    print("\n--- Testing DataLoader ---")
    try:
        # Use a small batch size for testing
        # Important: Ensure your dataset's original_dataset (ImageFolder) is actually loading images
        # If it's a dummy ImageFolder with no actual images, DataLoader might error or be empty.
        # For this test, if your dummy dataset has 2 items, batch_size=2 is fine.
        batch_size_test = min(2, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size_test, shuffle=True)
        
        for i, batch in enumerate(train_loader):
            images, targets_batch_tuple, original_labels_batch, assigned_task_ids_batch = batch
            main_targets_batch, task_detector_targets_batch = targets_batch_tuple
            
            print(f"\n--- Batch {i} from DataLoader ---")
            print(f"Images shape: {images.shape}")
            print(f"Main Targets shape: {main_targets_batch.shape}, Dtype: {main_targets_batch.dtype}")
            print(f"Task Detector Targets shape: {task_detector_targets_batch.shape}, Dtype: {task_detector_targets_batch.dtype}")
            print(f"Original Labels in batch: {original_labels_batch}")
            print(f"Assigned Task IDs in batch: {assigned_task_ids_batch}")
            
            if i >= 0: # Just check the first batch for this test
                break
        print("\nDataLoader iteration test successful (checked one batch)!")
    except Exception as e:
        print(f"Error during DataLoader iteration: {e}")
        import traceback
        traceback.print_exc()