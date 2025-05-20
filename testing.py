# testing.py (Adapted for CIFAR-100)
import torch
from torch.utils.data import DataLoader
# Adjust the import path if your get_dataset is in model.datasets.registry
# from model.datasets import get_dataset 
from misc.datasets import get_dataset # Assuming it's here from your previous structure
from torchdistill.common import yaml_util
import os
import torchvision # For type checking ImageFolder/CIFAR100 if needed in debug
from torchdistill.models.registry import get_model # For instantiating the model via factory
from model.network import FiLMedHFactorizedPriorCompressionModule # Import your main network class for type checking
from misc.util import load_model # Your utility for loading models
# 1. UPDATE THIS PATH to your new CIFAR-100 YAML configuration file
config_path = "config/mantis/cifar-100/resnet18.yaml" 
# This should be the "Fully Specified YAML for CIFAR-100 Test" we worked on.

config = yaml_util.load_yaml_file(os.path.expanduser(config_path))
datasets_config = config['datasets']

# 2. UPDATE THIS to the key used in your CIFAR-100 YAML for the LabelChunkedTaskDataset
#    In the YAML I provided, this was 'cifar100_5tasks_chunked'.
dataset_config_key_name = 'cifar100_5tasks_chunked' 

if dataset_config_key_name not in datasets_config:
    raise ValueError(f"Dataset configuration key '{dataset_config_key_name}' not found in YAML. "
                     f"Available keys: {list(datasets_config.keys())}")

train_split_config = datasets_config[dataset_config_key_name]['splits']['train']
val_split_config = datasets_config[dataset_config_key_name]['splits']['val']

# Construct full config for get_dataset for train split
full_train_dataset_conf = {
    'type': datasets_config[dataset_config_key_name]['type'], # Should be 'LabelChunkedTaskDataset'
    **train_split_config 
    # train_split_config already contains 'params' for LabelChunkedTaskDataset
    # and within that, 'params.original_dataset' has the config for CIFAR100
}
print("Attempting to load training dataset...")
train_dataset = get_dataset(full_train_dataset_conf)
print(f"Successfully loaded training dataset: {train_dataset.dataset_id if hasattr(train_dataset, 'dataset_id') else dataset_config_key_name + '/train'}") # Use dataset_id if available
print(f"Length of training dataset: {len(train_dataset)}")

# Construct full config for get_dataset for val split
full_val_dataset_conf = {
    'type': datasets_config[dataset_config_key_name]['type'], # Should be 'LabelChunkedTaskDataset'
    **val_split_config
}
print("Attempting to load validation dataset...")
val_dataset = get_dataset(full_val_dataset_conf)
print(f"Successfully loaded validation dataset: {val_dataset.dataset_id if hasattr(val_dataset, 'dataset_id') else dataset_config_key_name + '/val'}")
print(f"Length of validation dataset: {len(val_dataset)}")

# --- Step 1 - Inspect __getitem__ output ---
if train_dataset and len(train_dataset) > 0:
    print("\n--- Inspecting train_dataset[0] ---")
    try:
        # Check type of original_dataset within the wrapper for debugging
        if hasattr(train_dataset, 'original_dataset'):
            print(f"Type of wrapped original_dataset: {type(train_dataset.original_dataset)}")
            if isinstance(train_dataset.original_dataset, torchvision.datasets.CIFAR100):
                 print(f"  Original dataset is CIFAR100. Found {len(train_dataset.original_dataset.data)} samples.")
            else:
                print(f"  Warning: Original dataset might not be a CIFAR100 instance as expected by this test script's specific debug prints.")
        
        # Fetch the first sample
        image, targets_tuple, original_label, assigned_task_id = train_dataset[0]
        main_target, task_detector_target = targets_tuple

        print(f"Image shape: {image.shape}, type: {image.dtype}")
        print(f"Original Label (from CIFAR-100, 0-99): {original_label}")
        print(f"Assigned Task ID (from YAML task_configs, e.g., 0-4): {assigned_task_id}")
        
        print(f"Targets Tuple:")
        print(f"  Main Target (task-specific label, 0-19 for 20 classes/task): {main_target}, shape: {main_target.shape}, type: {main_target.dtype}")
        print(f"  Task Detector Target (one-hot): {task_detector_target}, shape: {task_detector_target.shape}, type: {task_detector_target.dtype}")
            
        num_defined_tasks = train_dataset.num_defined_tasks
        print(f"Number of distinct task chunks defined for detector: {num_defined_tasks}")
        
        assert task_detector_target.shape[0] == num_defined_tasks, \
            f"Task detector target dim ({task_detector_target.shape[0]}) != num_defined_tasks ({num_defined_tasks})"
        
        if num_defined_tasks > 0 and task_detector_target.sum().item() == 1.0:
            print(f"  Task detector target is one-hot. Active task dense index: {torch.argmax(task_detector_target).item()}")
        elif num_defined_tasks > 0:
            print(f"  Warning: Task detector target is NOT strictly one-hot (sum={task_detector_target.sum().item()}) or assigned_task_id was default.")


    except Exception as e:
        print(f"Error inspecting dataset[0]: {e}")
        import traceback
        traceback.print_exc()

# --- Step 2 - Test DataLoader Iteration ---
if train_dataset and len(train_dataset) > 0:
    print("\n--- Testing DataLoader ---")
    try:
        # For CIFAR-100, train_dataset.original_dataset.data contains the actual images (numpy arrays)
        # So it won't be empty in the same way ImageFolder would be if path is wrong.
        # Download will be triggered if 'download: True' and data not present.
        batch_size_test = min(4, len(train_dataset)) # Use a slightly larger batch for testing if possible
        if batch_size_test == 0:
             print("Train dataset is empty, skipping DataLoader test.")
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size_test, shuffle=True)
            
            batch_count = 0
            for i, batch in enumerate(train_loader):
                images, targets_batch_tuple, original_labels_batch, assigned_task_ids_batch = batch
                main_targets_batch, task_detector_targets_batch = targets_batch_tuple
                
                print(f"\n--- Batch {i} from DataLoader ---")
                print(f"Images shape: {images.shape}")
                print(f"Main Targets shape: {main_targets_batch.shape}, Dtype: {main_targets_batch.dtype}")
                print(f"Task Detector Targets shape: {task_detector_targets_batch.shape}, Dtype: {task_detector_targets_batch.dtype}")
                # print(f"Original Labels in batch: {original_labels_batch}")
                # print(f"Assigned Task IDs in batch: {assigned_task_ids_batch}")
                batch_count += 1
                if i >= 0: # Just check the first batch for this initial test
                    break
            if batch_count > 0:
                print("\nDataLoader iteration test successful (checked at least one batch)!")
            else:
                print("\nDataLoader did not yield any batches. Dataset might be effectively empty or too small for batch size.")
    except Exception as e:
        print(f"Error during DataLoader iteration: {e}")

print("\n--- Phase 2: Model Instantiation and Forward Pass ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

models_config = config['models']
student_model_config = models_config['student_model']

try:
    print("\nInstantiating student model...")
    # Use your load_model utility, which should handle the factory function
    # Set skip_ckpt=True if you don't want to load weights for this test
    # Set distributed=False as this is a simple test script
    student_model = load_model(
        model_config=student_model_config, 
        device=device, 
        distributed=False, 
        skip_ckpt=True # Don't try to load weights for this initial test
    )
    student_model.eval() # Set to eval mode for forward pass test

    # Check if the model is the correct type
    if not isinstance(student_model, FiLMedHFactorizedPriorCompressionModule):
        print(f"Warning: Loaded model is type {type(student_model)}, expected FiLMedHFactorizedPriorCompressionModule or a wrapped version.")

    print("Student model instantiated successfully.")

    # Get a batch of data
    if train_dataset and len(train_dataset) > 0:
        # Re-create loader if you want a fresh one, or use one from previous test
        data_loader_for_model_test = DataLoader(train_dataset, batch_size=4, shuffle=False) 
        image_batch, targets_batch_tuple, _, _ = next(iter(data_loader_for_model_test))
        
        image_batch = image_batch.to(device)
        # main_targets_batch, task_detector_targets_batch = targets_batch_tuple
        # main_targets_batch = main_targets_batch.to(device)
        # task_detector_targets_batch = task_detector_targets_batch.to(device)
        
        # The 'targets' argument to model.forward is optional for inference,
        # but your model's forward pass might use it to form the output dict for training.
        # Let's pass it as the model expects a 'targets' argument in its forward method.
        # We need to ensure targets_batch_tuple is also on the correct device if model uses it.
        targets_for_model = (targets_batch_tuple[0].to(device), targets_batch_tuple[1].to(device))


        print("\nPerforming a forward pass with a batch...")
        with torch.no_grad(): # No need to compute gradients for this test
             # Assuming your model.forward takes (x, targets=None)
            output_dict = student_model(image_batch, targets=targets_for_model) 

        print("\n--- Model Forward Pass Output Dictionary ---")
        if isinstance(output_dict, dict):
            for key, value in output_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"Key: '{key}', Shape: {value.shape}, Device: {value.device}, Dtype: {value.dtype}")
                else:
                    print(f"Key: '{key}', Type: {type(value)}, Value: {value}")
            
            # Specific checks for expected keys and shapes
            assert 'main_output' in output_dict, "Model output missing 'main_output'"
            assert 'conditioning_signal_preview' in output_dict, "Model output missing 'conditioning_signal_preview'"
            
            # Example shape checks (adjust batch size and dimensions as per your config)
            # Expected main_output shape: (batch_size, num_classes_per_task_chunk)
            # Expected conditioning_signal_preview shape: (batch_size, task_probability_model.output_cond_signal_dim)
            # These will be checked implicitly by the loss calculation step later.

        else:
            print(f"Model output was not a dictionary, but type: {type(output_dict)}")
            if isinstance(output_dict, torch.Tensor):
                 print(f"Output tensor shape: {output_dict.shape}")


        print("\nModel forward pass test completed.")

except Exception as e:
    print(f"Error during model instantiation or forward pass: {e}")
    import traceback
    traceback.print_exc()