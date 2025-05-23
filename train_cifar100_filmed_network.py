# train_cifar100_filmed_network.py
import torch
from torch.utils.data import DataLoader
import torch.optim as optim # Added
import argparse # Added
# Adjust the import path if your get_dataset is in model.datasets.registry
# from model.datasets import get_dataset 
from misc.datasets import get_dataset # Assuming it's here from your previous structure
from misc.loss import MultiLabelTaskRelevancyBCELoss, BppLossOrig # Added
from torchdistill.common import yaml_util
import os
import torchvision # For type checking ImageFolder/CIFAR100 if needed in debug
from torchdistill.models.registry import get_model # For instantiating the model via factory
from model.network import FiLMedHFactorizedPriorCompressionModule # Import your main network class for type checking
from misc.util import load_model # Your utility for loading models


def inspect_data_and_model(config_path, device_str):
    """
    Performs the original inspection functionality:
    - Loads dataset and student model from YAML.
    - Prints sample info.
    - Performs a single forward pass and prints output dict.
    """
    print(f"--- Running in INSPECT mode ---")
    print(f"Using config: {config_path}")
    device = torch.device(device_str)
    print(f"Using device: {device}")

    config = yaml_util.load_yaml_file(os.path.expanduser(config_path))
    datasets_config = config['datasets']

    # In the YAML I provided, this was 'cifar100_5tasks_chunked'.
    dataset_config_key_name = 'cifar100_5tasks_chunked' 

    if dataset_config_key_name not in datasets_config:
        raise ValueError(f"Dataset configuration key '{dataset_config_key_name}' not found in YAML. "
                         f"Available keys: {list(datasets_config.keys())}")

    train_split_config = datasets_config[dataset_config_key_name]['splits']['train']
    # val_split_config = datasets_config[dataset_config_key_name]['splits']['val'] # Not used in inspect

    # Construct full config for get_dataset for train split
    full_train_dataset_conf = {
        'type': datasets_config[dataset_config_key_name]['type'], # Should be 'LabelChunkedTaskDataset'
        **train_split_config 
    }
    print("Attempting to load training dataset for inspection...")
    train_dataset = get_dataset(full_train_dataset_conf)
    print(f"Successfully loaded training dataset: {train_dataset.dataset_id if hasattr(train_dataset, 'dataset_id') else dataset_config_key_name + '/train'}")
    print(f"Length of training dataset: {len(train_dataset)}")

    # --- Step 1 - Inspect __getitem__ output ---
    if train_dataset and len(train_dataset) > 0:
        print("\n--- Inspecting train_dataset[0] ---")
        try:
            if hasattr(train_dataset, 'original_dataset'):
                print(f"Type of wrapped original_dataset: {type(train_dataset.original_dataset)}")
                if isinstance(train_dataset.original_dataset, torchvision.datasets.CIFAR100):
                     print(f"  Original dataset is CIFAR100. Found {len(train_dataset.original_dataset.data)} samples.")
                else:
                    print(f"  Warning: Original dataset might not be a CIFAR100 instance as expected by this test script's specific debug prints.")
            
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

    # --- Step 2 - Test DataLoader Iteration (simplified for inspection) ---
    if train_dataset and len(train_dataset) > 0:
        print("\n--- Testing DataLoader (first batch) ---")
        try:
            batch_size_test = min(4, len(train_dataset))
            if batch_size_test == 0:
                 print("Train dataset is empty, skipping DataLoader test.")
            else:
                train_loader = DataLoader(train_dataset, batch_size=batch_size_test, shuffle=True)
                
                batch_count = 0
                for i, batch in enumerate(train_loader):
                    images, targets_batch_tuple, _, _ = batch
                    main_targets_batch, task_detector_targets_batch = targets_batch_tuple
                    
                    print(f"\n--- Batch {i} from DataLoader ---")
                    print(f"Images shape: {images.shape}")
                    print(f"Main Targets shape: {main_targets_batch.shape}, Dtype: {main_targets_batch.dtype}")
                    print(f"Task Detector Targets shape: {task_detector_targets_batch.shape}, Dtype: {task_detector_targets_batch.dtype}")
                    batch_count += 1
                    if i >= 0: # Just check the first batch
                        break
                if batch_count > 0:
                    print("\nDataLoader iteration test successful (checked first batch)!")
                else:
                    print("\nDataLoader did not yield any batches.")
        except Exception as e:
            print(f"Error during DataLoader iteration: {e}")

    print("\n--- Phase 2: Model Instantiation and Forward Pass (Inspection) ---")
    
    models_config = config['models']
    student_model_config = models_config['student_model']

    try:
        print("\nInstantiating student model for inspection...")
        student_model = load_model(
            model_config=student_model_config, 
            device=device, 
            distributed=False, 
            skip_ckpt=True 
        )
        student_model.eval()

        if not isinstance(student_model, FiLMedHFactorizedPriorCompressionModule):
            print(f"Warning: Loaded model is type {type(student_model)}, expected FiLMedHFactorizedPriorCompressionModule or a wrapped version.")
        print("Student model instantiated successfully.")

        if train_dataset and len(train_dataset) > 0:
            data_loader_for_model_test = DataLoader(train_dataset, batch_size=min(4, len(train_dataset)), shuffle=False) 
            image_batch, targets_batch_tuple, _, _ = next(iter(data_loader_for_model_test))
            
            image_batch = image_batch.to(device)
            targets_for_model = (targets_batch_tuple[0].to(device), targets_batch_tuple[1].to(device))

            print("\nPerforming a forward pass with a batch (inspection)...")
            with torch.no_grad():
                output_dict = student_model(image_batch, targets=targets_for_model) 

            print("\n--- Model Forward Pass Output Dictionary (Inspection) ---")
            if isinstance(output_dict, dict):
                for key, value in output_dict.items():
                    if isinstance(value, torch.Tensor):
                        print(f"Key: '{key}', Shape: {value.shape}, Device: {value.device}, Dtype: {value.dtype}")
                    else:
                        print(f"Key: '{key}', Type: {type(value)}, Value: {value}")
                
                assert 'main_output' in output_dict, "Model output missing 'main_output'"
                assert 'conditioning_signal_preview' in output_dict, "Model output missing 'conditioning_signal_preview'"
            else:
                print(f"Model output was not a dictionary, but type: {type(output_dict)}")
                if isinstance(output_dict, torch.Tensor):
                     print(f"Output tensor shape: {output_dict.shape}")
            print("\nModel forward pass test completed for inspection.")

    except Exception as e:
        print(f"Error during model instantiation or forward pass in inspection: {e}")
        import traceback
        traceback.print_exc()
    print(f"--- Finished INSPECT mode ---")


def run_training(config_path, device_str, num_epochs, batches_per_epoch):
    """
    Runs the training loop for the FiLMed network.
    """
    print(f"--- Running in TRAIN mode ---")
    print(f"Config: {config_path}, Device: {device_str}, Epochs: {num_epochs}, Batches/Epoch: {batches_per_epoch if batches_per_epoch else 'Full'}")
    
    device = torch.device(device_str)
    config = yaml_util.load_yaml_file(os.path.expanduser(config_path))
    
    # --- 1. Load Datasets and DataLoaders ---
    datasets_config = config['datasets']
    dataset_config_key_name = 'cifar100_5tasks_chunked' # As used in inspect and assumed for training

    train_split_config = datasets_config[dataset_config_key_name]['splits']['train']
    full_train_dataset_conf = {
        'type': datasets_config[dataset_config_key_name]['type'],
        **train_split_config
    }
    train_dataset = get_dataset(full_train_dataset_conf)
    print(f"Loaded training dataset: {len(train_dataset)} samples.")

    # Val dataset (optional for this snippet, but good practice)
    # val_split_config = datasets_config[dataset_config_key_name]['splits']['val']
    # full_val_dataset_conf = {
    # 'type': datasets_config[dataset_config_key_name]['type'],
    # **val_split_config
    # }
    # val_dataset = get_dataset(full_val_dataset_conf)
    # print(f"Loaded validation dataset: {len(val_dataset)} samples.")

    train_batch_size = config['train'].get('batch_size', 32) # Default if not in YAML
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- 2. Load Model ---
    models_config = config['models']
    student_model_config = models_config['student_model']
    student_model = load_model(
        model_config=student_model_config,
        device=device,
        distributed=False, # Assuming non-distributed training for this script
        skip_ckpt=False # Try to load weights if available, otherwise starts fresh
    )
    # Ensure the model is on the correct device (load_model might already do this)
    student_model = student_model.to(device) 
    print(f"Student model '{student_model_config['name']}' loaded on {device}.")
    
    # --- 3. Define Loss functions ---
    # Loss weights from config
    loss_weights = config['train']['criterion']['weights']
    lw_main = loss_weights.get('lw_main', 1.0)
    lw_task_detector = loss_weights.get('lw_task_detector', 0.5)
    lw_rate = loss_weights.get('lw_rate', 0.05)

    main_loss_fn = torch.nn.CrossEntropyLoss().to(device)
    task_detector_loss_fn = MultiLabelTaskRelevancyBCELoss().to(device)
    
    # BppLossOrig setup
    # Assuming image input size for CIFAR-100 is 32x32.
    # The entropy_module_path needs to point to the attribute name of the entropy bottleneck
    # within the student_model. Typically 'compression_module.entropy_bottleneck'.
    rate_loss_fn = BppLossOrig(
        entropy_module_path='compression_module.entropy_bottleneck',
        input_sizes=[32, 32], # CIFAR-100 default size
        reduction='mean'
    ).to(device)
    print("Loss functions instantiated.")

    # --- 4. Define Optimizer ---
    optimizer_config = config['train']['optimizer']
    lr = optimizer_config.get('lr', 0.001)
    weight_decay = optimizer_config.get('weight_decay', 0.01)
    
    optimizer = optim.AdamW(student_model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"Optimizer AdamW instantiated with lr={lr}, weight_decay={weight_decay}.")

    # --- 5. Training loop ---
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        student_model.train() # Set model to training mode
        running_loss_main = 0.0
        running_loss_task_detector = 0.0
        running_loss_rate = 0.0
        running_loss_total = 0.0
        
        num_batches_to_run = batches_per_epoch if batches_per_epoch is not None else len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                break

            images, targets_batch_tuple, _, _ = batch # original_labels, assigned_task_ids not directly used in loss
            
            images = images.to(device)
            main_targets_batch = targets_batch_tuple[0].to(device)
            task_detector_targets_batch = targets_batch_tuple[1].to(device)
            
            # Model expects targets to be a tuple on the correct device for internal processing
            targets_for_model = (main_targets_batch, task_detector_targets_batch)

            optimizer.zero_grad()
            
            # --- Forward Pass ---
            output_dict = student_model(images, targets=targets_for_model)
            
            # --- Loss Calculation ---
            loss_main = main_loss_fn(output_dict['main_output'], main_targets_batch)
            loss_task_detector = task_detector_loss_fn(output_dict['conditioning_signal_preview'], task_detector_targets_batch)
            
            # Rate Loss (BPP)
            # This part assumes student_model.compression_module.entropy_bottleneck.likelihoods exists
            # and is populated correctly after the forward pass.
            # It might be a dict {'y': y_likelihoods, 'z': z_likelihoods} or a direct tensor.
            # The BppLossOrig seems to expect the 'y' likelihoods for hyperpriors.
            try:
                eb_module = student_model.compression_module.entropy_bottleneck
                if hasattr(eb_module, 'likelihoods'):
                    if isinstance(eb_module.likelihoods, dict) and 'y' in eb_module.likelihoods:
                        actual_likelihoods = eb_module.likelihoods['y']
                    elif isinstance(eb_module.likelihoods, torch.Tensor): # For simpler non-hyperprior models
                        actual_likelihoods = eb_module.likelihoods
                    else:
                        raise AttributeError("Likelihoods found but not in expected format (dict with 'y' or tensor).")
                    
                    mock_model_io_dict = {'compression_module.entropy_bottleneck': {'output': (None, actual_likelihoods)}}
                    loss_rate = rate_loss_fn(mock_model_io_dict)
                elif 'entropy_bottleneck_likelihoods_y' in output_dict: # Alternative: check output_dict
                     actual_likelihoods = output_dict['entropy_bottleneck_likelihoods_y']
                     mock_model_io_dict = {'compression_module.entropy_bottleneck': {'output': (None, actual_likelihoods)}}
                     loss_rate = rate_loss_fn(mock_model_io_dict)
                else:
                    print("Warning: Could not find likelihoods for rate loss. Setting loss_rate to 0.")
                    loss_rate = torch.tensor(0.0, device=device) # Avoid error if likelihoods aren't there
            except AttributeError as e:
                print(f"Warning: Error accessing likelihoods for rate loss: {e}. Setting loss_rate to 0.")
                loss_rate = torch.tensor(0.0, device=device)
            
            total_loss = (loss_main * lw_main) + \
                         (loss_task_detector * lw_task_detector) + \
                         (loss_rate * lw_rate)
            
            # --- Backward Pass and Optimizer Step ---
            total_loss.backward()
            optimizer.step()
            
            running_loss_main += loss_main.item()
            running_loss_task_detector += loss_task_detector.item()
            running_loss_rate += loss_rate.item() if isinstance(loss_rate, torch.Tensor) else loss_rate # handle if it became a float
            running_loss_total += total_loss.item()

            if (batch_idx + 1) % 10 == 0: # Log every 10 batches
                avg_loss_main = running_loss_main / (batch_idx + 1)
                avg_loss_task = running_loss_task_detector / (batch_idx + 1)
                avg_loss_rate = running_loss_rate / (batch_idx + 1)
                avg_loss_total = running_loss_total / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{num_batches_to_run}] | "
                      f"Loss Total: {avg_loss_total:.4f} | Main: {avg_loss_main:.4f} | "
                      f"TaskDet: {avg_loss_task:.4f} | Rate: {avg_loss_rate:.4f}")

        epoch_loss_total = running_loss_total / num_batches_to_run
        epoch_loss_main = running_loss_main / num_batches_to_run
        epoch_loss_task = running_loss_task_detector / num_batches_to_run
        epoch_loss_rate = running_loss_rate / num_batches_to_run
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Avg Loss Total: {epoch_loss_total:.4f} | Main: {epoch_loss_main:.4f} | "
              f"TaskDet: {epoch_loss_task:.4f} | Rate: {epoch_loss_rate:.4f}")
        print(f"--- Finished epoch {epoch+1} ---")

    print("--- Training Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or inspect a FiLMed network for CIFAR-100.")
    parser.add_argument('--mode', type=str, default='inspect', choices=['inspect', 'train'],
                        help="Mode to run the script in: 'inspect' or 'train'.")
    parser.add_argument('--config_path', type=str, default="config/mantis/cifar-100/resnet18.yaml",
                        help="Path to the YAML configuration file.")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training or inspection (e.g., 'cuda', 'cpu').")
    parser.add_argument('--num_epochs', type=int, default=3,
                        help="Number of epochs for training.")
    parser.add_argument('--batches_per_epoch', type=int, default=20,
                        help="Number of batches per epoch for training. If 0 or None, runs a full epoch.")

    args = parser.parse_args()

    if args.batches_per_epoch == 0:
        args.batches_per_epoch = None # Sentinel for full epoch

    if args.mode == 'inspect':
        inspect_data_and_model(args.config_path, args.device)
    elif args.mode == 'train':
        run_training(args.config_path, args.device, args.num_epochs, args.batches_per_epoch)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")