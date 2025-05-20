# model/datasets/registry.py
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets as torchvision_datasets # For easier access


DATASET_CLASS_DICT = dict()
DATASET_FUNC_DICT = dict() # For functions that return dataset instances
def parse_transform_config_list(transform_config_list: list):
    """
    Parses a list of transform configurations from YAML into a torchvision.transforms.Compose object.
    """
    if transform_config_list is None:
        return None
        
    active_transforms = []
    for tc_entry in transform_config_list:
        t_type = tc_entry['type']
        t_params = tc_entry.get('params', {})
        
        if t_type == 'RandomCrop':
            active_transforms.append(transforms.RandomCrop(**t_params))
        elif t_type == 'RandomResizedCrop':
            active_transforms.append(transforms.RandomResizedCrop(**t_params))
        elif t_type == 'RandomHorizontalFlip':
            active_transforms.append(transforms.RandomHorizontalFlip(**t_params))
        elif t_type == 'ToTensor':
            active_transforms.append(transforms.ToTensor()) # Usually no params needed from YAML for this
        elif t_type == 'Normalize':
            active_transforms.append(transforms.Normalize(**t_params))
        elif t_type == 'Resize':
            active_transforms.append(transforms.Resize(**t_params))
        elif t_type == 'CenterCrop':
            active_transforms.append(transforms.CenterCrop(**t_params))
        # Add more torchvision transforms as needed
        else:
            raise ValueError(f"Unsupported transform type in YAML: {t_type}")
    return transforms.Compose(active_transforms)
def register_dataset_class(cls):
    DATASET_CLASS_DICT[cls.__name__] = cls
    return cls

def register_dataset_func(func):
    DATASET_FUNC_DICT[func.__name__] = func
    return func

def get_dataset(config: dict) -> Dataset:
    if not isinstance(config, dict):
        raise TypeError(f"Dataset configuration must be a dict, but got {type(config)}")
    
    dataset_type = config.get('type', None)
    if dataset_type is None:
        raise ValueError(f"Dataset configuration must include a 'type' field. Config: {config}")

    dataset_params_config = config.get('params', {}) # Original params from YAML for this level
    
    processed_constructor_params = {}
    for key, value in dataset_params_config.items():
        if key == 'original_dataset' and isinstance(value, dict) and 'type' in value:
            # This is the nested 'original_dataset' for wrappers like LabelChunkedTaskDataset
            processed_constructor_params[key] = get_dataset(value) # Recursive call
        elif key == 'named_datasets' and isinstance(value, dict): # For MultiSourceTaskDataset
            processed_named_datasets = {}
            for task_name, dataset_conf_for_task in value.items():
                if isinstance(dataset_conf_for_task, dict) and 'type' in dataset_conf_for_task:
                    processed_named_datasets[task_name] = get_dataset(dataset_conf_for_task)
                else: # Should already be an instance if not a config dict
                    processed_named_datasets[task_name] = dataset_conf_for_task 
            processed_constructor_params[key] = processed_named_datasets
        elif key == 'transform' and isinstance(value, dict) and '_target_' in value and 'config' in value:
            # Handle the specific transform structure: {'_target_': ..., 'config': [list_of_transform_dicts]}
            # Assuming 'config' key holds the list of transform parameters
            print(f"DEBUG: Parsing transform config for key '{key}'")
            transform_list_config = value.get('config', [])
            processed_constructor_params[key] = parse_transform_config_list(transform_list_config)
        elif key == 'transform_params' and isinstance(value, list): # Legacy or direct list of transform configs
            print(f"DEBUG: Parsing transform_params list directly for key '{key}'")
            processed_constructor_params['transform'] = parse_transform_config_list(value) # Store as 'transform'
        else:
            processed_constructor_params[key] = value

    # After processing all params:
    if dataset_type in DATASET_CLASS_DICT:
        dataset_class = DATASET_CLASS_DICT[dataset_type]
        # Remove transform_params if transform object was created, to avoid passing both
        if 'transform' in processed_constructor_params and 'transform_params' in processed_constructor_params:
            del processed_constructor_params['transform_params']
        return dataset_class(**processed_constructor_params)
    elif dataset_type in DATASET_FUNC_DICT:
        # ... (same as above for transform_params if functions take it directly) ...
        return DATASET_FUNC_DICT[dataset_type](**processed_constructor_params)
    else:
        # Fallback for common torchvision datasets
        try:
            # Ensure 'transform_params' is removed if 'transform' was generated
            if 'transform' in processed_constructor_params and 'transform_params' in processed_constructor_params:
                 del processed_constructor_params['transform_params']

            if dataset_type == 'ImageFolder':
                return torchvision_datasets.ImageFolder(**processed_constructor_params)
            elif dataset_type == 'CIFAR100':
                return torchvision_datasets.CIFAR100(**processed_constructor_params)
            elif dataset_type == 'CIFAR10':
                return torchvision_datasets.CIFAR10(**processed_constructor_params)
        except Exception as e:
            print(f"ERROR: Failed to instantiate dataset '{dataset_type}' using torchvision fallback. "
                  f"Params: {processed_constructor_params}. Error: {e}")
            # Potentially re-raise or handle more gracefully
            raise e # Re-raise the error to see what went wrong during CIFAR100/ImageFolder init

        raise ValueError(f"Dataset type '{dataset_type}' not found in any known registry or torchvision fallbacks.")

