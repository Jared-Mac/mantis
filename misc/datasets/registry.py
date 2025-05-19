# model/datasets/registry.py
from torch.utils.data import Dataset

DATASET_CLASS_DICT = dict()
DATASET_FUNC_DICT = dict() # For functions that return dataset instances

def register_dataset_class(cls):
    """
    Registers a dataset class.
    To be used as a class decorator.
    """
    DATASET_CLASS_DICT[cls.__name__] = cls
    return cls

def register_dataset_func(func):
    """
    Registers a function that returns a dataset instance.
    To be used as a function decorator.
    """
    DATASET_FUNC_DICT[func.__name__] = func
    return func

def get_dataset(config: dict) -> Dataset:
    """
    Builds and returns a dataset object based on the provided configuration.
    This function can handle nested dataset configurations.

    Args:
        config (dict): A dictionary containing dataset configuration.
                       Must include 'type' (class name or function name) 
                       and 'params' (a dictionary of parameters for the constructor/function).
                       'params' can contain nested dataset configurations.

    Returns:
        Dataset: An instance of the specified dataset.
    """
    if not isinstance(config, dict):
        raise TypeError(f"Dataset configuration must be a dict, but got {type(config)}")
    
    dataset_type = config.get('type', None)
    if dataset_type is None:
        raise ValueError(f"Dataset configuration must include a 'type' field. Config: {config}")

    dataset_params = config.get('params', {})
    
    # Recursively instantiate nested datasets if they are defined with 'type' and 'params'
    processed_params = {}
    for key, value in dataset_params.items():
        if isinstance(value, dict) and 'type' in value and 'params' in value and key != 'transform_params' and key != 'original_dataset': # Added conditions to skip transform_params and original_dataset
            # This is a nested dataset definition
            processed_params[key] = get_dataset(value)
        elif isinstance(value, list): # Handle list of nested datasets (e.g., for MultiSourceTaskDataset's values)
            processed_list = []
            is_list_of_datasets = True
            for item in value:
                # Ensure items in lists are not transforms before attempting to process as datasets
                if isinstance(item, dict) and 'type' in item and 'params' in item and not _is_transform_config(item):
                    processed_list.append(get_dataset(item))
                elif isinstance(item, dict) and _is_transform_config(item):
                    # If it's a transform config, keep it as is in the list
                    processed_list.append(item) 
                    # We still assume it's a list of datasets if some items are datasets
                    # and others are transforms. The dataset class itself should handle this.
                else:
                    is_list_of_datasets = False # Mark if any item is not a processable dataset or a transform
                    break 
            
            if is_list_of_datasets: # If all items were either datasets or transforms
                processed_params[key] = processed_list
            else: # If the list contained other types or was mixed in an unhandled way
                processed_params[key] = value # Keep original list
        elif isinstance(value, dict) and not ('type' in value and 'params' in value):
            # Handle dict of nested datasets (e.g. for MultiSourceTaskDataset)
            # The keys of this dict are task_ids, values are dataset configs
            # Also, ensure this dictionary itself is not a 'params' block of a transform or 'original_labels'
            if key == 'transform_params' or key == 'original_labels' or key == 'params': # check added
                 processed_params[key] = value
                 continue

            processed_dict_of_datasets = {}
            is_dict_of_datasets = True
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict) and 'type' in sub_value and 'params' in sub_value:
                    processed_dict_of_datasets[sub_key] = get_dataset(sub_value)
                else:
                    is_dict_of_datasets = False
                    break
            if is_dict_of_datasets:
                 processed_params[key] = processed_dict_of_datasets
            else:
                processed_params[key] = value # Keep original dict
        else:
            # This branch handles simple parameters and also transform_params dicts that were skipped by the first 'if'
            processed_params[key] = value

    if dataset_type in DATASET_CLASS_DICT:
        dataset_class = DATASET_CLASS_DICT[dataset_type]
        return dataset_class(**processed_params)
    elif dataset_type in DATASET_FUNC_DICT:
        dataset_func = DATASET_FUNC_DICT[dataset_type]
        return dataset_func(**processed_params)
    else:
        # Fallback: try to import from common libraries if not in custom registry
        # This part can be expanded or made more robust
        try:
            if dataset_type == 'ImageFolder':
                from torchvision.datasets import ImageFolder
                # ImageFolder expects root and transform.
                # Your YAML uses transform_params, so some adaptation might be needed here
                # or handled by a specific wrapper function if complex.
                # For now, assume basic ImageFolder instantiation.
                # This is a simplified example; robust handling of torchvision transforms from YAML is more involved.
                # Typically, a dedicated function or wrapper would parse 'transform_params'.
                # For now, we'll assume 'transform' is directly passable or handled by the calling framework.
                if 'transform_params' in processed_params and 'transform' not in processed_params:
                    # This is where you'd parse transform_params into a torchvision.transforms.Compose object
                    # For simplicity, this example won't implement the full transform parser.
                    # print(f"Warning: 'transform_params' found for ImageFolder but not processed into 'transform'.")
                    pass # Assuming transform is handled by calling framework or a dedicated transform parser
                return ImageFolder(**processed_params)
            # Add other common datasets like CIFAR10, MNIST etc. if needed
            elif dataset_type == 'CIFAR10':
                from torchvision.datasets import CIFAR10
                return CIFAR10(**processed_params)
            elif dataset_type == 'CIFAR100':
                from torchvision.datasets import CIFAR100
                return CIFAR100(**processed_params)
        except ImportError:
            pass # Will be caught by the final error

        raise ValueError(f"Dataset type '{dataset_type}' not found in DATASET_CLASS_DICT, "
                         f"DATASET_FUNC_DICT, or common torchvision datasets.")

def _is_transform_config(item_config: dict) -> bool:
    """
    Helper function to identify if a dictionary config looks like a transform.
    A more robust check might involve a predefined list of known transform types.
    """
    # Simple check: if 'type' ends with 'Crop', 'Flip', 'ToTensor', 'Normalize', etc.
    # This is a heuristic and might need to be more robust.
    known_transform_keywords = ['Crop', 'Flip', 'ToTensor', 'Normalize', 'Resize', 'Pad', 'ColorJitter', 'Grayscale', 'AugMix', 'RandAugment', 'TrivialAugmentWide']
    item_type = item_config.get('type', '')
    if any(keyword in item_type for keyword in known_transform_keywords):
        return True
    # Add any other specific checks if needed, e.g. based on parameter names
    return False