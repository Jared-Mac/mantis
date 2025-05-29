# misc/datasets/registry.py
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets as torchvision_datasets 
import webdataset # Make sure this is installed
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe # For type checking
import os
import json

from torchdistill.datasets import util as dataset_util # For build_transform if needed for legacy paths


DATASET_CLASS_DICT = dict()
DATASET_FUNC_DICT = dict() 

def parse_transform_config_list(transform_config_list: list):
    # ... (your existing parse_transform_config_list logic) ...
    if transform_config_list is None:
        return None
        
    active_transforms = []
    for tc_entry in transform_config_list:
        t_type = tc_entry['type']
        t_params = tc_entry.get('params', {})
        
        if hasattr(transforms, t_type):
            active_transforms.append(getattr(transforms, t_type)(**t_params))
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

    dataset_params_config = config.get('params', {})
    
    processed_constructor_params = {}
    for key, value in dataset_params_config.items():
        if key == 'original_dataset' and isinstance(value, dict) and 'type' in value:
            processed_constructor_params[key] = get_dataset(value) 
        elif key == 'named_datasets' and isinstance(value, dict): 
            processed_named_datasets = {}
            for task_name, dataset_conf_for_task in value.items():
                if isinstance(dataset_conf_for_task, dict) and 'type' in dataset_conf_for_task:
                    processed_named_datasets[task_name] = get_dataset(dataset_conf_for_task)
                else: 
                    processed_named_datasets[task_name] = dataset_conf_for_task 
            processed_constructor_params[key] = processed_named_datasets
        elif key == 'transform' and isinstance(value, dict) and '_target_' in value and value['_target_'] == 'misc.datasets.registry.parse_transform_config_list' and 'config' in value:
            transform_list_config = value.get('config', [])
            processed_constructor_params[key] = parse_transform_config_list(transform_list_config)
        elif key == 'transform_params' and isinstance(value, list): # Legacy torchdistill direct list
             # This path should ideally be updated in YAMLs to use the _target_ structure for clarity
            processed_constructor_params['transform'] = dataset_util.build_transform({'transform_params': value})
        else:
            processed_constructor_params[key] = value

    if dataset_type == 'WebDataset':
        if 'url' not in processed_constructor_params and 'urls' not in processed_constructor_params:
            raise ValueError("WebDataset config must include 'url' or 'urls' parameter.")
        
        urls = processed_constructor_params.pop('urls', processed_constructor_params.pop('url'))
        info_json_path = processed_constructor_params.pop('info_json_path', None)
        split_name = processed_constructor_params.pop('split_name', None) # e.g. 'train', 'validation'
        # 'transform' should already be processed into a Compose object if it was in params
        user_transform = processed_constructor_params.pop('transform', None) 

        length = None
        if info_json_path and split_name:
            expanded_info_path = os.path.expanduser(info_json_path)
            if os.path.exists(expanded_info_path):
                try:
                    with open(expanded_info_path, 'r') as f:
                        info_data = json.load(f)
                    # Expected format: {"splits": {"train": {"num_samples": X}, "validation": {"num_samples": Y}}}
                    length = info_data.get("splits", {}).get(split_name, {}).get("num_samples")
                    if length:
                        logger.info(f"Using length {length} from {expanded_info_path} for split '{split_name}'.")
                    else:
                        logger.warning(f"Could not find num_samples for split '{split_name}' in {expanded_info_path}. Length will be estimated by WebDataset if possible.")
                except Exception as e:
                    logger.warning(f"Could not read or parse info_json_path '{expanded_info_path}': {e}")
            else:
                logger.warning(f"info_json_path '{expanded_info_path}' does not exist.")
        else:
            logger.warning("info_json_path or split_name not provided for WebDataset. Length might not be available for DataLoader.")

        pipeline = [webdataset.SimpleShardList(urls)]
        # Add shuffling and tarfile_to_samples from WebDataset standard recommendations
        # These are often done by WebDataset itself or helper functions.
        # For torchdistill DataLoader, an IterableDataset is fine.
        # pipeline.extend([
        #     webdataset.split_by_worker, # if using multiple workers
        #     webdataset.tarfile_to_samples(), # Decodes samples from tar
        # ])
        # Decoding specific keys. 'autodecode' can simplify this.
        # imagehandler("torchrgb") converts to PIL, then to RGB, then to Tensor
        # If specific keys and decoders are needed:
        # pipeline.append(webdataset.decode(webdataset.imagehandler("torchrgb"), json_handler, ...))
        # Assuming autodecode handles common image formats to PIL/Tensor and labels.
        # The `image_key` and `label_key` in `LabelChunkedTaskDataset` will then pick these up.
        pipeline = webdataset.WebDataset(urls) # This handles sharding and tar reading

        if length is not None:
             pipeline = pipeline.with_length(length)
        
        # Common decoding pipeline:
        # This assumes standard keys like '.jpg', '.cls' in your tars.
        # Adjust image_decoder and label_decoder as needed.
        # webdataset.autodecode.imagehandler("torchrgb") first Tries to decode to PIL, then converts to RGB, then to Tensor if 'torch' is in the string.
        # It will use the extension of the file in the tar.
        # So, if your tar has 'sample.jpg' and 'sample.cls', and LabelChunkedTaskDataset uses image_key='jpg', label_key='cls'
        pipeline = pipeline.decode(webdataset.autodecode.imagehandler("torchrgb"))
        
        # Apply user-defined transforms (like Normalize, RandomCrop) after initial decoding
        if user_transform:
            # WebDataset samples are dicts. Transform needs to operate on the image within the dict.
            # Assuming the image key is 'jpg', 'png', or similar from the tar.
            # The image_key for LabelChunkedTaskDataset will be used *after* this transform
            # if this transform doesn't change the key.
            # It's common that `imagehandler` produces a key like 'jpg' or 'png'.
            def apply_transform_to_dict_image(sample):
                # Try common image keys that imagehandler might produce
                img_keys_to_try = ['jpg', 'png', 'jpeg', 'ppm', 'img', 'image']
                transformed = False
                for ik in img_keys_to_try:
                    if ik in sample:
                        sample[ik] = user_transform(sample[ik])
                        transformed = True
                        break
                if not transformed:
                    logger.warning(f"No common image key found in WebDataset sample to apply transform. Keys: {sample.keys()}")
                return sample
            pipeline = pipeline.map(apply_transform_to_dict_image)
        
        return pipeline

    elif dataset_type in DATASET_CLASS_DICT:
        dataset_class = DATASET_CLASS_DICT[dataset_type]
        # Special handling for LabelChunkedTaskDataset's original_dataset transform
        if dataset_type == 'LabelChunkedTaskDataset' and 'original_dataset' in processed_constructor_params:
            original_ds_instance = processed_constructor_params['original_dataset']
            # If the original_dataset instance itself doesn't have a transform,
            # but there was a transform_params for it in the YAML at a higher level,
            # it should have been applied during its own get_dataset call.
            # This part is tricky if LabelChunkedTaskDataset itself expects to apply a transform
            # to an already instantiated dataset. Torchdistill usually expects datasets to come pre-transformed.
            pass # Transform should have been handled when 'original_dataset' was recursively processed.

        return dataset_class(**processed_constructor_params)
    elif dataset_type in DATASET_FUNC_DICT:
        return DATASET_FUNC_DICT[dataset_type](**processed_constructor_params)
    else:
        # Fallback for common torchvision datasets
        try:
            if 'transform' not in processed_constructor_params and 'transform_params' in config: # Check top-level config for transform_params
                 processed_constructor_params['transform'] = dataset_util.build_transform(config)

            if dataset_type == 'ImageFolder':
                return torchvision_datasets.ImageFolder(**processed_constructor_params)
            elif dataset_type == 'CIFAR100':
                return torchvision_datasets.CIFAR100(**processed_constructor_params)
            elif dataset_type == 'CIFAR10':
                return torchvision_datasets.CIFAR10(**processed_constructor_params)
        except Exception as e:
            logger.error(f"ERROR: Failed to instantiate dataset '{dataset_type}' using torchvision fallback. "
                         f"Params: {processed_constructor_params}. Error: {e}")
            raise e
        raise ValueError(f"Dataset type '{dataset_type}' not found in any known registry or torchvision fallbacks.")