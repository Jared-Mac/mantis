# model/datasets/multitask_wrappers.py (or misc/datasets/datasets.py)
import torch
from torch.utils.data import IterableDataset, ConcatDataset # Changed Dataset to IterableDataset
from .registry import register_dataset_class # Assuming registry is in the same directory level

@register_dataset_class
class LabelChunkedTaskDataset(IterableDataset): # Changed to IterableDataset
    def __init__(self, 
                 original_dataset: IterableDataset, # Type hint to IterableDataset
                 task_configs: list, 
                 default_task_id: int = -1, 
                 default_task_specific_label: int = -1,
                 image_key='jpg',  # Default image key
                 label_key='cls'): # Default label key
                 # Removed info_json_path and split_name, handled by WebDataset in registry

        print(f"DEBUG: LabelChunkedTaskDataset received original_dataset of type: {type(original_dataset)}")
        # No longer check isinstance(original_dataset, Dataset) strictly,
        # as it's now expected to be an IterDataPipe/IterableDataset.

        self.original_dataset = original_dataset
        self.default_task_id = default_task_id
        self.default_task_specific_label = default_task_specific_label
        self.image_key = image_key
        self.label_key = label_key

        self.processed_task_configs = []
        self.task_id_to_dense_idx_map = {} 

        unique_task_ids_from_config = []
        for i, tc in enumerate(task_configs):
            if not all(k in tc for k in ['task_id', 'original_labels']):
                raise ValueError("Each task_config must contain 'task_id' and 'original_labels'.")

            task_id = tc['task_id']
            if task_id not in unique_task_ids_from_config:
                unique_task_ids_from_config.append(task_id)

            current_one_hot_idx = unique_task_ids_from_config.index(task_id)
            self.task_id_to_dense_idx_map[task_id] = current_one_hot_idx

            original_labels_input = tc['original_labels']
            if isinstance(original_labels_input, dict) and 'range' in original_labels_input:
                if not (isinstance(original_labels_input['range'], list) and len(original_labels_input['range']) == 2):
                    raise ValueError("original_labels range must be a list of two integers [start, end].")
                start, end = original_labels_input['range']
                original_labels_set = set(range(start, end))
            elif isinstance(original_labels_input, list):
                original_labels_set = set(original_labels_input)
            else:
                raise TypeError(f"original_labels for task {task_id} must be a list or a dict like {{\'range\': [start, end]}}.")

            sorted_original_labels_for_task = sorted(list(original_labels_set))
            label_remapping = {
                original_label_val: task_specific_idx 
                for task_specific_idx, original_label_val in enumerate(sorted_original_labels_for_task)
            }

            self.processed_task_configs.append({
                'task_id': task_id,
                'dense_idx': current_one_hot_idx,
                'original_labels_set': original_labels_set,
                'label_remapping': label_remapping,
                'num_classes_in_task': len(original_labels_set)
            })

        self.num_defined_tasks = len(unique_task_ids_from_config)
        if self.num_defined_tasks == 0 and len(task_configs) > 0:
            print("Warning: Task configs provided but resulted in zero distinct tasks for the detector.")

    def __iter__(self): # Implemented __iter__
        for sample in self.original_dataset: # Iterate through the WebDataset
            # Process the sample (logic moved from __getitem__)
            if self.image_key is not None and self.label_key is not None:
                if self.image_key not in sample:
                    print(f"Warning: image_key '{self.image_key}' not found in sample keys: {list(sample.keys())}. Skipping sample.")
                    continue
                if self.label_key not in sample:
                    print(f"Warning: label_key '{self.label_key}' not found in sample keys: {list(sample.keys())}. Skipping sample.")
                    continue
                image = sample[self.image_key]
                original_label = sample[self.label_key]
            else:
                print("Warning: image_key or label_key not specified for LabelChunkedTaskDataset with WebDataset. Sample format might be unexpected.")
                try:
                    image, original_label = sample
                except (TypeError, ValueError) as e:
                    print(f"Error unpacking sample when image_key/label_key are not set. Sample: {sample}. Error: {e}. Skipping sample.")
                    continue

            if isinstance(original_label, torch.Tensor):
                original_label = original_label.item()

            assigned_task_id_val = self.default_task_id 
            task_specific_label_for_main_task = torch.tensor(self.default_task_specific_label, dtype=torch.long)
            task_detector_target = torch.zeros(self.num_defined_tasks, dtype=torch.float32)
            first_match_found = False # Initialize for each sample

            for config in self.processed_task_configs:
                if original_label in config['original_labels_set']:
                    matched_dense_idx = config['dense_idx'] 
                    if 0 <= matched_dense_idx < self.num_defined_tasks:
                        task_detector_target[matched_dense_idx] = 1.0 # Set for multi-label

                    if not first_match_found: # Assign primary task based on first match
                        assigned_task_id_val = config['task_id']
                        task_specific_label_for_main_task = torch.tensor(config['label_remapping'][original_label], dtype=torch.long)
                        first_match_found = True
                    # Removed break to allow checking all configs for multi-label task_detector_target
            
            targets_for_loss = (task_specific_label_for_main_task, task_detector_target)
            yield image, targets_for_loss, original_label, assigned_task_id_val

    # __len__ method is removed as this is now an IterableDataset.
    # The length is handled by WebDataset(...).with_length() for the DataLoader.

    def get_task_info(self):
        main_tasks_info = [{
            'task_id': tc['task_id'], 
            'num_classes': tc['num_classes_in_task'],
            'dense_idx': tc['dense_idx']
            } for tc in self.processed_task_configs
        ]
        return {
            "main_task_chunks_details": main_tasks_info,
            "num_distinct_task_chunks_for_predictor": self.num_defined_tasks 
        }

# Ensure MultiSourceTaskDataset is also in this file if not separate
# Note: MultiSourceTaskDataset uses ConcatDataset, which is map-style. 
# If it were to also wrap iterable datasets, it would need similar changes.
# For now, assuming it works with map-style datasets.
from torch.utils.data import Dataset # For MultiSourceTaskDataset if it remains map-style
@register_dataset_class
class MultiSourceTaskDataset(Dataset): # Keeping as Dataset for now
    """
    Combines multiple datasets, treating each as a distinct task.
    Returns (image, label_from_original_dataset, task_id).
    The task_id corresponds to the key provided in the named_datasets dictionary.
    """
    def __init__(self, 
                 named_datasets: dict, 
                 image_key=None, 
                 label_key=None): 
        if not named_datasets:
            raise ValueError("named_datasets dictionary cannot be empty.")
            
        self.named_datasets = named_datasets
        self.image_key = image_key
        self.label_key = label_key
        self.task_ids_ordered = list(named_datasets.keys())

        # Ensure all provided datasets are map-style if ConcatDataset is used
        for task_id, ds in named_datasets.items():
            if not isinstance(ds, Dataset) or isinstance(ds, IterableDataset):
                raise TypeError(f"Dataset for task \'{task_id}\' in MultiSourceTaskDataset must be a map-style Dataset, not IterableDataset, when using ConcatDataset.")

        datasets_for_concat = [self.named_datasets[task_id] for task_id in self.task_ids_ordered]
        
        self.concat_dataset = ConcatDataset(datasets_for_concat)
        
        self.sample_idx_to_task_id = []
        for i, task_id in enumerate(self.task_ids_ordered):
            # This len call assumes datasets_for_concat[i] is map-style
            self.sample_idx_to_task_id.extend([task_id] * len(datasets_for_concat[i]))
            
        assert len(self.concat_dataset) == len(self.sample_idx_to_task_id), \
            "Mismatch in total length after concatenation and task ID mapping."

    def __getitem__(self, index: int):
        sample_from_concat = self.concat_dataset[index] # ConcatDataset returns item from underlying dataset
        task_id = self.sample_idx_to_task_id[index]
        
        # Determine if the sample from concat_dataset is a dict (from WebDataset-like source) or tuple
        current_dataset_for_task = self.named_datasets[task_id]

        # Heuristic: if the original dataset instance stored in named_datasets has image_key/label_key attributes,
        # it suggests it was intended to be dictionary-like (e.g., a WebDataset).
        # This is a bit indirect. A more robust way would be to check types or have explicit config.
        is_dict_sample = False
        if hasattr(current_dataset_for_task, 'image_key') and getattr(current_dataset_for_task, 'image_key') is not None and \
           hasattr(current_dataset_for_task, 'label_key') and getattr(current_dataset_for_task, 'label_key') is not None:
            is_dict_sample = True 
        elif isinstance(sample_from_concat, dict) and self.image_key in sample_from_concat and self.label_key in sample_from_concat:
            is_dict_sample = True

        if is_dict_sample:
            image = sample_from_concat[self.image_key]
            label_from_original_dataset = sample_from_concat[self.label_key]
        elif isinstance(sample_from_concat, tuple) and len(sample_from_concat) == 2:
            image, label_from_original_dataset = sample_from_concat
        else:
            raise ValueError(f"Unexpected sample format from dataset for task \'{task_id}\'. Expected dict with keys or 2-tuple. Got: {type(sample_from_concat)}")
            
        return image, label_from_original_dataset, task_id

    def __len__(self) -> int:
        return len(self.concat_dataset)

    def get_task_info(self):
        # This len call assumes self.named_datasets[task_id] is map-style
        return [{
            'task_id': task_id, 
            'num_samples': len(self.named_datasets[task_id]),
            } for task_id in self.task_ids_ordered
        ]