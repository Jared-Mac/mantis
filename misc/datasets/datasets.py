# model/datasets/multitask_wrappers.py (or misc/datasets/datasets.py)
import torch
from torch.utils.data import Dataset, ConcatDataset # Keep ConcatDataset for MultiSourceTaskDataset
from .registry import register_dataset_class # Assuming registry is in the same directory level

@register_dataset_class
class LabelChunkedTaskDataset(Dataset):
    def __init__(self, 
                 original_dataset: Dataset, 
                 task_configs: list, 
                 default_task_id: int = -1, 
                 default_task_specific_label: int = -1,
                 image_key=None,  # Added for WebDataset compatibility
                 label_key=None): # Added for WebDataset compatibility

        print(f"DEBUG: LabelChunkedTaskDataset received original_dataset of type: {type(original_dataset)}")
        if not isinstance(original_dataset, Dataset):
            print(f"DEBUG: ERROR - original_dataset is NOT a PyTorch Dataset instance!")

        self.original_dataset = original_dataset
        self.default_task_id = default_task_id
        self.default_task_specific_label = default_task_specific_label # Make sure this matches the parameter name
        self.image_key = image_key   # Store image_key
        self.label_key = label_key   # Store label_key

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
            # Ensure the map uses the actual task_id from config as key
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
                raise TypeError(f"original_labels for task {task_id} must be a list or a dict like {{'range': [start, end]}}.")

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

    def __getitem__(self, index: int):
        sample = self.original_dataset[index]

        if self.image_key is not None and self.label_key is not None:
            # Assume sample is a dictionary (e.g., from WebDataset)
            image = sample[self.image_key]
            original_label = sample[self.label_key]
        else:
            # Assume sample is a tuple, fallback to original behavior
            image, original_label = sample

        if isinstance(original_label, torch.Tensor):
            original_label = original_label.item()

        assigned_task_id_val = self.default_task_id 
        task_specific_label_for_main_task = torch.tensor(self.default_task_specific_label, dtype=torch.long)

        task_detector_target = torch.zeros(self.num_defined_tasks, dtype=torch.float32)

        matched_dense_idx = -1

        for config in self.processed_task_configs:
            if original_label in config['original_labels_set']:
                assigned_task_id_val = config['task_id']
                task_specific_label_for_main_task = torch.tensor(config['label_remapping'][original_label], dtype=torch.long)
                matched_dense_idx = config['dense_idx'] # Use the stored dense_idx
                if 0 <= matched_dense_idx < self.num_defined_tasks:
                    task_detector_target[matched_dense_idx] = 1.0
                break 

        targets_for_loss = (task_specific_label_for_main_task, task_detector_target)

        return image, targets_for_loss, original_label, assigned_task_id_val

    def __len__(self) -> int:
        return len(self.original_dataset)

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
@register_dataset_class
class MultiSourceTaskDataset(Dataset):
    """
    Combines multiple datasets, treating each as a distinct task.
    Returns (image, label_from_original_dataset, task_id).
    The task_id corresponds to the key provided in the named_datasets dictionary.
    """
    def __init__(self, 
                 named_datasets: dict, 
                 image_key=None,  # Added for WebDataset compatibility
                 label_key=None): # Added for WebDataset compatibility
        """
        Args:
            named_datasets (dict): A dictionary where keys are task_ids (str or int recommended)
                                   and values are the corresponding PyTorch Dataset instances.
                                   These dataset instances will be provided by recursive calls to get_dataset.
        """
        if not named_datasets:
            raise ValueError("named_datasets dictionary cannot be empty.")
            
        self.named_datasets = named_datasets
        self.image_key = image_key   # Store image_key
        self.label_key = label_key   # Store label_key
        self.task_ids_ordered = list(named_datasets.keys())

        datasets_for_concat = [self.named_datasets[task_id] for task_id in self.task_ids_ordered]
        
        self.concat_dataset = ConcatDataset(datasets_for_concat)
        
        self.sample_idx_to_task_id = []
        for i, task_id in enumerate(self.task_ids_ordered):
            self.sample_idx_to_task_id.extend([task_id] * len(datasets_for_concat[i]))
            
        assert len(self.concat_dataset) == len(self.sample_idx_to_task_id), \
            "Mismatch in total length after concatenation and task ID mapping."

    def __getitem__(self, index: int):
        sample = self.concat_dataset[index]
        task_id = self.sample_idx_to_task_id[index]

        if self.image_key is not None and self.label_key is not None:
            # Assume sample is a dictionary (e.g., from WebDataset)
            image = sample[self.image_key]
            label_from_original_dataset = sample[self.label_key]
        else:
            # Assume sample is a tuple, fallback to original behavior
            image, label_from_original_dataset = sample
            
        return image, label_from_original_dataset, task_id

    def __len__(self) -> int:
        return len(self.concat_dataset)

    def get_task_info(self):
        return [{
            'task_id': task_id, 
            'num_samples': len(self.named_datasets[task_id]),
            } for task_id in self.task_ids_ordered
        ]