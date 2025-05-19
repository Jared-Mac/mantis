# model/datasets/__init__.py
from .registry import DATASET_CLASS_DICT, DATASET_FUNC_DICT, register_dataset_class, register_dataset_func, get_dataset
from .datasets import LabelChunkedTaskDataset, MultiSourceTaskDataset

# You can also explicitly add common torchvision datasets to your registry here if preferred,
# or handle them directly in the get_dataset function as shown.
# For example:
# from torchvision import datasets as tv_datasets
# if hasattr(tv_datasets, 'ImageFolder'):
#     register_dataset_class(tv_datasets.ImageFolder)
# if hasattr(tv_datasets, 'CIFAR10'):
#     register_dataset_class(tv_datasets.CIFAR10)
# ... and so on for datasets you commonly use from torchvision by string name in YAML.
# This is an alternative to the fallback logic in get_dataset.