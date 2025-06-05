"""
Registry module for MANTiS components with torchdistill.

This module registers all custom models, losses, and other components
so they can be referenced by key in YAML configuration files.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torchdistill.models.registry import register_model
from torchdistill.losses.registry import register_loss_wrapper
from torchdistill.datasets.registry import register_dataset_wrapper

# Import our components
from models import MantisStage1, MantisStage2, MantisNoFiLMStage2, MantisOracleStage2
from losses import (
    VIBLossStage1, VIBLossStage2, MultiTaskDownstreamLoss, 
    MultiTaskCriterionWrapper, TaskDetectorLoss, CombinedMantisLoss
)
from datasets import ImageNetSubgroupsDataset, MultiDatasetWrapper, create_mantis_datasets


# Register models
@register_model
def mantis_stage1(**kwargs):
    """MANTiS Stage 1 model for VIB + Head Distillation."""
    return MantisStage1(**kwargs)


@register_model
def mantis_stage2(**kwargs):
    """MANTiS Stage 2 model for task-aware learning."""
    return MantisStage2(**kwargs)


@register_model
def mantis_no_film_stage2(**kwargs):
    """MANTiS Stage 2 ablation without FiLM."""
    return MantisNoFiLMStage2(**kwargs)


@register_model
def mantis_oracle_stage2(**kwargs):
    """MANTiS Stage 2 oracle baseline."""
    return MantisOracleStage2(**kwargs)


# Register loss functions
@register_loss_wrapper
def vib_loss_stage1(**kwargs):
    """VIB rate loss for Stage 1."""
    return VIBLossStage1(**kwargs)


@register_loss_wrapper
def vib_loss_stage2(**kwargs):
    """VIB rate loss for Stage 2."""
    return VIBLossStage2(**kwargs)


@register_loss_wrapper
def multi_task_downstream_loss(**kwargs):
    """Multi-task downstream loss."""
    return MultiTaskDownstreamLoss(**kwargs)


@register_loss_wrapper
def multi_task_criterion_wrapper(**kwargs):
    """Multi-task criterion wrapper for torchdistill."""
    return MultiTaskCriterionWrapper(**kwargs)


@register_loss_wrapper
def task_detector_loss(**kwargs):
    """Task detection loss."""
    return TaskDetectorLoss(**kwargs)


@register_loss_wrapper
def combined_mantis_loss(**kwargs):
    """Combined MANTiS loss for Stage 2."""
    return CombinedMantisLoss(**kwargs)


# Register datasets
@register_dataset_wrapper
def imagenet_subgroups_dataset(**kwargs):
    """ImageNet subgroups dataset for multi-task learning."""
    return ImageNetSubgroupsDataset(**kwargs)


@register_dataset_wrapper
def multi_dataset_wrapper(**kwargs):
    """Multi-dataset wrapper."""
    return MultiDatasetWrapper(**kwargs)


# Helper functions that can be called from YAML
def create_imagenet_task_definitions():
    """Helper function for YAML configs."""
    from datasets import create_imagenet_task_definitions as _create_imagenet_task_definitions
    return _create_imagenet_task_definitions()


def get_imagenet_transforms(is_training=True, image_size=224):
    """Helper function for YAML configs."""
    from datasets import get_imagenet_transforms as _get_imagenet_transforms
    return _get_imagenet_transforms(is_training=is_training, image_size=image_size)


# New dataset constructor functions
def get_mantis_train_dataset(data_root, image_size=224, use_multidataset=False, **kwargs):
    """
    Gets the MANTiS training dataset.

    Args:
        data_root: Root directory for datasets.
        image_size: Target image size.
        use_multidataset: Whether to use the multi-dataset setup (e.g., CIFAR100 + Flowers102)
                          or the default ImageNet subgroups.
        **kwargs: Additional arguments to pass to create_mantis_datasets.

    Returns:
        The training dataset object.
    """
    datasets_dict = create_mantis_datasets(
        data_root=data_root,
        image_size=image_size,
        use_multidataset=use_multidataset,
        **kwargs
    )
    return datasets_dict['train']


def get_mantis_val_dataset(data_root, image_size=224, use_multidataset=False, **kwargs):
    """
    Gets the MANTiS validation dataset.

    Args:
        data_root: Root directory for datasets.
        image_size: Target image size.
        use_multidataset: Whether to use the multi-dataset setup (e.g., CIFAR100 + Flowers102)
                          or the default ImageNet subgroups.
        **kwargs: Additional arguments to pass to create_mantis_datasets.

    Returns:
        The validation dataset object.
    """
    datasets_dict = create_mantis_datasets(
        data_root=data_root,
        image_size=image_size,
        use_multidataset=use_multidataset,
        **kwargs
    )
    return datasets_dict['val']


print("MANTiS components registered with torchdistill!")