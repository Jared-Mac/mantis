#!/usr/bin/env python3
"""
Script to count parameters in the MANTiS client model (FrankenSplit architecture).
"""

import sys
from pathlib import Path
import torch

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from client.models import MANTiSClient


def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_component_parameters(model):
    """Count parameters for each component of the MANTiS client."""
    components = {
        'SharedStem': model.stem,
        'TaskDetector': model.task_detector,
        'FiLMGenerator': model.film_generator,
        'FiLMedEncoder': model.filmed_encoder,
    }
    
    component_counts = {}
    for name, component in components.items():
        component_counts[name] = count_parameters(component)
    
    return component_counts


def main():
    print("=== MANTiS Client Parameter Count (FrankenSplit Architecture) ===\n")
    
    # Create MANTiS client model
    model = MANTiSClient(num_tasks=3, latent_channels=48)
    
    # Count total parameters
    total_params = count_parameters(model)
    print(f"Total MANTiS Client Parameters: {total_params:,}")
    
    # Count parameters per component
    component_counts = count_component_parameters(model)
    print(f"\nComponent breakdown:")
    for component, count in component_counts.items():
        percentage = (count / total_params) * 100
        print(f"  {component}: {count:,} parameters ({percentage:.1f}%)")
    
    # Target comparison
    target_params = 140_000
    print(f"\nTarget: {target_params:,} parameters")
    print(f"Current: {total_params:,} parameters")
    
    if total_params <= target_params:
        percentage_of_target = (total_params / target_params) * 100
        print(f"✓ Under target by {target_params - total_params:,} parameters ({percentage_of_target:.1f}% of target)")
    else:
        print(f"✗ Over target by {total_params - target_params:,} parameters")
    
    # Test forward pass to get output dimensions
    print(f"\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        z, task_logits = model(x)
        print(f"Input shape: {tuple(x.shape)}")
        print(f"Latent shape: {tuple(z.shape)} = {z.numel():,} elements")
        print(f"Task logits shape: {tuple(task_logits.shape)}")


if __name__ == '__main__':
    main() 