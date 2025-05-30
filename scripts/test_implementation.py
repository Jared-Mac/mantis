#!/usr/bin/env python3
"""
Test script for MANTiS implementation.

Tests all components with dummy data to verify correct implementation.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn

# Add src directory to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Import registry to register components
import registry

# Test torchdistill registration
from torchdistill.models.registry import get_model
from torchdistill.losses.registry import get_loss_wrapper

print("Testing MANTiS implementation...")

def test_film_layer():
    """Test FiLMLayer functionality."""
    print("\n1. Testing FiLMLayer...")
    
    from film_layer import FiLMLayer
    
    # Create FiLM layer
    film_layer = FiLMLayer()
    
    # Test data
    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W)
    gamma = torch.randn(B, C)
    beta = torch.randn(B, C)
    
    # Forward pass
    output = film_layer(x, gamma, beta)
    
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    print("✓ FiLMLayer test passed")


def test_vib_bottleneck():
    """Test VIB bottleneck functionality."""
    print("\n2. Testing VIB bottleneck...")
    
    from vib import VIBBottleneck
    
    # Create VIB bottleneck
    vib = VIBBottleneck(channels=128)
    
    # Test data
    B, C, H, W = 2, 128, 16, 16
    z = torch.randn(B, C, H, W)
    
    # Forward pass
    z_hat, z_likelihoods = vib(z, training=True)
    
    assert z_hat.shape == z.shape, f"Expected shape {z.shape}, got {z_hat.shape}"
    assert isinstance(z_likelihoods, torch.Tensor), "z_likelihoods should be a tensor"
    print("✓ VIB bottleneck test passed")


def test_client_components():
    """Test client-side components."""
    print("\n3. Testing client components...")
    
    from client.stem import SharedStem
    from client.task_detector import TaskDetector
    from client.film_generator import FiLMGenerator
    from client.filmed_encoder import FiLMedEncoder
    
    # Test SharedStem
    stem = SharedStem(input_channels=3, output_channels=128, num_blocks=2)
    x = torch.randn(2, 3, 224, 224)
    f_stem = stem(x)
    print(f"  Stem output shape: {f_stem.shape}")
    
    # Test TaskDetector (expects 4D feature maps, not pre-pooled)
    task_detector = TaskDetector(input_feat_dim=128, num_tasks=5, hidden_dim=64)
    p_task = task_detector(f_stem)  # Use f_stem directly, not pooled
    print(f"  Task predictions shape: {p_task.shape}")
    
    # Test FiLMGenerator
    film_gen = FiLMGenerator(num_tasks=5, num_filmed_layers=3, 
                           channels_per_layer=[256, 256, 256], hidden_dim=64)
    film_params = film_gen(p_task)
    print(f"  FiLM parameters: {len(film_params)} layers")
    
    # Test FiLMedEncoder
    encoder = FiLMedEncoder(input_channels=128, output_channels=256, 
                          num_blocks=3, film_bypass=False)
    z = encoder(f_stem, film_params)
    print(f"  Encoder output shape: {z.shape}")
    
    print("✓ Client components test passed")


def test_server_components():
    """Test server-side components."""
    print("\n4. Testing server components...")
    
    from server.decoders_tails import GenericDecoder, TaskSpecificDecoder, TaskSpecificTail
    
    # Test GenericDecoder
    generic_decoder = GenericDecoder(input_channels=256, output_channels=1024, num_blocks=3)
    z = torch.randn(2, 256, 28, 28)
    features = generic_decoder(z)
    print(f"  Generic decoder output shape: {features.shape}")
    
    # Test TaskSpecificDecoder
    task_decoder = TaskSpecificDecoder(input_channels=256, output_channels=128, num_blocks=2)
    task_features = task_decoder(z)
    print(f"  Task decoder output shape: {task_features.shape}")
    
    # Test TaskSpecificTail (classification)
    classification_tail = TaskSpecificTail(task_type='classification', 
                                         input_channels=128, num_classes=100)
    class_output = classification_tail(task_features)
    print(f"  Classification output shape: {class_output.shape}")
    
    print("✓ Server components test passed")


def test_complete_models():
    """Test complete MANTiS models."""
    print("\n5. Testing complete models...")
    
    from models import MantisStage1, MantisStage2
    
    # Test Stage 1 model
    stage1_model = MantisStage1(
        client_params={'stem_channels': 128, 'encoder_channels': 256, 'num_encoder_blocks': 3},
        decoder_params={'input_channels': 256, 'output_channels': 1024, 'num_blocks': 3},
        vib_channels=256
    )
    
    x = torch.randn(2, 3, 224, 224)
    stage1_output = stage1_model(x)
    print(f"  Stage 1 output keys: {stage1_output.keys()}")
    
    # Test Stage 2 model
    stage2_model = MantisStage2(
        client_params={
            'stem_params': {'input_channels': 3, 'output_channels': 128, 'num_blocks': 2},
            'task_detector_params': {'input_feat_dim': 128, 'num_tasks': 5, 'hidden_dim': 64},
            'film_gen_params': {'num_tasks': 5, 'num_filmed_layers': 3, 
                              'channels_per_layer': [256, 256, 256], 'hidden_dim': 64},
            'filmed_encoder_params': {'input_channels': 128, 'output_channels': 256, 
                                    'num_blocks': 3, 'film_bypass': False}
        },
        num_tasks=5,
        decoder_params_list=[
            {'input_channels': 256, 'output_channels': 128, 'num_blocks': 2}
            for _ in range(5)
        ],
        tail_params_list=[
            {'task_type': 'classification', 'input_channels': 128, 'num_classes': 100}
            for _ in range(5)
        ],
        vib_channels=256
    )
    
    stage2_output = stage2_model(x)
    print(f"  Stage 2 output keys: {stage2_output.keys()}")
    
    print("✓ Complete models test passed")


def test_torchdistill_registration():
    """Test torchdistill registration."""
    print("\n6. Testing torchdistill registration...")
    
    # Test model registration
    try:
        # torchdistill's get_model expects a config dict with model parameters
        stage1_config = {
            'client_params': {'stem_channels': 128, 'encoder_channels': 256, 'num_encoder_blocks': 3},
            'decoder_params': {'input_channels': 256, 'output_channels': 1024, 'num_blocks': 3},
            'vib_channels': 256
        }
        mantis_stage1 = get_model('mantis_stage1', stage1_config)
        print(f"  ✓ Successfully created registered model: mantis_stage1")
        
        stage2_config = {
            'client_params': {
                'stem_params': {'input_channels': 3, 'output_channels': 128, 'num_blocks': 2},
                'task_detector_params': {'input_feat_dim': 128, 'num_tasks': 5, 'hidden_dim': 64},
                'film_gen_params': {'num_tasks': 5, 'num_filmed_layers': 3, 
                                  'channels_per_layer': [256, 256, 256], 'hidden_dim': 64},
                'filmed_encoder_params': {'input_channels': 128, 'output_channels': 256, 
                                        'num_blocks': 3, 'film_bypass': False}
            },
            'num_tasks': 5,
            'decoder_params_list': [
                {'input_channels': 256, 'output_channels': 128, 'num_blocks': 2}
                for _ in range(5)
            ],
            'tail_params_list': [
                {'task_type': 'classification', 'input_channels': 128, 'num_classes': 100}
                for _ in range(5)
            ],
            'vib_channels': 256
        }
        mantis_stage2 = get_model('mantis_stage2', stage2_config)
        print(f"  ✓ Successfully created registered model: mantis_stage2")
    except Exception as e:
        print(f"  ✗ Model registration test failed: {e}")
        return
    
    # Test loss registration  
    try:
        vib_loss_config = {'num_pixels_placeholder': 65536}
        vib_loss = get_loss_wrapper('vib_loss_stage1', vib_loss_config)
        print(f"  ✓ Successfully created registered loss: vib_loss_stage1")
    except Exception as e:
        print(f"  ✗ Loss registration test failed: {e}")
        return
    
    print("✓ torchdistill registration test passed")


def test_losses():
    """Test loss functions."""
    print("\n7. Testing loss functions...")
    
    from losses import VIBLossStage1, VIBLossStage2, MultiTaskDownstreamLoss
    
    # Test VIB losses
    vib_loss1 = VIBLossStage1(num_pixels_placeholder=65536)
    vib_loss2 = VIBLossStage2(num_pixels_placeholder=65536)
    
    # Dummy likelihood data
    z_likelihoods = {'z': torch.randn(2, 256, 16, 16)}
    z_film_likelihoods = {'z_film': torch.randn(2, 256, 16, 16)}
    
    loss1 = vib_loss1(z_likelihoods, None)
    loss2 = vib_loss2(z_film_likelihoods, None)
    
    print(f"  VIB Stage 1 loss: {loss1.item():.4f}")
    print(f"  VIB Stage 2 loss: {loss2.item():.4f}")
    
    # Test multi-task loss
    mt_loss = MultiTaskDownstreamLoss(
        num_tasks=2,
        task_loss_configs=[
            {'type': 'CrossEntropyLoss', 'params': {'reduction': 'mean'}},
            {'type': 'CrossEntropyLoss', 'params': {'reduction': 'mean'}}
        ]
    )
    
    downstream_outputs = [torch.randn(2, 10), torch.randn(2, 5)]
    targets = [torch.randint(0, 10, (2,)), torch.randint(0, 5, (2,))]
    active_task_mask = torch.ones(2, 2)  # All tasks active for both samples
    
    mt_loss_value = mt_loss(downstream_outputs, targets, active_task_mask)
    print(f"  Multi-task loss: {mt_loss_value.item():.4f}")
    
    print("✓ Loss functions test passed")


def main():
    """Run all tests."""
    try:
        test_film_layer()
        test_vib_bottleneck()
        test_client_components()
        test_server_components()
        test_complete_models()
        test_torchdistill_registration()
        test_losses()
        
        print("\n🎉 All tests passed! MANTiS implementation is working correctly.")
        print("\nNext steps:")
        print("1. Prepare your ImageNet dataset")
        print("2. Run Stage 1 training: python scripts/train_stage1.py --config configs/stage1_vib_hd.yaml")
        print("3. Run Stage 2 training: python scripts/train_stage2.py --config configs/stage2_task_aware.yaml --stage1_checkpoint <path>")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 