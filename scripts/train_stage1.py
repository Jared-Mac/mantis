#!/usr/bin/env python3
"""
Training script for MANTiS Stage 1: VIB + Head Distillation.

This script trains the initial stage of MANTiS which learns compressed
representations while preserving semantic information through head distillation.
"""

import os
import sys
import argparse
from pathlib import Path
import wandb

# Add src directory to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Import registry to register our custom components
import registry

# Import torchdistill components
from torchdistill.misc.log import setup_logger, init_logging
from torchdistill.core.distillation import get_distillation_box
from torchdistill.core.training import get_training_box, SimpleTrainer
from torchdistill.datasets import util as dataset_util
from torchdistill.misc.util import set_random_seed


def get_argparser():
    parser = argparse.ArgumentParser(description='MANTiS Stage 1 Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--log', type=str, default='info',
                        help='Logging level')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Run identifier for logging')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of distributed processes')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:23456',
                        help='URL for distributed training')
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='mantis-stage1-torchdistill',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[],
                        help='Wandb tags')
    
    return parser


def main():
    # Parse arguments
    parser = get_argparser()
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Setup logging
    init_logging(args.log)
    logger = setup_logger(__name__, args.log)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            config=vars(args)
        )
        logger.info("✓ Wandb initialized")
    
    logger.info(f"Starting MANTiS Stage 1 training with config: {args.config}")
    logger.info(f"Using device: {args.device}")
    
    # Get distillation box (loads models and datasets from config)
    distillation_box = get_distillation_box(args.config, args.device)
    
    # Get training box (loads training configurations)
    training_box = get_training_box(distillation_box, args.config, args.device)
    
    # Create trainer
    trainer = SimpleTrainer(training_box, device=args.device, log_freq=100)
    
    # Create checkpoint directory
    ckpt_dir = Path(distillation_box.config['save']['checkpoint_path'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training...")
    
    # Train the model
    trainer.train(
        num_epochs=distillation_box.config['train']['num_epochs'],
        save_interval=distillation_box.config['save']['checkpoint_interval'],
        save_dir=str(ckpt_dir)
    )
    
    logger.info("Training completed!")
    logger.info(f"Checkpoints saved to: {ckpt_dir}")
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()
        logger.info("✓ Wandb run completed")


if __name__ == '__main__':
    main() 