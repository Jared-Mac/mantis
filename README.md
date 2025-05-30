# MANTiS: Multi-Task Adaptive Neural Image Transmission System

This repository contains a PyTorch implementation of MANTiS, a neural compression system designed for multi-task learning scenarios. MANTiS uses Feature-wise Linear Modulation (FiLM) and Variational Information Bottleneck (VIB) principles to create task-adaptive compressed representations.

## Features

- **Two-stage training pipeline**: Generic VIB + Head Distillation → Task-aware multi-task learning
- **FiLM-based adaptation**: Task-conditioned feature modulation for efficient representation
- **CompressAI integration**: Leverages state-of-the-art neural compression components
- **torchdistill framework**: Structured knowledge distillation with configurable loss functions
- **Multi-task support**: Handles multiple downstream tasks with shared compressed representations
- **Ablation studies**: Built-in support for No-FiLM and Oracle baselines

## Architecture Overview

```
Input Image → [Client] → Compressed Representation → [Server] → Task Outputs
              ↓                                       ↓
         Task Detection                        Task-Specific
         FiLM Generation                       Decoders & Heads
```

### Client-Side Components
- **SharedStem**: Task-agnostic feature extraction
- **TaskDetector**: Predicts task probabilities from stem features  
- **FiLMGenerator**: Converts task probabilities to modulation parameters
- **FiLMedEncoder**: Task-conditioned feature encoding with FiLM layers

### Server-Side Components
- **VIB Bottleneck**: Variational quantization and entropy coding
- **TaskSpecificDecoders**: Per-task feature reconstruction
- **TaskSpecificTails**: Classification/segmentation heads

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mantis
```

2. Create a conda environment:
```bash
conda create -n mantis python=3.8
conda activate mantis
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install CompressAI and torchdistill:
```bash
# CompressAI
pip install compressai

# torchdistill from source (for latest features)
pip install git+https://github.com/yoshitomo-matsubara/torchdistill.git
```

## Quick Start

### 1. Test the Implementation
```bash
python scripts/test_implementation.py
```

This script verifies that all components work correctly with dummy data.

### 2. Prepare Data
Set up your ImageNet dataset:
```bash
# Create data directory structure
mkdir -p data/imagenet/{train,val}

# Symlink or copy your ImageNet data
ln -s /path/to/imagenet/train data/imagenet/train
ln -s /path/to/imagenet/val data/imagenet/val
```

### 3. Stage 1 Training (VIB + Head Distillation)
```bash
python scripts/train_stage1.py --config configs/stage1_vib_hd.yaml
```

### 4. Stage 2 Training (Task-Aware Learning)
```bash
python scripts/train_stage2.py --config configs/stage2_task_aware.yaml
```

## Configuration

The system uses YAML configuration files for flexible experimentation:

### Stage 1 Configuration (`configs/stage1_vib_hd.yaml`)
- Teacher model: Pre-trained ResNet-50
- Student model: MANTiS Stage 1 (stem + encoder + generic decoder)
- Loss: MSE head distillation + VIB rate loss
- Optimization: AdamW with cosine annealing

### Stage 2 Configuration (`configs/stage2_task_aware.yaml`)
- Student model: Full MANTiS with task detection and FiLM
- Multi-task dataset: ImageNet subgroups
- Loss: Task detection + downstream tasks + VIB rate
- Fine-tuning: Lower learning rates for pre-trained components

## Project Structure

```
mantis_project/
├── configs/                     # YAML configuration files
│   ├── stage1_vib_hd.yaml      # Stage 1 training config
│   └── stage2_task_aware.yaml   # Stage 2 training config
├── data/                        # Dataset symlinks/downloads
│   └── imagenet/
├── src/                         # Source code
│   ├── client/                  # Client-side components
│   │   ├── stem.py             # SharedStem
│   │   ├── task_detector.py    # TaskDetector
│   │   ├── film_generator.py   # FiLMGenerator
│   │   └── filmed_encoder.py   # FiLMedEncoder
│   ├── server/                  # Server-side components
│   │   └── decoders_tails.py   # Decoders and task heads
│   ├── film_layer.py           # FiLM layer implementation
│   ├── vib.py                  # VIB components
│   ├── models.py               # Complete model definitions
│   ├── losses.py               # Custom loss functions
│   └── datasets.py             # Multi-task dataset handling
├── scripts/                     # Training and evaluation scripts
│   ├── train_stage1.py         # Stage 1 training script
│   ├── train_stage2.py         # Stage 2 training script
│   ├── evaluate.py             # Evaluation script
│   └── test_implementation.py  # Implementation test
├── results/                     # Experimental results
├── saved_checkpoints/           # Model checkpoints
└── requirements.txt             # Dependencies
```

## Training Pipeline

### Stage 1: Generic VIB & Head Distillation
1. **Objective**: Learn a compressed representation that preserves semantic information
2. **Teacher**: Pre-trained ResNet-50 (frozen)
3. **Student**: MANTiS client encoder + generic decoder
4. **Loss**: `L = λ_hd * MSE(f_reconstructed, f_teacher) + β * Rate(z)`
5. **Output**: Pre-trained stem and encoder weights

### Stage 2: Task-Aware Multi-Task Learning
1. **Objective**: Learn task-specific adaptations and multi-task classification
2. **Initialization**: Load Stage 1 weights (freeze stem, fine-tune encoder)
3. **New Components**: Task detector, FiLM generator, task-specific decoders/heads
4. **Loss**: `L = λ_task * BCE(p_task, y_task) + Σ_k L_task_k + β' * Rate(z_film)`
5. **Output**: Complete MANTiS system

## Evaluation

### Metrics
- **Task Detection Accuracy**: How well the system identifies active tasks
- **Downstream Task Accuracy**: Classification performance on each task
- **Compression Rate**: Bits per pixel (BPP) of the compressed representation
- **Rate-Distortion Trade-off**: Accuracy vs. compression rate curves

### Baselines
- **MANTiS (No FiLM)**: Ablation without task-adaptive modulation
- **Oracle Task Selector**: Upper bound with ground truth task labels
- **Individual Task Models**: Separate models for each task (no compression)

## Customization

### Adding New Tasks
1. Update task definitions in `src/datasets.py`
2. Modify decoder/tail configurations in config files
3. Adjust loss configurations for new task types

### Custom Datasets
1. Implement dataset class following `ImageNetSubgroupsDataset` pattern
2. Update configuration files with new dataset paths
3. Modify task definitions as needed

### Architecture Changes
1. Modify component implementations in `src/`
2. Update model definitions in `src/models.py`
3. Adjust configuration parameters

## Advanced Usage

### Distributed Training
```bash
# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 \
    scripts/train_stage2.py --config configs/stage2_task_aware.yaml --distributed
```

### Hyperparameter Tuning
Modify key parameters in config files:
- Learning rates: `train.optimizer.kwargs.lr`
- Loss weights: `train.criterion.kwargs.sub_terms.*.weight`
- Architecture sizes: `models.student_model.kwargs.*`

### Custom Loss Functions
1. Implement new loss in `src/losses.py`
2. Register with torchdistill (if needed)
3. Update configuration to use new loss

## Research Extensions

### Potential Improvements
- **Dynamic Task Selection**: Learn to skip irrelevant tasks
- **Hierarchical Tasks**: Support for task taxonomies
- **Online Adaptation**: Adapt to new tasks without retraining
- **Attention Mechanisms**: Replace/augment FiLM with attention
- **Efficiency Optimizations**: Quantization-aware training, pruning

### Experimental Ideas
- Compare FiLM vs. other conditioning mechanisms
- Evaluate on different task combinations
- Study scaling behavior with number of tasks
- Investigate transfer learning capabilities

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{mantis2024,
  title={MANTiS: Multi-Task Adaptive Neural Image Transmission System},
  author={Your Name},
  journal={Conference/Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Acknowledgments

- [CompressAI](https://github.com/InterDigitalInc/CompressAI) for neural compression components
- [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) for knowledge distillation framework
- [PyTorch](https://pytorch.org/) ecosystem for deep learning infrastructure
