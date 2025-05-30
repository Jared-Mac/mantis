# MANTiS Research Plan: Leveraging CompressAI and torchdistill

This document outlines a plan to implement MANTiS using CompressAI and torchdistill.

## Recommended Directory Structure

A well-organized directory structure is crucial for a project of this complexity.

```
mantis_project/
├── configs/                     # YAML configuration files for torchdistill
│   ├── stage1_vib_hd.yaml
│   └── stage2_task_aware.yaml
├── data/                        # Placeholder for symlinks or dataset download scripts
│   ├── imagenet/
│   ├── cifar100/
│   └── ...
├── pretrained_models/           # Pretrained teacher models (e.g., ResNet-50)
├── results/                     # For storing RD curves, metrics, plots
│   ├── stage1/
│   └── stage2/
├── saved_checkpoints/           # Checkpoints from your MANTiS training
│   ├── stage1/
│   └── stage2/
├── src/                         # Your MANTiS source code
│   ├── client/                  # Client-side components
│   │   ├── __init__.py
│   │   ├── stem.py
│   │   ├── task_detector.py
│   │   ├── film_generator.py
│   │   └── filmed_encoder.py    # Contains FiLMed residual blocks
│   ├── server/                  # Server-side components
│   │   ├── __init__.py
│   │   └── decoders_tails.py    # Generic and task-specific decoders/tails
│   ├── film_layer.py            # FiLM layer implementation
│   ├── vib.py                   # VIB components (quantization, prior)
│   ├── datasets.py              # Custom dataset handling (e.g., ImageNet-Subgroups)
│   ├── losses.py                # Custom loss definitions if needed beyond torchdistill
│   └── models.py                # Top-level MANTiS model definition
├── scripts/                     # Helper scripts
│   ├── prepare_imagenet_subgroups.py
│   ├── train_stage1.py
│   ├── train_stage2.py
│   └── evaluate.py
└── README.md
```

## Implementation Plan using CompressAI and torchdistill

We'll go through each section of the MANTiS plan and detail how the libraries fit in.

### 2. Client-Side Architecture

#### Shared Stem (Block 1) & FiLMed Encoder (Blocks 2 and 3):

*   **Library**: CompressAI (`compressai.layers`)
*   **Implementation**:
    *   In `src/client/stem.py`: Implement `SharedStem` as an `nn.Module` using `ResidualBlockWithStride` or similar from `compressai.layers`. This will be your \\(g_a\\) (or part of it) for Stage 1.
    *   In `src/client/filmed_encoder.py`:
        *   Define `FiLMedResidualBlock` which takes a standard `ResidualBlock` (from `compressai.layers`) and integrates your custom `FiLMLayer` (see below).
        *   `FiLMedEncoder` will be an `nn.Module` composed of these `FiLMedResidualBlocks`.

#### `src/film_layer.py`:

*   **Library**: PyTorch (`torch.nn`)
*   **Implementation**: Define a `FiLMLayer(nn.Module)`:

```python
import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        # x: (B, C, H, W) feature map
        # gamma, beta: (B, C, 1, 1) or (B, C) - need to reshape
        # Ensure gamma and beta are broadcastable
        if gamma.ndim == 2: # (B, C)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        return gamma * x + beta
```

#### Task Detector:

*   **Library**: PyTorch (`torch.nn`)
*   **Implementation**: In `src/client/task_detector.py`:

```python
import torch
import torch.nn as nn

class TaskDetector(nn.Module):
    def __init__(self, input_feat_dim, num_tasks, hidden_dim=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_tasks),
            nn.Sigmoid()
        )
    def forward(self, f_stem):
        x = self.pool(f_stem)
        x = self.flatten(x)
        return self.fc_layers(x)
```

#### FiLM Generator:

*   **Library**: PyTorch (`torch.nn`)
*   **Implementation**: In `src/client/film_generator.py`:

```python
import torch
import torch.nn as nn

class FiLMGenerator(nn.Module):
    def __init__(self, num_tasks, num_filmed_layers, channels_per_layer, hidden_dim=32):
        # channels_per_layer: list of channel counts for each FiLMed conv layer
        super().__init__()
        self.num_filmed_layers = num_filmed_layers
        self.channels_per_layer = channels_per_layer
        total_film_params = sum(c * 2 for c in channels_per_layer) # *2 for gamma and beta

        self.fc_layers = nn.Sequential(
            nn.Linear(num_tasks, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, total_film_params)
        )

    def forward(self, p_task):
        film_params_flat = self.fc_layers(p_task)
        # Reshape film_params_flat into a list of (gamma, beta) tuples for each layer
        output_params = []
        current_idx = 0
        for num_channels in self.channels_per_layer:
            gamma = film_params_flat[:, current_idx : current_idx + num_channels]
            beta = film_params_flat[:, current_idx + num_channels : current_idx + 2 * num_channels]
            output_params.append((gamma, beta))
            current_idx += 2 * num_channels
        return output_params # List of (gamma_layer_j, beta_layer_j)
```
You'll need to carefully manage how these parameters are passed to and used by the `FiLMedEncoder`.

#### Quantization and Coding (VIB components):

*   **Library**: CompressAI (`compressai.entropy_models`, `compressai.ops`)
*   **Implementation**: In `src/vib.py`:

    *   **Training Simulation**:
        ```python
        import torch

        def add_quantization_noise(z, training=True):
            if training:
                # Uniform noise for training
                noise = torch.empty_like(z).uniform_(-0.5, 0.5)
                return z + noise
            # Simple rounding for inference (actual quantization handled separately)
            return torch.round(z)
        ```

    *   **Prior for VIB (Rate Term)**:
        Use `compressai.entropy_models.EntropyBottleneck` for the factorized Gaussian prior.
        The `forward` pass of this `EntropyBottleneck` instance on \\(z\\) (or \\(z\\) + noise) will give you `z_hat` and `z_likelihoods`. The rate term is then `(-z_likelihoods.log2()).sum() / num_pixels`.

    *   **Inference Quantization & Coding**:
        *   **Quantization**: Simple `torch.round()` can be a starting point for simulation. For actual deployment, you'd need integer quantization.
        *   **Entropy Coding**: CompressAI's `EntropyBottleneck` has `compress` and `decompress` methods which internally use an entropy coder (e.g., ANS). You'd call `entropy_bottleneck.update(force=True)` after training. Then `entropy_bottleneck.compress(z_quantized)` would give you the bitstream. However, the plan mentions Arithmetic coding. CompressAI primarily uses ANS. If Arithmetic coding is a strict requirement, you might need to integrate a separate library or implement it, and use the PMFs/CDFs from CompressAI's entropy models. For simplicity with CompressAI, stick to its built-in coders first.

### 3. Server-Side Architecture

*   **Library**: PyTorch (`torch.nn`), potentially `compressai.layers` for decoder building blocks.
*   **Implementation**: In `src/server/decoders_tails.py`:
    *   `GenericDecoder(nn.Module)`: Mirrors the encoder's upsampling path (e.g., using `ResidualBlockUpsample` from `compressai.layers`).
    *   `TaskSpecificDecoder(nn.Module)`: Could be instances of `GenericDecoder` or specialized.
    *   `TaskSpecificTail(nn.Module)`: Standard classification/segmentation heads.

    ```python
    import torch
    import torch.nn as nn

    # Example for a classification task
    class ClassificationTail(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(input_dim, num_classes)
        def forward(self, x):
            x = self.pool(x)
            x = self.flatten(x)
            return self.fc(x)
    ```

### 4. Training Strategy

This is where `torchdistill` shines.

#### 4.1. Stage 1: Generic VIB & HD Training

*   **YAML Config (`configs/stage1_vib_hd.yaml`)**:

    ```yaml
    models:
      teacher_model:
        key: 'resnet50' # Or your chosen pretrained backbone
        _weights: !import_get
          key: 'torchvision.models.resnet.ResNet50_Weights'
        kwargs:
          weights: !getattr [!import_get {key: 'torchvision.models.resnet.ResNet50_Weights'}, 'IMAGENET1K_V1']
        forward_hook:
          output: ['layer3'] # Example: output of ResNet's layer3 as P_h(x)

      student_model: # This is your MANTiS client + generic server decoder
        key: '!import_call' # Your MANTiS Stage 1 model class
        init:
          kwargs:
            # Params for SharedStem, FiLMedEncoder (in identity mode), GenericDecoder
            # e.g., num_channels_stem, num_channels_encoder, num_channels_decoder
            # VIB entropy bottleneck channels
        forward_hook:
          output: ['g_s_output', 'z_likelihoods'] # Output of generic decoder and VIB likelihoods

    train:
      # ... (dataloader, optimizer, scheduler configs) ...
      criterion:
        key: 'WeightedSumLoss'
        kwargs:
          sub_terms:
            hd_loss:
              criterion:
                key: 'MSELoss' # MSE for Head Distillation
                kwargs:
                  reduction: 'mean'
              criterion_wrapper:
                key: 'SimpleLossWrapper'
                kwargs:
                  input: # Student's reconstructed features (output of generic_decoder)
                    is_from_teacher: False
                    module_path: '.' # Assuming g_s_output is a direct output
                    io: 'g_s_output'
                  target: # Teacher's head output
                    is_from_teacher: True
                    module_path: 'layer3' # Matches teacher_model.forward_hook
                    io: 'output'
              weight: 1.0 # Your lambda_hd equivalent
            
            rate_loss: # VIB rate term
              criterion:
                key: '!import_call' # Custom rate loss module from src.losses
                init:
                  kwargs:
                    # pass num_pixels or calculate inside
              criterion_wrapper:
                key: 'SimpleLossWrapper' # or custom if needed
                kwargs:
                  input:
                    is_from_teacher: False
                    module_path: '.' # Assuming z_likelihoods is a direct output
                    io: 'z_likelihoods'
                  target: # Dummy target, not used by rate loss
                    uses_label: False # Or provide a dummy
              weight: 0.01 # Your beta_stage1
    ```

*   **`src/models.py` (for Stage 1)**:

    ```python
    import torch
    from torch import nn
    # Assuming these modules are defined in your project structure
    # from .client.stem import SharedStem
    # from .client.filmed_encoder import FiLMedEncoder # Configured for identity FiLM
    # from .server.decoders_tails import GenericDecoder
    # from .vib import add_quantization_noise # Assuming this is defined
    from compressai.entropy_models import EntropyBottleneck # For P(z)

    # Placeholder for actual imports if they exist in the specified paths
    class SharedStem(nn.Module): # Placeholder
        def __init__(self, *args, **kwargs): super().__init__(); self.conv = nn.Conv2d(3, 64, 3, 1, 1)
        def forward(self, x): return self.conv(x)

    class FiLMedEncoder(nn.Module): # Placeholder
        def __init__(self, *args, film_bypass=True, **kwargs): super().__init__(); self.conv = nn.Conv2d(64, 128, 3, 1, 1)
        def forward(self, x, film_params_list=None): return self.conv(x)
            
    class GenericDecoder(nn.Module): # Placeholder
        def __init__(self, *args, **kwargs): super().__init__(); self.conv = nn.Conv2d(128, 64, 3, 1, 1)
        def forward(self, x): return self.conv(x)

    def add_quantization_noise(z, training=True): # Placeholder from text
        if training:
            noise = torch.empty_like(z).uniform_(-0.5, 0.5)
            return z + noise
        return torch.round(z)


    class MantisStage1Client(nn.Module):
        def __init__(self, stem_channels, encoder_channels_list, latent_dim):
            super().__init__()
            self.stem = SharedStem() # Pass actual params
            self.encoder = FiLMedEncoder(film_bypass=True) # Pass actual params
            # No TaskDetector, No FiLMGenerator in Stage 1
        def forward(self, x):
            f_stem = self.stem(x)
            z = self.encoder(f_stem) # FiLM is bypassed
            return z

    class MantisStage1(nn.Module):
        def __init__(self, client_params, decoder_params, vib_channels):
            super().__init__()
            self.client_encoder = MantisStage1Client(**client_params)
            self.vib_bottleneck = EntropyBottleneck(vib_channels) # For P(z)
            self.generic_decoder = GenericDecoder(**decoder_params)

        def forward(self, x):
            z_raw = self.client_encoder(x)
            # Add quantization noise (from src.vib)
            z_quant_sim = add_quantization_noise(z_raw, self.training)
            z_hat, z_likelihoods = self.vib_bottleneck(z_quant_sim) # Or z_raw if noise is part of VIB
            
            # For HD loss, we might want to use z_hat or z_quant_sim. Paper uses z_raw + eta.
            # Let's assume HD uses the output that's also used for rate calculation.
            reconstructed_features = self.generic_decoder(z_hat) # or z_quant_sim for decoder input
            
            return {
                "g_s_output": reconstructed_features,
                "z_likelihoods": {"z": z_likelihoods} # For the rate term
            }
    ```

*   **`src/losses.py` (Rate Loss for Stage 1)**:

    ```python
    import torch
    import torch.nn as nn
    import math

    class VIBlossStage1(nn.Module):
        def __init__(self, num_pixels_placeholder=256*256, **kwargs): # Or pass dynamically
            super().__init__()
            self.num_pixels_placeholder = num_pixels_placeholder

        def forward(self, student_io_dict, target_dummy=None, **kwargs):
            # student_io_dict might contain the z_likelihoods under a specific key from forward_hook
            # For example, if forward_hook output name for likelihoods is 'z_likelihoods'
            # and it directly returns the dictionary {"z": likelihoods_tensor}
            # then student_io_dict might be student_io_dict['.']['z_likelihoods'] based on SimpleLossWrapper
            
            # Assuming z_likelihoods_dict is passed directly or extracted correctly
            # This part needs to align with how SimpleLossWrapper passes 'input'
            if isinstance(student_io_dict, dict) and "z" in student_io_dict:
                 z_likelihoods_dict = student_io_dict
            elif isinstance(student_io_dict, torch.Tensor): # If it's just the tensor from EB
                 z_likelihoods_dict = {"z": student_io_dict}
            else:
                # This case needs to be handled based on actual structure from torchdistill
                # For now, let's assume it's passed as a dict as intended by the model output
                # This is a common point of debugging in torchdistill setups.
                # We'll assume the wrapper handles extraction and student_io_dict is the {"z": likelihoods_tensor}
                if not (isinstance(student_io_dict, dict) and "z" in student_io_dict):
                    raise ValueError("VIBlossStage1 expects z_likelihoods as a dict {'z': tensor} or just the tensor.")
                z_likelihoods_dict = student_io_dict


            z_likelihoods = z_likelihoods_dict["z"]
            
            # The VIB rate term is E[log P(Z)] or sum(-log P_model(z_i))
            # CompressAI likelihoods are P(z), so rate is -log2(P(z))
            
            # If z_likelihoods is (B, C, H, W)
            if z_likelihoods.ndim == 4: # (B, C, H, W) from .likelihoods()
                b = z_likelihoods.shape[0]
                # num_elements = z_likelihoods.shape[1] * z_likelihoods.shape[2] * z_likelihoods.shape[3]
                # Rate per image: sum over C,H,W then mean over B
                # Rate per pixel: sum over B,C,H,W then divide by B * H * W
                # The definition in paper usually is sum(-log P_model(z_i)) / (N*num_pixels)
                # Let's assume num_pixels_placeholder is H*W for one image
                rate = torch.sum(torch.log(z_likelihoods)) / (-math.log(2) * b * self.num_pixels_placeholder)
            # If z_likelihoods from EntropyBottleneck.forward() is (B, C, H, W) for likelihoods
            # and not the (C, 1, B*H*W_permuted) from older CompressAI or direct pmf manipulation
            # The shape of `z_likelihoods` from `EntropyBottleneck.forward` is typically the same as `z_hat`.
            elif z_likelihoods.ndim > 1 : # (B, C, H, W) or similar
                b = z_likelihoods.shape[0]
                # Assuming self.num_pixels_placeholder is H*W (spatial dimensions)
                # The rate is sum over all elements (Batch, Channel, Spatial) divided by (Batch_size * num_spatial_elements)
                # Or sum over (Channel, Spatial) and mean over Batch, then divide by num_spatial_elements
                # Rate = E_{x~p_data} E_{q(z|x)} [log q(z|x) - log p(z)]
                # Here we only have -log p(z) from likelihoods
                # sum(-log2 P(z)) / (B * H * W)
                rate = torch.sum(-torch.log2(z_likelihoods)) / (b * self.num_pixels_placeholder)

            else: 
                # This case for (C, 1, B*H*W_permuted) is less common with modern EB forward output
                # It might occur if manipulating internal PMFs directly
                # Let's assume z_likelihoods is per-element probability (B, C, H, W)
                # If it's something else, this part needs careful adjustment
                actual_batch_size = kwargs.get("batch_size", 1) # Needs to be passed in supp_dict if not B from shape
                rate = torch.sum(torch.log(z_likelihoods)) / (-math.log(2) * actual_batch_size * self.num_pixels_placeholder)
            return rate
    ```

You'll need to register `VIBlossStage1` with `torchdistill` if you define it this way.

#### 4.2. Stage 2: Joint Task-Aware & Task-Specific Training

*   **YAML Config (`configs/stage2_task_aware.yaml`)**:

    ```yaml
    models:
      student_model: # MANTiS full client + N server decoders/tails
        key: '!import_call' # Your MANTiS Stage 2 model class
        init:
          kwargs:
            # stem_checkpoint_path: path to Stage 1 stem weights
            # encoder_conv_checkpoint_path: path to Stage 1 encoder conv weights
            # num_tasks, FiLM generator params, FiLMedEncoder params,
            # task_decoder_params_list, task_tail_params_list
            # VIB entropy bottleneck channels
        # src_ckpt: # This would load the whole MantisStage2 model if resuming
        # Alternatively, load parts selectively in the model's __init__
        # For fine-tuning, parts of the student_model could be frozen:
        frozen_modules: ['client.stem'] # Freeze shared stem
        forward_hook:
          output: ['task_predictions', 'downstream_outputs', 'z_film_likelihoods']
          # downstream_outputs should be a list or dict of N task outputs

    train:
      # ... (dataloader for multi-label, multi-task data, optimizer, scheduler configs) ...
      # Optimizer might need to handle different LRs for fine-tuning encoder
      optimizer:
        key: 'AdamW'
        module_wise_configs:
          - module_path: 'client.filmed_encoder' # Fine-tune FiLMed Encoder convs
            kwargs: {lr: 0.00001}
          - module_path: 'client.task_detector'
            kwargs: {lr: 0.0001}
          - module_path: 'client.film_generator'
            kwargs: {lr: 0.0001}
          - module_path: 'server_decoders_tails' # ModuleList/Dict of D_k, T_k
            kwargs: {lr: 0.0001}
        kwargs: # Default for other params if any (e.g. VIB bottleneck)
          lr: 0.0001

      criterion:
        key: 'WeightedSumLoss' # Use torchdistill's for combining
        kwargs:
          sub_terms:
            task_detector_loss:
              criterion:
                key: 'BCEWithLogitsLoss' # Or BCELoss if P_task is already sigmoid
              criterion_wrapper:
                key: 'SimpleLossWrapper'
                kwargs:
                  input: # P_task from student
                    is_from_teacher: False
                    module_path: '.' # Assuming student_model output dict has 'task_predictions'
                    io: 'task_predictions'
                  target: # Y_task from dataset
                    uses_label: True # Special handling in dataloader to provide Y_task
                    is_ground_truth: True # Custom flag for your dataloader/criterion
                    label_key: 'Y_task' # Key from dataset's target dictionary
              weight: 1.0 # Your lambda_task

            downstream_task_losses: # This needs a custom wrapper
              criterion:
                key: '!import_call' # Your src.losses.MultiTaskDownstreamLoss
                init:
                  kwargs:
                    # num_tasks, individual_loss_types (e.g., ['CrossEntropy', 'MSE'])
                    # task_weights (w_k, handled by active tasks)
              criterion_wrapper:
                key: '!import_call' # src.losses.MultiTaskCriterionWrapper
                init: # kwargs for the wrapper's init
                  # low_level_loss will be auto-injected by torchdistill
                  # Define how to get Output_k and Y_downstream_k
                  # And how to pass active_task_mask
                  # These are handled inside the wrapper's forward method
                  pass
              weight: 1.0 # Overall weight for sum of downstream losses
            
            rate_loss_z_film:
              criterion:
                key: '!import_call' # src.losses.VIBlossStage2 (similar to Stage1 but for z_film)
                init:
                  kwargs:
                    # num_pixels_placeholder, etc.
              criterion_wrapper:
                key: 'SimpleLossWrapper'
                kwargs:
                  input:
                    is_from_teacher: False
                    module_path: '.' # Assuming student_model output dict has 'z_film_likelihoods'
                    io: 'z_film_likelihoods'
                  target: # Dummy
                    uses_label: False
              weight: 0.01 # Your beta_prime
    ```

*   **`src/models.py` (for Stage 2)**:

    ```python
    import torch
    from torch import nn
    # Assuming these modules are defined in your project structure
    # from .client.stem import SharedStem
    # from .client.task_detector import TaskDetector
    # from .client.film_generator import FiLMGenerator
    # from .client.filmed_encoder import FiLMedEncoder # Now with FiLM active
    # from .server.decoders_tails import TaskSpecificDecoder, TaskSpecificTail # And heads
    # from .vib import add_quantization_noise
    from compressai.entropy_models import EntropyBottleneck

    # Placeholders for actual imports
    class SharedStem(nn.Module): # Placeholder
        def __init__(self, *args, **kwargs): super().__init__(); self.conv = nn.Conv2d(3, 64, 3, 1, 1)
        def forward(self, x): return self.conv(x)
    
    class TaskDetector(nn.Module): # Placeholder
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = nn.Linear(64, 10) # num_tasks=10
        def forward(self, x): return torch.sigmoid(self.fc(x.mean(dim=[2,3]))) # Simplified

    class FiLMGenerator(nn.Module): # Placeholder
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = nn.Linear(10, 2*128*2) # num_tasks=10, 2 layers, 128 channels each
        def forward(self, p_task):
            params = self.fc(p_task)
            # Simplified: actual reshaping and splitting needed
            return [(params[:, :128], params[:, 128:256]), (params[:, 256:384], params[:, 384:])]


    class FiLMedEncoder(nn.Module): # Placeholder
        def __init__(self, *args, film_bypass=False, **kwargs):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 128, 3, 1, 1)
            self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
            self.film_bypass = film_bypass
        def forward(self, x, film_params_list=None):
            x = self.conv1(x)
            if not self.film_bypass and film_params_list:
                 gamma1, beta1 = film_params_list[0]
                 if gamma1.ndim == 2: gamma1 = gamma1.unsqueeze(-1).unsqueeze(-1); beta1 = beta1.unsqueeze(-1).unsqueeze(-1)
                 x = gamma1 * x + beta1
            x = self.conv2(x)
            if not self.film_bypass and film_params_list and len(film_params_list) > 1:
                 gamma2, beta2 = film_params_list[1]
                 if gamma2.ndim == 2: gamma2 = gamma2.unsqueeze(-1).unsqueeze(-1); beta2 = beta2.unsqueeze(-1).unsqueeze(-1)
                 x = gamma2 * x + beta2
            return x

    class TaskSpecificDecoder(nn.Module): # Placeholder
        def __init__(self, *args, **kwargs): super().__init__(); self.conv = nn.Conv2d(128, 64, 3, 1, 1)
        def forward(self, x): return self.conv(x)

    class TaskSpecificTail(nn.Module): # Placeholder
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = nn.Linear(64, 100) # num_classes=100
        def forward(self, x): return self.fc(x.mean(dim=[2,3]))

    def add_quantization_noise(z, training=True): # Placeholder
        if training:
            noise = torch.empty_like(z).uniform_(-0.5, 0.5)
            return z + noise
        return torch.round(z)

    class MantisClientStage2(nn.Module):
        def __init__(self, stem_params, task_detector_params, film_gen_params, filmed_encoder_params):
            super().__init__()
            self.stem = SharedStem(**stem_params)
            self.task_detector = TaskDetector(**task_detector_params)
            self.film_generator = FiLMGenerator(**film_gen_params)
            self.filmed_encoder = FiLMedEncoder(**filmed_encoder_params, film_bypass=False)
        
        def load_stem_weights(self, checkpoint_path):
            # Logic to load only stem weights
            pass
        
        def load_encoder_conv_weights(self, checkpoint_path):
            # Logic to load only conv weights of filmed_encoder from Stage 1
            pass

        def forward(self, x):
            f_stem = self.stem(x)
            p_task = self.task_detector(f_stem)
            film_params_list = self.film_generator(p_task)
            z_film = self.filmed_encoder(f_stem, film_params_list)
            return z_film, p_task

    class MantisStage2(nn.Module):
        def __init__(self, client_params, num_tasks, decoder_params_list, tail_params_list, vib_channels):
            super().__init__()
            self.client = MantisClientStage2(**client_params)
            self.vib_bottleneck_film = EntropyBottleneck(vib_channels) # For P(z_film)

            self.server_decoders = nn.ModuleList()
            self.server_tails = nn.ModuleList()
            for i in range(num_tasks):
                self.server_decoders.append(TaskSpecificDecoder(**decoder_params_list[i]))
                self.server_tails.append(TaskSpecificTail(**tail_params_list[i]))
        
        def forward(self, x, active_task_mask=None): # active_task_mask (B, num_tasks) from dataset
            z_film_raw, p_task = self.client(x)
            
            # Add quantization noise
            z_film_quant_sim = add_quantization_noise(z_film_raw, self.training)
            z_film_hat, z_film_likelihoods = self.vib_bottleneck_film(z_film_quant_sim)

            downstream_outputs = [] # List of B tensors, or a dict
            # active_task_mask might be part of the targets dictionary from the dataloader
            # and not passed directly to forward. If so, it's retrieved in the loss wrapper.
            # For direct model usage, it might be passed as an argument.

            for k in range(len(self.server_tails)):
                # If active_task_mask is provided and used for conditional execution:
                if active_task_mask is None or active_task_mask[:, k].any(): 
                    task_k_features = self.server_decoders[k](z_film_hat) 
                    task_k_output = self.server_tails[k](task_k_features)
                    downstream_outputs.append(task_k_output)
                else:
                    # To ensure list length consistency for loss calculation
                    # It's often better to compute all and let loss function handle masking
                    task_k_features = self.server_decoders[k](z_film_hat) 
                    task_k_output = self.server_tails[k](task_k_features)
                    # Or append a placeholder that the loss function knows to ignore
                    downstream_outputs.append(task_k_output) # Or None, if loss handles it


            return {
                "task_predictions": p_task, # For task_detector_loss
                "downstream_outputs": downstream_outputs, # For downstream_task_losses
                "z_film_likelihoods": {"z_film": z_film_likelihoods} # For rate_loss_z_film
            }
    ```

*   **`src/losses.py` (Custom `MultiTaskDownstreamLoss` and Wrapper for Stage 2)**:

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchdistill.losses.registry import register_loss_wrapper # Ensure this import path is correct

    # In src/losses.py
    class MultiTaskDownstreamLoss(nn.Module):
        def __init__(self, num_tasks, task_loss_fns_configs):
            super().__init__()
            self.num_tasks = num_tasks
            self.task_loss_fns = nn.ModuleList()
            for config in task_loss_fns_configs: # e.g. [{'type': 'CrossEntropyLoss', 'params': {}}, ...]
                if config['type'] == 'CrossEntropyLoss':
                    self.task_loss_fns.append(nn.CrossEntropyLoss(**config.get('params', {})))
                elif config['type'] == 'MSELoss':
                    self.task_loss_fns.append(nn.MSELoss(**config.get('params', {})))
                # Add other loss types as needed
                else:
                    raise ValueError(f"Unsupported loss type: {config['type']}")

        def forward(self, downstream_outputs_list, y_downstream_list, active_task_mask):
            # downstream_outputs_list: list of model outputs for each task [Output_1, Output_2, ...]
            # y_downstream_list: list of ground truth labels for each task [Y_downstream_1, Y_downstream_2, ...]
            # active_task_mask: (B, num_tasks) boolean tensor indicating active tasks for each sample
            
            total_loss = torch.tensor(0.0, device=active_task_mask.device) # Ensure loss is on correct device
            # batch_size = active_task_mask.shape[0] # Not strictly needed if averaging per task loss
            num_active_task_losses_summed = 0

            for k in range(self.num_tasks):
                if downstream_outputs_list[k] is None: # Task not processed or output is None
                    continue

                # Select samples for which task k is active
                active_samples_mask_k = active_task_mask[:, k].bool() # Ensure boolean
                if not active_samples_mask_k.any():
                    continue
                    
                output_k = downstream_outputs_list[k][active_samples_mask_k]
                target_k = y_downstream_list[k][active_samples_mask_k]
                
                if output_k.nelement() > 0 and target_k.nelement() > 0:
                     task_k_loss = self.task_loss_fns[k](output_k, target_k)
                     total_loss += task_k_loss 
                     num_active_task_losses_summed +=1
            
            # Average over the number of tasks that contributed to the loss in this batch
            if num_active_task_losses_summed > 0:
                return total_loss / num_active_task_losses_summed
            return total_loss # Or torch.tensor(0.0) if no tasks were active


    # In src/losses.py (or a new file like src/torchdistill_wrappers.py)
    @register_loss_wrapper 
    class MultiTaskCriterionWrapper(nn.Module):
        def __init__(self, criterion, **kwargs): # criterion is MultiTaskDownstreamLoss instance
            super().__init__()
            self.criterion = criterion 
            # kwargs might contain info about how to extract data, if not hardcoded
        
        def forward(self, student_io_dict, teacher_io_dict, targets, supp_dict=None, **kwargs):
            # Assuming student_io_dict['.']['output'] is the dict from MantisStage2.forward
            # This structure depends on your forward_hook setup in YAML
            student_model_output = student_io_dict['.']['output'] # Example path
            downstream_outputs_list = student_model_output['downstream_outputs']
            
            # 'targets' from dataloader should be a dict: {'Y_task': ..., 'Y_downstream': [labels_task1, ...]}
            # or however your custom dataset structures it.
            y_downstream_list = targets['Y_downstream'] 
            active_task_mask = targets['Y_task'] # (B, num_tasks)
            
            return self.criterion(downstream_outputs_list, y_downstream_list, active_task_mask)

    # VIBlossStage2 would be similar to VIBlossStage1, just ensure input io key matches 'z_film_likelihoods'
    class VIBlossStage2(VIBlossStage1): # Inherits from Stage 1, can override if needed
        def forward(self, student_io_dict, target_dummy=None, **kwargs):
            if isinstance(student_io_dict, dict) and "z_film" in student_io_dict:
                 z_likelihoods_dict = {"z": student_io_dict["z_film"]} # Adapt key
            elif isinstance(student_io_dict, torch.Tensor):
                 z_likelihoods_dict = {"z": student_io_dict}
            else:
                raise ValueError("VIBlossStage2 expects z_film_likelihoods as a dict {'z_film': tensor} or just the tensor.")
            return super().forward(z_likelihoods_dict, target_dummy, **kwargs)

    ```

### 5. Datasets and Tasks

*   **Library**: PyTorch (`torch.utils.data.Dataset`), `torchvision.datasets`
*   **Implementation**: In `src/datasets.py`:
    *   `ImageNetSubgroupsDataset(ImageFolder)`: Subclass `ImageFolder`. Override `__init__` to filter samples based on your class subgroup definitions. Override `__getitem__` to return `(image, {'Y_task': multi_hot_task_vector, 'Y_downstream': [class_label_for_task0, None, class_label_for_task2, ...]})`. The `Y_downstream` would contain the original ImageNet label, but perhaps mapped to a task-specific range if tasks are mutually exclusive classifications. For overlapping tasks like "Various Animals" and "Domestic Animals", the `Y_downstream` might be the same original label, and the loss calculation handles which head is active.
    *   For multiple datasets (e.g., CIFAR-100 + Food-101), use `torch.utils.data.ConcatDataset` or a custom dataset that samples from them. The target structure would need to adapt.

### 6. Key Hyperparameters

These will be managed in your `configs/*.yaml` files, passed as arguments to your models or loss functions.

### 7. Evaluation Metrics

*   **Implementation**: In `scripts/evaluate.py`:
    *   Load your trained Stage 2 MANTiS model.
    *   Iterate through your test dataset.
    *   For each sample:
        1.  **Client-side**: `z_film_raw`, `p_task` = `client(x)`
        2.  **Simulate quantization**: `z_film_q = torch.round(z_film_raw)`
        3.  **Entropy code** `z_film_q` using the VIB bottleneck: `strings = client.vib_bottleneck_film.compress(z_film_q)`. Calculate BPP from `len(strings)`.
        4.  **Server-side**: `z_film_hat = client.vib_bottleneck_film.decompress(strings, shape)`. Then for each active task \\(k\\): `Output_k = server_tails[k](server_decoders[k](z_film_hat))`.
        5.  Calculate accuracy for active task \\(k\\).
    *   Aggregate BPP and accuracies.

### 8. Points Requiring Further Detail

These are mostly conceptual and relate to how you structure the experiments, which the libraries can support once the logic is defined.

*   **MANTiS (No FiLM) Ablation**:
    *   In `src/models.py`, create `MantisNoFiLMStage2(MantisStage2)` where `self.client.film_generator` is an identity (or produces fixed \\(\gamma=1, \beta=0\\)) and `self.client.filmed_encoder` always uses these fixed parameters.
    *   Train this with the Stage 2 config. If generic decoder from Stage 1: Server side would use one `GenericDecoder` and \\(N\\) `TaskSpecificTails`. If task-specific decoders: \\(N\\) `TaskSpecificDecoder` and \\(N\\) `TaskSpecificTails`.

*   **Oracle Task Selector Baseline**:
    *   In `src/models.py`, create `MantisOracleClientStage2(MantisClientStage2)` where `forward` takes `Y_task_ground_truth` as input.
    *   `p_task = Y_task_ground_truth` (or pass it directly to `film_generator`).
    *   Train this with the Stage 2 config, but the `task_detector_loss` weight in YAML would be 0, and `TaskDetector` wouldn't be updated or used for generating `p_task` fed to FiLM generator.

### General Workflow with `torchdistill`

1.  **Implement Modules**: Create your Python modules in `src/`.
2.  **Register (if needed)**: If you create custom losses, models, dataset wrappers not already known to `torchdistill` or PyTorch/Torchvision, use the `@register_*` decorators from `torchdistill`.
3.  **Configure YAML**:
    *   Use `!import_call` to instantiate your custom modules from `src/`.
        *   Example: `key: '!import_call {module: "src.client.stem", class: "SharedStem", init: {kwargs: {num_input_channels: 3, ...}}}'`
    *   Set up `forward_hook` for student and teacher models to extract intermediate features needed by your distillation/task losses.
    *   Configure `criterion` with `WeightedSumLoss` and define all sub-terms.
4.  **Train**: Use a generic training script from `torchdistill/examples` (like `image_classification.py`, potentially adapting it if your input/output structure is very custom) and point it to your YAML config.

    ```bash
    python -m torchdistill.entry.train --config configs/stage1_vib_hd.yaml
    python -m torchdistill.entry.train --config configs/stage2_task_aware.yaml
    ```
    (Note: `torchdistill.examples.image_classification` might have moved or been updated; the typical entry point is often `torchdistill.entry.train` or similar. Please check `torchdistill` documentation for the correct training script invocation.)


</rewritten_file>

