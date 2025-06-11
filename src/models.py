import torch
import torch.nn as nn
from client.stem import SharedStem
from client.task_detector import TaskDetector
from client.film_generator import FiLMGenerator
from client.filmed_encoder import FiLMedEncoder
from server.decoders_tails import GenericDecoderStage1, TaskSpecificDecoder, TaskSpecificTail, ResNetCompatibleTail, FrankenSplitDecoder
from vib import VIBBottleneck


class MantisStage1Client(nn.Module):
    """
    MANTiS client for Stage 1: Generic VIB & Head Distillation training.
    
    Only contains stem and encoder (no task detector or FiLM generator).
    Simplified FrankenSplit architecture: 32→64→48 channels.
    """
    
    def __init__(self, stem_channels=32, encoder_channels=48):
        super().__init__()
        
        self.stem = SharedStem(
            input_channels=3,
            output_channels=stem_channels
        )
        
        self.encoder = FiLMedEncoder(
            input_channels=stem_channels,
            latent_channels=encoder_channels
        )
        
        # Register identity FiLM parameters for Stage 1 (no task conditioning)
        self.register_buffer('identity_film_params', 
                           torch.cat([torch.ones(encoder_channels), torch.zeros(encoder_channels)]))
        
    def forward(self, x):
        """
        Forward pass through Stage 1 client.
        
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            z: Latent representation (B, encoder_channels, H//8, W//8)
        """
        f_stem = self.stem(x)
        
        # Create identity FiLM parameters for this batch (no task conditioning in Stage 1)
        batch_size = f_stem.size(0)
        identity_film_batch = self.identity_film_params.unsqueeze(0).expand(batch_size, -1)
        
        z = self.encoder(f_stem, identity_film_batch)
        return z


class MantisStage1(nn.Module):
    """
    Complete MANTiS model for Stage 1 training.
    Includes client encoder, VIB bottleneck, and FrankenSplit decoder for head distillation.
    """
    def __init__(self, client_params, decoder_params, vib_channels):
        super().__init__()
        self.client_encoder = MantisStage1Client(**client_params)
        self.vib_bottleneck = VIBBottleneck(vib_channels)
        self.generic_decoder = FrankenSplitDecoder(**decoder_params)

    def forward(self, x):
        z_raw = self.client_encoder(x)
        z_hat, z_likelihoods = self.vib_bottleneck(z_raw, training=self.training)
        reconstructed_features = self.generic_decoder(z_hat)
        
        return {
            "g_s_output": reconstructed_features,
            "z_likelihoods": {"z": z_likelihoods},
            # For debugging NaN:
            "debug_z_raw_mean": z_raw.detach().mean(),
            "debug_z_hat_mean": z_hat.detach().mean(),
            "debug_reconstructed_features_mean": reconstructed_features.detach().mean()
        }

class MantisClientStage2(nn.Module):
    """
    MANTiS client for Stage 2: Task-aware training.
    
    Includes all client components: stem, task detector, FiLM generator, and FiLMed encoder.
    Simplified FrankenSplit architecture: 32→64→48 channels.
    """
    
    def __init__(self, stem_params, task_detector_params, film_gen_params, filmed_encoder_params):
        super().__init__()
        
        self.stem = SharedStem(**stem_params)
        self.task_detector = TaskDetector(**task_detector_params)
        self.film_generator = FiLMGenerator(**film_gen_params)
        
        # Ensure film_bypass is set to False for Stage 2
        filmed_encoder_params = filmed_encoder_params.copy()
        self.filmed_encoder = FiLMedEncoder(**filmed_encoder_params)
        
    def load_stage1_weights(self, stage1_checkpoint_path):
        """
        Load weights from Stage 1 model.
        
        Args:
            stage1_checkpoint_path: Path to Stage 1 checkpoint
        """
        # Set weights_only=False as checkpoint contains argparse.Namespace
        checkpoint = torch.load(stage1_checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load stem weights
        stem_state_dict = {}
        encoder_state_dict = {}
        
        # Check if model_state_dict exists
        if 'model_state_dict' not in checkpoint:
            raise KeyError("Checkpoint does not contain 'model_state_dict'. Make sure it's a valid model checkpoint.")

        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('client_encoder.stem.'):
                new_key = key.replace('client_encoder.stem.', '')
                stem_state_dict[new_key] = value
            elif key.startswith('client_encoder.encoder.'):
                new_key = key.replace('client_encoder.encoder.', '')
                encoder_state_dict[new_key] = value
                
        # Load with missing keys allowed for encoder (FiLM params are new)
        # strict=True for stem as its architecture should match
        self.stem.load_state_dict(stem_state_dict, strict=True)
        self.filmed_encoder.load_state_dict(encoder_state_dict, strict=False)
        
    def forward(self, x):
        """
        Forward pass through Stage 2 client.
        
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            z_film: Task-conditioned latent representation
            p_task: Task probabilities
        """
        # Shared stem processing
        f_stem = self.stem(x)
        
        # Task detection
        p_task = self.task_detector(f_stem)
        
        # FiLM parameter generation
        film_params_list = self.film_generator(p_task)
        
        # FiLM-conditioned encoding
        z_film = self.filmed_encoder(f_stem, film_params_list)
        
        return z_film, p_task


class MantisStage2(nn.Module):
    """
    Complete MANTiS model for Stage 2 training.
    
    Includes client, VIB bottleneck, and multiple task-specific server components.
    """
    
    def __init__(self, client_params, num_tasks, decoder_params_list, tail_params_list, vib_channels):
        super().__init__()
        
        self.client = MantisClientStage2(**client_params)
        self.vib_bottleneck_film = VIBBottleneck(vib_channels)
        self.num_tasks = num_tasks
        
        # Task-specific server components - use FrankenSplitDecoder to ensure architectural match
        self.server_decoders = nn.ModuleList([
            FrankenSplitDecoder(**decoder_params) 
            for decoder_params in decoder_params_list
        ])
        
        # Use ResNetCompatibleTail for proper ResNet layer4 architecture
        self.server_tails = nn.ModuleList([
            ResNetCompatibleTail(input_channels=tail_params['input_channels'], 
                               num_classes=tail_params['num_classes'])
            for tail_params in tail_params_list
        ])
        
    def load_stage1_weights(self, checkpoint_path, rank=0):
        if rank == 0:
            print("\n--- Loading Stage‑1 CLIENT weights ---")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        s1_state = ckpt["model_state_dict"]
        s2_state = self.client.state_dict()
        new_state = {}
        loaded = 0

        # ⬇︎  accept BOTH the old‑style *and* the new‑style prefixes
        prefix_map = {
            # old runs (pre‑refactor)  →  Stage‑2
            "client_encoder.stem."          : "stem.",
            "client_encoder.encoder."       : "filmed_encoder.",
            # new runs (Stage‑1 wrapper)    →  Stage‑2
            "client.stem."                  : "stem.",
            "client.filmed_encoder."        : "filmed_encoder.",
        }

        for s1_k, s1_v in s1_state.items():
            for old_pref, new_pref in prefix_map.items():
                if s1_k.startswith(old_pref):
                    s2_k = s1_k.replace(old_pref, new_pref)
                    if s2_k in s2_state and s2_state[s2_k].shape == s1_v.shape:
                        new_state[s2_k] = s1_v
                        loaded += 1
                    break

        self.client.load_state_dict(new_state, strict=False)
        if rank == 0:
            print(f"  ✓ Loaded {loaded} tensors → client stem / encoder.")
            print("--- CLIENT weight transfer complete ---")

    def load_stage1_decoder_weights(self, checkpoint_path, rank=0):
        if rank == 0:
            print("\n--- Loading Stage‑1 DECODER weights ---")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        s1_state = ckpt["model_state_dict"]

        # ⬇︎  Stage‑1 wrapper saves the decoder under 'decoder.*'
        dec_state = {k.replace("decoder.", ""): v
                     for k, v in s1_state.items() if k.startswith("decoder.")}

        if not dec_state:
            if rank == 0:
                print("  ERROR: no 'decoder.*' tensors found – did Stage‑1 train finish?")
            return

        total = 0
        for idx, dec in enumerate(self.server_decoders):
            dec.load_state_dict(dec_state, strict=True)
            total += sum(p.numel() for p in dec_state.values())
            if rank == 0:
                print(f"  ✓ Task {idx}: decoder weights loaded.")

        if rank == 0:
            print(f"--- DECODER weight transfer complete ({total:,} params) ---")

    def load_teacher_tail_weights(self, task_definitions, rank=0):
        if rank == 0:
            print("\n--- Loading pretrained TAIL weights (ResNet50) ---")
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            teacher_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            teacher_fc_weight = teacher_model.fc.weight.data
            teacher_fc_bias = teacher_model.fc.bias.data
            if rank == 0:
                print("  Successfully loaded pretrained ResNet50 model and extracted FC layer weights.")
        except ImportError:
            if rank == 0:
                print("  Warning: torchvision.models.resnet50 not found. Tails will be randomly initialized.")
            return

        for i, tail in enumerate(self.server_tails):
            if isinstance(tail, ResNetCompatibleTail):
                task_name = list(task_definitions.keys())[i]
                class_indices = task_definitions[task_name]
                # Call the correct method 'load_task_weights'
                tail.load_task_weights(class_indices, teacher_fc_weight, teacher_fc_bias)
                if rank == 0:
                    print(f"  ✓ For task '{task_name}', loaded weights for {len(class_indices)} classes into ResNet tail's FC layer.")
            elif rank == 0:
                print(f"  - Skipping tail {i} as it is not a ResNetCompatibleTail.")
        if rank == 0:
            print("--- TAIL weight transfer complete ---")

    def forward(self, x, active_task_mask=None):
        """
        Forward pass for Stage 2 training.
        
        Args:
            x: Input image (B, 3, H, W)
            active_task_mask: Optional mask indicating active tasks (B, num_tasks)
            
        Returns:
            Dictionary with outputs for loss computation
        """
        # Client-side processing
        z_film_raw, p_task = self.client(x)
        
        # VIB bottleneck for task-conditioned representation
        z_film_hat, z_film_likelihoods = self.vib_bottleneck_film(z_film_raw, training=self.training)
        
        # Server-side processing for all tasks
        downstream_outputs = []
        
        for k in range(self.num_tasks):
            # Decode and classify for task k
            task_k_features = self.server_decoders[k](z_film_hat)
            task_k_output = self.server_tails[k](task_k_features)
            downstream_outputs.append(task_k_output)
            
        return {
            "task_predictions": p_task,  # For task detector loss
            "downstream_outputs": downstream_outputs,  # For downstream task losses
            "z_film_likelihoods": {"z_film": z_film_likelihoods}  # For rate loss
        }


class MantisNoFiLMStage2(MantisStage2):
    """
    Ablation model: MANTiS without FiLM conditioning.
    
    Uses fixed identity FiLM parameters (gamma=1, beta=0).
    """
    
    def __init__(self, client_params, num_tasks, decoder_params_list, tail_params_list, vib_channels):
        super().__init__(client_params, num_tasks, decoder_params_list, tail_params_list, vib_channels)
        
        # Override client's film generator to produce identity parameters
        self.client.film_generator = IdentityFiLMGenerator(output_channels=48)


class IdentityFiLMGenerator(nn.Module):
    """FiLM generator that always produces identity parameters (gamma=1, beta=0)."""
    
    def __init__(self, output_channels=48):
        super().__init__()
        self.output_channels = output_channels
        
    def forward(self, p_task):
        """Always return identity FiLM parameters regardless of input."""
        batch_size = p_task.shape[0]
        device = p_task.device
        
        gamma = torch.ones(batch_size, self.output_channels, device=device)
        beta = torch.zeros(batch_size, self.output_channels, device=device)
        
        return [(gamma, beta)]


class MantisOracleStage2(MantisStage2):
    """
    Oracle baseline: MANTiS with ground truth task labels.
    
    Uses true task labels instead of predicted ones for FiLM generation.
    """
    
    def forward(self, x, y_task_ground_truth):
        """
        Forward pass with oracle task information.
        
        Args:
            x: Input image (B, 3, H, W)
            y_task_ground_truth: Ground truth task labels (B, num_tasks)
            
        Returns:
            Dictionary with outputs for loss computation
        """
        # Use ground truth task labels instead of predictions
        f_stem = self.client.stem(x)
        p_task = y_task_ground_truth.float()  # Use ground truth as "predictions"
        
        # Generate FiLM parameters from ground truth
        film_params_list = self.client.film_generator(p_task)
        z_film_raw = self.client.filmed_encoder(f_stem, film_params_list)
        
        # Rest is the same as regular Stage 2
        z_film_hat, z_film_likelihoods = self.vib_bottleneck_film(z_film_raw, training=self.training)
        
        downstream_outputs = []
        for k in range(self.num_tasks):
            task_k_features = self.server_decoders[k](z_film_hat)
            task_k_output = self.server_tails[k](task_k_features)
            downstream_outputs.append(task_k_output)
            
        return {
            "task_predictions": p_task,  # Use ground truth
            "downstream_outputs": downstream_outputs,
            "z_film_likelihoods": {"z_film": z_film_likelihoods}
        }