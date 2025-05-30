import torch
import torch.nn as nn
from client.stem import SharedStem
from client.task_detector import TaskDetector
from client.film_generator import FiLMGenerator
from client.filmed_encoder import FiLMedEncoder
from server.decoders_tails import GenericDecoder, TaskSpecificDecoder, TaskSpecificTail
from vib import VIBBottleneck


class MantisStage1Client(nn.Module):
    """
    MANTiS client for Stage 1: Generic VIB & Head Distillation training.
    
    Only contains stem and encoder (no task detector or FiLM generator).
    """
    
    def __init__(self, stem_channels=128, encoder_channels=256, num_encoder_blocks=3):
        super().__init__()
        
        self.stem = SharedStem(
            input_channels=3,
            output_channels=stem_channels,
            num_blocks=2
        )
        
        self.encoder = FiLMedEncoder(
            input_channels=stem_channels,
            output_channels=encoder_channels,
            num_blocks=num_encoder_blocks,
            film_bypass=True  # No FiLM in Stage 1
        )
        
    def forward(self, x):
        """
        Forward pass through Stage 1 client.
        
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            z: Latent representation (B, encoder_channels, H//8, W//8)
        """
        f_stem = self.stem(x)
        z = self.encoder(f_stem)  # FiLM is bypassed
        return z


class MantisStage1(nn.Module):
    """
    Complete MANTiS model for Stage 1 training.
    
    Includes client encoder, VIB bottleneck, and generic decoder for head distillation.
    """
    
    def __init__(self, client_params, decoder_params, vib_channels):
        super().__init__()
        
        self.client_encoder = MantisStage1Client(**client_params)
        self.vib_bottleneck = VIBBottleneck(vib_channels)
        self.generic_decoder = GenericDecoder(**decoder_params)
        
    def forward(self, x):
        """
        Forward pass for Stage 1 training.
        
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            Dictionary with outputs for loss computation
        """
        # Client-side encoding
        z_raw = self.client_encoder(x)
        
        # VIB bottleneck
        z_hat, z_likelihoods = self.vib_bottleneck(z_raw, training=self.training)
        
        # Server-side decoding for head distillation
        reconstructed_features = self.generic_decoder(z_hat)
        
        return {
            "g_s_output": reconstructed_features,  # For head distillation loss
            "z_likelihoods": {"z": z_likelihoods}  # For rate loss
        }


class MantisClientStage2(nn.Module):
    """
    MANTiS client for Stage 2: Task-aware training.
    
    Includes all client components: stem, task detector, FiLM generator, and FiLMed encoder.
    """
    
    def __init__(self, stem_params, task_detector_params, film_gen_params, filmed_encoder_params):
        super().__init__()
        
        self.stem = SharedStem(**stem_params)
        self.task_detector = TaskDetector(**task_detector_params)
        self.film_generator = FiLMGenerator(**film_gen_params)
        
        # Ensure film_bypass is set to False for Stage 2
        filmed_encoder_params = filmed_encoder_params.copy()
        filmed_encoder_params['film_bypass'] = False
        self.filmed_encoder = FiLMedEncoder(**filmed_encoder_params)
        
    def load_stage1_weights(self, stage1_checkpoint_path):
        """
        Load weights from Stage 1 model.
        
        Args:
            stage1_checkpoint_path: Path to Stage 1 checkpoint
        """
        checkpoint = torch.load(stage1_checkpoint_path, map_location='cpu')
        
        # Load stem weights
        stem_state_dict = {}
        encoder_state_dict = {}
        
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('client_encoder.stem.'):
                new_key = key.replace('client_encoder.stem.', '')
                stem_state_dict[new_key] = value
            elif key.startswith('client_encoder.encoder.'):
                new_key = key.replace('client_encoder.encoder.', '')
                encoder_state_dict[new_key] = value
                
        # Load with missing keys allowed (task detector and film generator are new)
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
        
        # Task-specific server components
        self.server_decoders = nn.ModuleList([
            TaskSpecificDecoder(**decoder_params) 
            for decoder_params in decoder_params_list
        ])
        
        self.server_tails = nn.ModuleList([
            TaskSpecificTail(**tail_params)
            for tail_params in tail_params_list
        ])
        
    def load_stage1_weights(self, stage1_checkpoint_path):
        """Load Stage 1 weights into the client."""
        self.client.load_stage1_weights(stage1_checkpoint_path)
        
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
        original_film_gen = self.client.film_generator
        channels_per_layer = original_film_gen.channels_per_layer
        
        self.client.film_generator = IdentityFiLMGenerator(channels_per_layer)


class IdentityFiLMGenerator(nn.Module):
    """FiLM generator that always produces identity parameters (gamma=1, beta=0)."""
    
    def __init__(self, channels_per_layer):
        super().__init__()
        self.channels_per_layer = channels_per_layer
        
    def forward(self, p_task):
        """Always return identity FiLM parameters regardless of input."""
        batch_size = p_task.shape[0]
        device = p_task.device
        
        output_params = []
        for num_channels in self.channels_per_layer:
            gamma = torch.ones(batch_size, num_channels, device=device)
            beta = torch.zeros(batch_size, num_channels, device=device)
            output_params.append((gamma, beta))
            
        return output_params


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