"""
This module defines various neural network architectures, primarily focusing on
splittable models for tasks like classification, which can incorporate compression
capabilities.

These networks typically utilize `CompressionModule` instances from
`model.modules.compressor`, backbones often sourced from TIMM (via
`model.modules.timm_models`), and specialized layers or components from
`model.modules` (like `SharedInputStem`, `TaskProbabilityModel`).

Key classes provided by the module include:
    - NetworkWithCompressionModule: Base class for networks that include a compression module.
    - SplittableClassifierWithCompressionModule: A classifier that can be split and includes compression.
    - SplittableNetworkWithSharedStem: A network that uses a shared input stem before branching.
    - FiLMedNetworkWithSharedStem: A network that uses FiLM conditioning with a shared input stem.
"""
import torch
from pytorch_grad_cam import XGradCAM
from sc2bench.models.layer import get_layer
from sc2bench.models.registry import get_compressai_model
from torch import nn
from torch import Tensor
from collections import OrderedDict

from torch.ao.quantization import HistogramObserver, QConfig, default_weight_observer
from torchdistill.common.constant import def_logger
from torchdistill.common.module_util import freeze_module_params
from torchdistill.models.registry import get_model, register_model_class, register_model_func

from misc.analyzers import AnalyzableModule
from model.modules.compressor import CompressionModule, FiLMedHFactorizedPriorCompressionModule
from model.modules.layers.transf import Tokenizer
from model.modules.module_registry import get_custom_compression_module
from model.modules.stem import SharedInputStem # Or from model.modules.analysis
from model.modules.task_predictors import TaskProbabilityModel

logger = def_logger.getChild(__name__)


@register_model_class
class MockTeacher(nn.Module):
    """
        Simplify learning regular image compression with torchdistill
    """

    def __init__(self):
        super(MockTeacher, self).__init__()
        self.no_op = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.no_op(x)


class NetworkWithCompressionModule(AnalyzableModule):
    def __init__(self,
                 compression_module, # Instantiated compression module
                 backbone, # Instantiated backbone
                 analysis_config=None): # analysis_config for THIS AnalyzableModule
        super().__init__(analysis_config.get("analyzers_config", []) if analysis_config else [])
        self.compression_module = compression_module
        self.backbone = backbone
        self.analyze_after_compress = analysis_config.get("analyze_after_compress", False) if analysis_config else False
        self.compressor_updated = False
        self.quantization_stage = None
        self.tokenizer = Tokenizer() # Generic tokenizer, or make it configurable

    def forward_compress(self): # Renamed for clarity vs. model.forward()
        return self.activated_analysis and self.compressor_updated and not self.training

    def forward(self, x, **kwargs): # Add **kwargs for flexibility (e.g. targets for training)
        raise NotImplementedError

    def update(self, force=False):
        logger.info("Updating NetworkWithCompressionModule (calling compression_module.update)")
        updated = self.compression_module.update(force=force)
        self.compressor_updated = updated # Or True if update was called
        return updated
@register_model_class
class SplittableClassifierWithCompressionModule(NetworkWithCompressionModule):
    """
        Regular image classifier networks with Compression Module (Regardless of transformer, cnn, mixer, e c.)

        reconstruction_layer is projection-like
    """
    def __init__(self,
                 compressor: CompressionModule,
                 reconstruction_layer,
                 backbone,
                 analysis_config=None,
                 tokenize_compression_output=False):
        super(SplittableClassifierWithCompressionModule, self).__init__(compressor,
                                                                        backbone,
                                                                        analysis_config)
        self.reconstruction_layer = reconstruction_layer
        self.compression_module = compressor
        self.backbone = backbone
        self.compression_module.g_s.final_layer = reconstruction_layer
        self.tokenizer = Tokenizer() if tokenize_compression_output else nn.Identity()

    def forward_compress(self):
        return self.activated_analysis and self.compressor_updated

    def forward(self, x):
        if self.forward_compress() and not self.training:
            compressed_obj = self.compression_module.compress(x)
            if self.activated_analysis:
                self.analyze(compressed_obj, img_shape=x.shape)
            h = self.compression_module.decompress(compressed_obj)
        else:
            h = self.compression_module(x)
            if isinstance(h, tuple):
                h = h[0]
        scores = self.backbone(self.tokenizer(h))
        return scores


@register_model_class
class SplittableClassifierWithCompressionModuleWithImagerRecon(NetworkWithCompressionModule):
    """
        Regular image classifier networks with Compression Module (Regardless of transformer, cnn, mixer, e c.)
    """
    def __init__(self,
                 compressor: CompressionModule,
                 reconstruction_layer,
                 backbone,
                 analysis_config=None,
                 **kwargs):
        super(SplittableClassifierWithCompressionModuleWithImagerRecon, self).__init__(compressor,
                                                                                       backbone,
                                                                                       analysis_config)
        self.reconstruction_layer = reconstruction_layer
        self.reconstruct_to_image = True
        self.compression_module = compressor
        self.backbone = backbone

    def forward_compress(self):
        return self.activated_analysis and self.compressor_updated

    def activate_image_compression(self):
        logger.info("Activated image compressionr")
        self.reconstruct_to_image = True

    def activate_feature_compression(self):
        logger.info("Activated feature compression, will skip reconstruction layer")
        self.reconstruct_to_image = False

    def forward(self, x):
        # note: output of likelihoods are registered by forward hooks applied by torchdistill
        if self.forward_compress() and not self.training:
            compressed_obj = self.compression_module.compress(x)
            if self.activated_analysis:
                self.analyze(compressed_obj, img_shape=x.shape)
            h = self.compression_module.decompress(compressed_obj)
        else:
            h = self.compression_module(x)
        if self.reconstruct_to_image:
            return self.reconstruction_layer(h)
        else:
            return self.backbone(h)


@register_model_class
class CompressionModelWithIdentityBackbone(NetworkWithCompressionModule):
    def __init__(self, analysis_config, compression_module_config):
        compression_model = get_compressai_model(compression_model_name=compression_module_config["name"],
                                                 **compression_module_config["params"]
                                                 )
        super(CompressionModelWithIdentityBackbone, self).__init__(
            backbone=nn.Identity(),
            analysis_config=analysis_config,
            compression_module=compression_model
        )

    def forward(self, x):
        return nn.Identity(x)


class SplittableObjectDetectorWithCompressionModule(NetworkWithCompressionModule):
    pass


class SplittableImageSegmentatorWithCompressionModule(NetworkWithCompressionModule):
    pass

@register_model_class
class SplittableNetworkWithSharedStem(NetworkWithCompressionModule):
    def __init__(self,
                 shared_stem_config,
                 task_probability_model_config,
                 compression_module_config,
                 backbone_config,
                 reconstruction_layer_for_backbone_config=None,
                 analysis_config_parent=None): # For NetworkWithCompressionModule's own analyzers

        self.shared_stem = SharedInputStem(**shared_stem_config["params"])
        stem_output_channels = self.shared_stem.get_output_channels()

        actual_task_prob_model_config_params = task_probability_model_config["params"].copy()
        actual_task_prob_model_config_params["input_channels_from_stem"] = stem_output_channels
        self.task_probability_model = TaskProbabilityModel(**actual_task_prob_model_config_params)
        
        actual_analysis_config_for_compressor = compression_module_config["params"]["analysis_config"]
        actual_analysis_config_for_compressor["params"]["input_channels_from_stem"] = stem_output_channels
        
        expected_film_cond_dim = self.task_probability_model.get_output_dim()
        if "film_cond_dim" not in actual_analysis_config_for_compressor["params"]:
            logger.info(f"Setting film_cond_dim in analysis_config from TaskProbabilityModel output_dim: {expected_film_cond_dim}")
            actual_analysis_config_for_compressor["params"]["film_cond_dim"] = expected_film_cond_dim
        elif actual_analysis_config_for_compressor["params"]["film_cond_dim"] != expected_film_cond_dim:
            raise ValueError(
                f"film_cond_dim in analysis_config ({actual_analysis_config_for_compressor['params']['film_cond_dim']}) "
                f"must match output_cond_signal_dim of task_probability_model ({expected_film_cond_dim})."
            )

        compressor = get_custom_compression_module(
            compression_module_config["name"],
            **compression_module_config["params"]
        )
        
        backbone = get_model(model_name=backbone_config["name"], **backbone_config["params"])

        super().__init__(compressor, backbone, analysis_config=analysis_config_parent)

        if reconstruction_layer_for_backbone_config and reconstruction_layer_for_backbone_config.get("name"):
            self.reconstruction_for_backbone = get_layer( # from sc2bench.models.layer
                reconstruction_layer_for_backbone_config["name"],
                **reconstruction_layer_for_backbone_config.get("params", {})
            )
            if hasattr(self.compression_module.g_s, 'final_layer'):
                logger.info(f"Setting final_layer of g_s to {reconstruction_layer_for_backbone_config['name']}")
                self.compression_module.g_s.final_layer = self.reconstruction_for_backbone
        else:
            self.reconstruction_for_backbone = nn.Identity()


    def forward(self, x, targets=None):
        stem_features = self.shared_stem(x)
        conditioning_signal = self.task_probability_model(stem_features)
        
        if self.forward_compress() and not self.training:
            compressed_obj = self.compression_module.compress(stem_features, conditioning_signal=conditioning_signal)
            if self.activated_analysis:
                self.analyze(compressed_obj, img_shape=x.shape) # Or stem_features.shape
            reconstructed_features = self.compression_module.decompress(compressed_obj)
        else: # Training or forward pass without explicit compress/decompress
            reconstructed_features = self.compression_module(stem_features, conditioning_signal=conditioning_signal)
            if isinstance(reconstructed_features, tuple): # e.g. (output, likelihoods)
                 reconstructed_features = reconstructed_features[0]
        
        if not (hasattr(self.compression_module.g_s, 'final_layer') and \
                isinstance(self.compression_module.g_s.final_layer, type(self.reconstruction_for_backbone))) \
                and not isinstance(self.reconstruction_for_backbone, nn.Identity):
            features_for_backbone = self.reconstruction_for_backbone(reconstructed_features)
        else: 
            features_for_backbone = reconstructed_features

        main_task_output = self.backbone(self.tokenizer(features_for_backbone))
        
        if self.training:
            return {"main_output": main_task_output,
                    "conditioning_signal_preview": conditioning_signal} 
        return main_task_output

@register_model_class
class FiLMedNetworkWithSharedStem(NetworkWithCompressionModule):
    def __init__(self,
                 shared_stem_config,
                 task_probability_model_config,
                 compression_module_config,
                 backbone_config,
                 reconstruction_layer_for_backbone_config=None,
                 analysis_config_parent=None): # For NetworkWithCompressionModule's analyzers

        # Instantiate components directly
        self.shared_stem = SharedInputStem(**shared_stem_config["params"])
        stem_output_channels = self.shared_stem.get_output_channels()

        actual_task_prob_model_config_params = task_probability_model_config["params"].copy()
        actual_task_prob_model_config_params["input_channels_from_stem"] = stem_output_channels
        # output_cond_signal_dim might be set dynamically based on dataset info later if not in YAML
        self.task_probability_model = TaskProbabilityModel(**actual_task_prob_model_config_params)
        
        expected_film_cond_dim = self.task_probability_model.get_output_dim()
        
        # Deep copy compressor config to modify it safely
        current_compression_module_config = yaml_util.load_yaml_file(yaml_util.dumps(compression_module_config)) # cheap deepcopy
        
        actual_analysis_config_for_g_a = current_compression_module_config["params"]["analysis_config"]
        actual_analysis_config_for_g_a["params"]["input_channels_from_stem"] = stem_output_channels
        
        if "film_cond_dim" not in actual_analysis_config_for_g_a["params"] or \
           actual_analysis_config_for_g_a["params"]["film_cond_dim"] is None:
            logger.info(f"Setting/Overriding film_cond_dim in analysis_config for g_a from TaskProbabilityModel output_dim: {expected_film_cond_dim}")
            actual_analysis_config_for_g_a["params"]["film_cond_dim"] = expected_film_cond_dim
        elif actual_analysis_config_for_g_a["params"]["film_cond_dim"] != expected_film_cond_dim:
            raise ValueError(
                f"film_cond_dim in analysis_config for g_a ({actual_analysis_config_for_g_a['params']['film_cond_dim']}) "
                f"must match output_cond_signal_dim of task_probability_model ({expected_film_cond_dim})."
            )

        compressor_instance = get_custom_compression_module(
            current_compression_module_config["name"],
            **current_compression_module_config["params"]
        )
        backbone_instance = get_model(model_name=backbone_config["name"], **backbone_config["params"])

        super().__init__(compressor_instance, backbone_instance, analysis_config=analysis_config_parent)
        
        # Reconstruction layer handling
        if reconstruction_layer_for_backbone_config and reconstruction_layer_for_backbone_config.get("name"):
            reconstruction_layer_instance = get_layer(
                reconstruction_layer_for_backbone_config["name"],
                **reconstruction_layer_for_backbone_config.get("params", {})
            )
            # If g_s has a 'final_layer' attribute, it means g_s itself can apply the final recon.
            # Otherwise, this network applies it after g_s.
            if hasattr(self.compression_module.g_s, 'final_layer'):
                logger.info(f"Setting final_layer of g_s to {reconstruction_layer_for_backbone_config['name']}")
                self.compression_module.g_s.final_layer = reconstruction_layer_instance
                self.reconstruction_for_backbone = nn.Identity() # g_s handles it
            else:
                self.reconstruction_for_backbone = reconstruction_layer_instance
        else:
            self.reconstruction_for_backbone = nn.Identity()


    def forward(self, x, targets=None): # `targets` might be used by some advanced scenarios, but typically not by model.forward
        stem_features = self.shared_stem(x)
        conditioning_signal = self.task_probability_model(stem_features)
        
        reconstructed_features_from_compressor_output = None
        if self.forward_compress(): # Inference with actual compression
            compressed_obj = self.compression_module.compress(stem_features, conditioning_signal=conditioning_signal)
            if self.activated_analysis:
                self.analyze(compressed_obj, img_shape=x.shape) 
            reconstructed_features_from_compressor_output = self.compression_module.decompress(compressed_obj)
        else: # Training or regular forward pass (quantization noise added if training)
            reconstructed_features_from_compressor_output = self.compression_module(
                stem_features, 
                conditioning_signal=conditioning_signal,
                return_likelihoods=self.training # Request likelihoods only if training (for BppLoss)
            )
        
        # Handle if compression_module returns (output, likelihoods)
        if isinstance(reconstructed_features_from_compressor_output, tuple) and \
           len(reconstructed_features_from_compressor_output) > 0 and \
           isinstance(reconstructed_features_from_compressor_output[0], torch.Tensor):
            # Assume first element is the feature, rest could be likelihoods etc.
            reconstructed_features = reconstructed_features_from_compressor_output[0]
            # likelihoods_y = reconstructed_features_from_compressor_output[1].get('y') if len(reconstructed_features_from_compressor_output) > 1 and isinstance(reconstructed_features_from_compressor_output[1], dict) else None
        else:
            reconstructed_features = reconstructed_features_from_compressor_output
            # likelihoods_y = None
            
        features_for_backbone = self.reconstruction_for_backbone(reconstructed_features)
        
        # Tokenizer might be part of the backbone's patch_embed or applied explicitly here
        # Assuming backbone expects tokenized input if it's a Transformer type without its own patch_embed.
        # If backbone is CNN based and skip_embed=True, it expects feature maps.
        # This needs to be consistent with backbone_config.
        # For ResNet with skip_embed=True, it expects feature maps.
        # For ViT types, they usually have their own patch_embed unless skip_embed means something different.
        # Let's assume self.tokenizer handles img->token if needed, or is Identity.
        # If backbone.patch_embed exists and is not Identity, it will handle tokenization.
        
        # If backbone is a typical timm model, it has its own forward.
        # If reconstruction_for_backbone output is C,H,W and backbone needs tokens, self.tokenizer is used.
        # If backbone needs C,H,W, self.tokenizer should be Identity.
        # For ResNet18 with skip_embed=True, it takes (B, C, H, W) where C is usually 64 for layer1.
        # So, self.tokenizer should be nn.Identity() in that case.
        # Making self.tokenizer part of the class, initialized based on backbone type or config.
        # For now, if features_for_backbone is 4D, and backbone is a typical timm model, it should work.
        # If backbone expects 3D (B, N, C) then tokenization is needed.
        
        # Let's assume `self.tokenizer` is nn.Identity() if the backbone's first layer (e.g. resnet.layer1) expects 4D tensor.
        # And it's a real Tokenizer if the backbone (e.g. ViT without patch_embed) expects 3D tensor.
        # For ResNet18 backbone with skip_embed=True, input to layer1 is 4D.
        # If self.tokenizer is Tokenizer(), it converts 4D to 3D. This depends on backbone specifics.
        # The provided YAML uses resnet18 with skip_embed=True, so its layer1 expects 4D. Tokenizer should be Identity.
        # For safety, we'll call a general self.tokenizer here. It should be nn.Identity for ResNet scenario.
        
        # Check dimensionality for tokenizer
        if features_for_backbone.dim() == 4 and self.backbone.__class__.__name__ not in ["ResNet", "ConvNeXt"]: # Add other CNNs
             # Assuming transformers take 3D input after tokenizer
            tokenized_features = self.tokenizer(features_for_backbone)
            main_task_output = self.backbone(tokenized_features)
        else: # CNNs usually take 4D
            main_task_output = self.backbone(features_for_backbone)
            
        # For loss calculation, the training script will expect a dictionary.
        # For GeneralizedCustomLoss, the paths in its `params` section will try to extract these.
        output_dict = {"main_output": main_task_output, 
                       "conditioning_signal_preview": conditioning_signal}
        
        # Include likelihoods for BppLoss during training if returned by compression_module
        # if self.training and likelihoods_y is not None:
        #     # This is tricky because BppLoss usually hooks the entropy_bottleneck directly.
        #     # If compression_module already returns likelihoods for the *entire* set (y, z etc),
        #     # then BppLoss could be configured to use this output.
        #     # However, standard BppLoss hooks EB.
        #     # For now, let's rely on GCL hooks for EB.
        #     pass

        return output_dict


@register_model_func
def splittable_network_with_compressor_with_shared_stem(
    shared_stem_config, 
    task_probability_model_config, 
    compression_module_config,
    backbone_config,
    reconstruction_layer_for_backbone_config=None, 
    analysis_config_parent=None, 
    network_type="FiLMedNetworkWithSharedStem" # Default to the new FiLMed one
):
    if network_type == "FiLMedNetworkWithSharedStem":
        network_class = FiLMedNetworkWithSharedStem
    # Add other types here if needed, e.g., "SplittableNetworkWithSharedStem" for a non-FiLMed version
    # elif network_type == "OriginalSplittableNetworkWithSharedStem":
    #     network_class = SplittableNetworkWithSharedStem # Assume this is defined elsewhere
    else:
        raise ValueError(f"Unknown network_type for splittable_network: {network_type}")

    # Dynamically adjust task_probability_model_config's output_cond_signal_dim
    # This ideally should happen after dataset is loaded and info is available.
    # For now, the YAML might need to provide a placeholder or the actual value.
    # If dataset info (num_task_chunks) is available here, we could override.
    # This is a bit tricky as model init usually precedes full dataset parsing.
    # The PhasedTrainingBox or main script might need to pass this info if it's truly dynamic.
    # For now, assume YAML value for output_cond_signal_dim is used.

    network = network_class(
        shared_stem_config=shared_stem_config,
        task_probability_model_config=task_probability_model_config, 
        compression_module_config=compression_module_config,
        backbone_config=backbone_config,
        reconstruction_layer_for_backbone_config=reconstruction_layer_for_backbone_config,
        analysis_config_parent=analysis_config_parent
    )
    return network
@register_model_func
def get_compression_model(compression_module_config):
    return get_custom_compression_module(compression_module_config["name"], **compression_module_config["params"])


@register_model_func
def get_mock_model(**kwargs):
    return MockTeacher()
