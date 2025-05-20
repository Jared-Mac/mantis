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
    """

    """

    def __init__(self,
                 compression_module: CompressionModule,
                 backbone,
                 analysis_config=None):
        if analysis_config is None:
            analysis_config = dict()
        super(NetworkWithCompressionModule, self).__init__(analysis_config.get("analyzers_config", list()))
        self.compression_module = compression_module
        self.backbone = backbone
        self.analyze_after_compress = analysis_config.get("analyze_after_compress", False)
        self.compressor_updated = False
        self.quantization_stage = None

    def activate_analysis(self):
        self.activated_analysis = True
        logger.info("Activated Analyzing Compression Module")

    def deactivate_analysis(self):
        self.activated_analysis = False
        logger.info("Deactivated Analyzing Compression Module")

    def forward(self, x):
        raise NotImplementedError

    def update(self, force=False):
        updated = self.compression_module.update(force=force)
        self.compressor_updated = updated
        self.compressor_updated = True
        return updated

    def compress(self, obj):
        return self.compression_module.compress(obj)

    def decompress(self, compressed_obj):
        return self.compression_module.decompress(compressed_obj)

    def load_state_dict(self, state_dict, **kwargs):
        compression_module_state_dict = OrderedDict()
        for key in list(state_dict.keys()):
            if key.startswith('compression_module.'):
                compression_module_state_dict[key.replace('compression_module.', '', 1)] = state_dict[key]

        self.compression_module.load_state_dict(compression_module_state_dict)
        super().load_state_dict(state_dict, strict=False)

    def get_quant_device(self):
        return self.compression_module.quantization_config.get("quant_device")

    def head_quantized(self):
        return self.quantization_stage == 'ready'

    def prepare_quantization(self):
        # removed due to scope
        pass

    def apply_quantization(self):
        if self.quantization_stage == 'ready':
            logger.info("Already applied quantization")
            return
        self.compression_module.apply_quantization()
        self.quantization_stage = 'ready'
        logger.info("Applied quantization to head")

    def quantize_entropy_bottleneck(self):
        self.compression_module.entropy_bottleneck.to("cpu")


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
    """
        removed due to scope
    """
    pass


class SplittableImageSegmentatorWithCompressionModule(NetworkWithCompressionModule):
    """
        removed due to scope
    """
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
        
        # 1. Instantiate Shared Stem
        self.shared_stem = SharedInputStem(**shared_stem_config["params"])
        stem_output_channels = self.shared_stem.get_output_channels()

        # 2. Instantiate Task Probability Model
        # Ensure its input_channels_from_stem matches the shared_stem's output
        actual_task_prob_model_config_params = task_probability_model_config["params"].copy()
        actual_task_prob_model_config_params["input_channels_from_stem"] = stem_output_channels
        self.task_probability_model = TaskProbabilityModel(**actual_task_prob_model_config_params)
        
        # 3. Prepare analysis_config for the compression_module
        #    It needs input_channels_from_stem and film_cond_dim
        actual_analysis_config_for_compressor = compression_module_config["params"]["analysis_config"]
        actual_analysis_config_for_compressor["params"]["input_channels_from_stem"] = stem_output_channels
        
        # Ensure film_cond_dim in analysis_config matches output of task_probability_model
        expected_film_cond_dim = self.task_probability_model.get_output_dim()
        if "film_cond_dim" not in actual_analysis_config_for_compressor["params"]:
            logger.info(f"Setting film_cond_dim in analysis_config from TaskProbabilityModel output_dim: {expected_film_cond_dim}")
            actual_analysis_config_for_compressor["params"]["film_cond_dim"] = expected_film_cond_dim
        elif actual_analysis_config_for_compressor["params"]["film_cond_dim"] != expected_film_cond_dim:
            raise ValueError(
                f"film_cond_dim in analysis_config ({actual_analysis_config_for_compressor['params']['film_cond_dim']}) "
                f"must match output_cond_signal_dim of task_probability_model ({expected_film_cond_dim})."
            )

        # 4. Instantiate Compression Module (which includes the FiLMed g_a)
        compressor = get_custom_compression_module(
            compression_module_config["name"],
            **compression_module_config["params"] 
        )
        
        # 5. Instantiate Backbone for the main task
        backbone = get_model(model_name=backbone_config["name"], **backbone_config["params"])

        super().__init__(compressor, backbone, analysis_config=analysis_config_parent)

        # 6. Optional: layer to adapt g_s output to backbone input
        if reconstruction_layer_for_backbone_config and reconstruction_layer_for_backbone_config.get("name"):
            self.reconstruction_for_backbone = get_layer( # from sc2bench.models.layer
                reconstruction_layer_for_backbone_config["name"],
                **reconstruction_layer_for_backbone_config.get("params", {})
            )
            # If this layer should be part of g_s (common pattern)
            if hasattr(self.compression_module.g_s, 'final_layer'):
                logger.info(f"Setting final_layer of g_s to {reconstruction_layer_for_backbone_config['name']}")
                self.compression_module.g_s.final_layer = self.reconstruction_for_backbone
        else:
            self.reconstruction_for_backbone = nn.Identity()


    def forward(self, x, targets=None): # targets might be used by main loss or task_prob_model loss
        # Stage 1: Shared Stem
        stem_features = self.shared_stem(x)

        # Stage 2: Task Probability Model -> Generates conditioning_signal
        # The TaskProbabilityModel might also be trained with 'targets' if it's a classifier itself.
        # For now, we assume it's unsupervised or targets are handled elsewhere.
        conditioning_signal = self.task_probability_model(stem_features) 
        
        # Stage 3: Compression Module (uses stem_features and conditioning_signal)
        if self.forward_compress() and not self.training:
            # Inference path with actual compression/decompression
            compressed_obj = self.compression_module.compress(stem_features, conditioning_signal=conditioning_signal)
            if self.activated_analysis:
                self.analyze(compressed_obj, img_shape=x.shape) # Or stem_features.shape
            reconstructed_features = self.compression_module.decompress(compressed_obj)
        else: # Training or forward pass without explicit compress/decompress
            reconstructed_features = self.compression_module(stem_features, conditioning_signal=conditioning_signal)
            if isinstance(reconstructed_features, tuple): # e.g. (output, likelihoods)
                 reconstructed_features = reconstructed_features[0]
        
        # Stage 4: Adapt features for backbone (if g_s.final_layer isn't already doing it)
        # This logic might be redundant if g_s.final_layer is set correctly.
        if not (hasattr(self.compression_module.g_s, 'final_layer') and \
                isinstance(self.compression_module.g_s.final_layer, type(self.reconstruction_for_backbone))) \
                and not isinstance(self.reconstruction_for_backbone, nn.Identity):
            features_for_backbone = self.reconstruction_for_backbone(reconstructed_features)
        else: # Assume g_s.final_layer (if it exists and is the recon layer) has already adapted it
            features_for_backbone = reconstructed_features

        # Stage 5: Main Task Backbone
        # Ensure tokenizer is used if backbone expects sequence data
        main_task_output = self.backbone(self.tokenizer(features_for_backbone)) 
        
        # For training, you might want to return a dictionary for multiple losses:
        # one for the main_task_output, another for the task_probability_model (if it has its own loss).
        if self.training:
            # Example: if task_probability_model's output (conditioning_signal) is also supervised
            # For now, just returning the main output and the signal itself.
            # The actual "task probability" might be an intermediate output of TaskProbabilityModel
            # if conditioning_signal is a further processed version.
            return {"main_output": main_task_output, 
                    "conditioning_signal_preview": conditioning_signal} # Or actual task probabilities
        return main_task_output

@register_model_class 
class FiLMedNetworkWithSharedStem(NetworkWithCompressionModule):
    def __init__(self,
                 shared_stem_config,
                 task_probability_model_config,
                 compression_module_config,     
                 backbone_config,
                 reconstruction_layer_for_backbone_config=None,
                 analysis_config_parent=None):
        
        # Step A: Instantiate components needed to determine parameters for compressor/backbone
        # Create them as local variables first if their attributes are needed before super().__init__
        _shared_stem_temp = SharedInputStem(**shared_stem_config["params"])
        stem_output_channels = _shared_stem_temp.get_output_channels()

        actual_task_prob_model_config_params = task_probability_model_config["params"].copy()
        actual_task_prob_model_config_params["input_channels_from_stem"] = stem_output_channels
        _task_probability_model_temp = TaskProbabilityModel(**actual_task_prob_model_config_params)
        expected_film_cond_dim = _task_probability_model_temp.get_output_dim()
        
        # Step B: Prepare full analysis_config for the compression_module's g_a
        # (This g_a is TaskConditionedFiLMedAnalysisNetwork)
        actual_analysis_config_for_g_a = compression_module_config["params"]["analysis_config"]
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

        # Step C: Instantiate compressor and backbone (as local variables)
        # These are needed for the super().__init__() call of NetworkWithCompressionModule
        compressor_instance = get_custom_compression_module(
            compression_module_config["name"],
            **compression_module_config["params"] 
        )
        backbone_instance = get_model(model_name=backbone_config["name"], **backbone_config["params"])

        # Step D: Call super().__init__() for NetworkWithCompressionModule FIRST.
        # This will call AnalyzableModule.__init__(), which calls nn.Module.__init__().
        # Pass the already instantiated compressor and backbone.
        super().__init__(compressor_instance, backbone_instance, analysis_config=analysis_config_parent)

        # Step E: Now it's safe to assign other nn.Module attributes to self.
        # Assign the temporarily created modules to self.
        self.shared_stem = _shared_stem_temp
        self.task_probability_model = _task_probability_model_temp
        
        # Handle reconstruction_layer_for_backbone_config
        if reconstruction_layer_for_backbone_config and reconstruction_layer_for_backbone_config.get("name"):
            # This get_layer returns an nn.Module instance
            reconstruction_layer_instance = get_layer(
                reconstruction_layer_for_backbone_config["name"],
                **reconstruction_layer_for_backbone_config.get("params", {})
            )
            # self.compression_module is available now (set by super().__init__)
            if hasattr(self.compression_module.g_s, 'final_layer'):
                logger.info(f"Setting final_layer of g_s to {reconstruction_layer_for_backbone_config['name']}")
                self.compression_module.g_s.final_layer = reconstruction_layer_instance
                # If it's part of g_s, self.reconstruction_for_backbone can be Identity or not used in forward
                self.reconstruction_for_backbone = nn.Identity() 
            else: 
                # If g_s doesn't have a final_layer attribute, assign it here to be applied in forward
                self.reconstruction_for_backbone = reconstruction_layer_instance
        else:
            self.reconstruction_for_backbone = nn.Identity()

    def forward(self, x, targets=None):
        # ... (rest of your forward method as previously defined)
        stem_features = self.shared_stem(x)
        conditioning_signal = self.task_probability_model(stem_features) 
        
        reconstructed_features_from_compressor_output = self.compression_module(stem_features, conditioning_signal=conditioning_signal)
        
        # Handle if compressor returns tuple (features, likelihoods)
        if isinstance(reconstructed_features_from_compressor_output, tuple) and \
           len(reconstructed_features_from_compressor_output) > 0 and \
           isinstance(reconstructed_features_from_compressor_output[0], torch.Tensor):
            reconstructed_features = reconstructed_features_from_compressor_output[0]
        else:
            reconstructed_features = reconstructed_features_from_compressor_output
           
        features_for_backbone = reconstructed_features
        # Apply self.reconstruction_for_backbone only if it's a real module and not already part of g_s
        if not isinstance(self.reconstruction_for_backbone, nn.Identity) and \
           not (hasattr(self.compression_module.g_s, 'final_layer') and \
                isinstance(self.compression_module.g_s.final_layer, type(self.reconstruction_for_backbone))):
            features_for_backbone = self.reconstruction_for_backbone(reconstructed_features)
            
        main_task_output = self.backbone(self.tokenizer(features_for_backbone)) 
           
        if self.training:
            return {"main_output": main_task_output, 
                    "conditioning_signal_preview": conditioning_signal} 
        return main_task_output
@register_model_func 
def splittable_network_with_compressor_with_shared_stem(
    shared_stem_config,
    task_probability_model_config, 
    compression_module_config,
    backbone_config,               
    reconstruction_layer_for_backbone_config=None,
    analysis_config_parent=None,    
    network_type="FiLMedNetworkWithSharedStem" # <<< UPDATED DEFAULT CLASS NAME
):
    network = get_model( 
        model_name=network_type, # Uses the (potentially overridden by YAML) network_type
        shared_stem_config=shared_stem_config,
        task_probability_model_config=task_probability_model_config,
        compression_module_config=compression_module_config,
        backbone_config=backbone_config, 
        reconstruction_layer_for_backbone_config=reconstruction_layer_for_backbone_config,
        analysis_config_parent=analysis_config_parent 
    )
    return network
@register_model_func
def splittable_network_with_compressor_with_shared_stem(
    shared_stem_config,                       # New
    task_probability_model_config,          # New (for later)
    compression_module_config,
    backbone_config,
    reconstruction_layer_for_backbone_config=None, # New/Renamed
    analysis_config_parent=None, # This is for the parent NetworkWithCompressionModule
    network_type="FiLMedHFactorizedPriorCompressionModule" # Use your new class
):
    # The actual analysis_config (for g_a) is inside compression_module_config
    # We will let FrankensplitNetworkWithSharedStem handle setting input_channels_from_stem
    
    # This will call FrankensplitNetworkWithSharedStem.__init__ via the registry
    network = get_model( 
        model_name=network_type, # Should be "FrankensplitNetworkWithSharedStem"
        shared_stem_config=shared_stem_config,
        task_probability_model_config=task_probability_model_config, # Now correctly passed
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
