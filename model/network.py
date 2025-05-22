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
    """
    Base class for neural networks that incorporate a `CompressionModule`.

    This class provides a foundational structure for models that combine a main
    task-specific backbone with a compression/decompression pipeline. It handles
    the integration of a `CompressionModule` and offers methods for its
    operation, such as updating the entropy bottleneck, compressing data, and
    decompressing data. It also includes functionalities for analysis (if
    `AnalyzableModule` is used) and quantization of the compression module.

    Key functionalities:
        - `update()`: Updates the entropy bottleneck of the compression module.
        - `compress()`: Compresses input data using the compression module.
        - `decompress()`: Decompresses data using the compression module.
        - `activate_analysis()`/`deactivate_analysis()`: Control analysis capabilities.
        - `prepare_quantization()`/`apply_quantization()`: Manage quantization of the
          compression module.

    Args:
        compression_module (CompressionModule): The compression module to be
            integrated into the network. This module handles the actual
            compression and decompression logic.
        backbone (nn.Module): The main task-specific neural network (e.g., a
            classifier, detector).
        analysis_config (dict, optional): Configuration for analysis features,
            often used by `AnalyzableModule`. Defaults to None.
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
    An image classifier network that integrates a `CompressionModule` to compress
    features before they are fed to the main backbone.

    This class allows for "splittable" execution where, during inference and
    if analysis is activated, features from the compression module's analysis
    transform (g_a) are compressed, then decompressed by the synthesis transform
    (g_s), and finally processed by the backbone. The `reconstruction_layer`
    (assigned to `g_s.final_layer`) adapts the output of the synthesis transform
    to be compatible with the backbone. During training, or when analysis is not
    active for compression, the input typically passes through the full
    compression module (g_a and g_s) before the backbone.

    Args:
        compressor (CompressionModule): The compression module responsible for
            feature compression and decompression.
        reconstruction_layer (nn.Module): A layer that reconstructs/adapts the
            features from the synthesis part of the compressor to match the
            input requirements of the backbone. This is typically assigned to
            `compressor.g_s.final_layer`.
        backbone (nn.Module): The main classification network that processes the
            (potentially decompressed) features.
        analysis_config (dict, optional): Configuration for analysis features.
            Defaults to None.
        tokenize_compression_output (bool, optional): If True, a `Tokenizer` is
            applied to the output of the compression module before passing it to
            the backbone. This is useful if the backbone expects tokenized input
            (e.g., a Transformer). Defaults to False.
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
    An image classifier network that includes a `CompressionModule` and is capable
    of reconstructing to an image-like representation or passing features to a backbone.

    This class offers flexibility in how the output of the compression module is
    used. It can either reconstruct to an image-like representation using the
    `reconstruction_layer` or pass the features directly to a `backbone` for
    classification. The behavior is controlled by `activate_image_compression()`
    and `activate_feature_compression()` methods.

    Args:
        compressor (CompressionModule): The compression module for feature
            compression and decompression.
        reconstruction_layer (nn.Module): A layer used to reconstruct the output
            of the compression module into an image-like format.
        backbone (nn.Module): The main classification network.
        analysis_config (dict, optional): Configuration for analysis features.
            Defaults to None.
        **kwargs: Additional keyword arguments passed to the parent class.

    Attributes:
        reconstruct_to_image (bool): If True, the forward pass reconstructs to an
            image using `reconstruction_layer`. If False, features are passed to
            the `backbone`.
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
                 analysis_config_parent=None):
        
        _shared_stem_temp = SharedInputStem(**shared_stem_config["params"])
        stem_output_channels = _shared_stem_temp.get_output_channels()

        actual_task_prob_model_config_params = task_probability_model_config["params"].copy()
        actual_task_prob_model_config_params["input_channels_from_stem"] = stem_output_channels
        _task_probability_model_temp = TaskProbabilityModel(**actual_task_prob_model_config_params)
        expected_film_cond_dim = _task_probability_model_temp.get_output_dim()
        
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

        compressor_instance = get_custom_compression_module(
            compression_module_config["name"],
            **compression_module_config["params"]
        )
        backbone_instance = get_model(model_name=backbone_config["name"], **backbone_config["params"])

        super().__init__(compressor_instance, backbone_instance, analysis_config=analysis_config_parent)

        self.shared_stem = _shared_stem_temp
        self.task_probability_model = _task_probability_model_temp
        
        if reconstruction_layer_for_backbone_config and reconstruction_layer_for_backbone_config.get("name"):
            reconstruction_layer_instance = get_layer(
                reconstruction_layer_for_backbone_config["name"],
                **reconstruction_layer_for_backbone_config.get("params", {})
            )
            if hasattr(self.compression_module.g_s, 'final_layer'):
                logger.info(f"Setting final_layer of g_s to {reconstruction_layer_for_backbone_config['name']}")
                self.compression_module.g_s.final_layer = reconstruction_layer_instance
                self.reconstruction_for_backbone = nn.Identity()
            else:
                self.reconstruction_for_backbone = reconstruction_layer_instance
        else:
            self.reconstruction_for_backbone = nn.Identity()

    def forward(self, x, targets=None):
        stem_features = self.shared_stem(x)
        conditioning_signal = self.task_probability_model(stem_features)
        
        reconstructed_features_from_compressor_output = self.compression_module(stem_features, conditioning_signal=conditioning_signal)
        
        if isinstance(reconstructed_features_from_compressor_output, tuple) and \
           len(reconstructed_features_from_compressor_output) > 0 and \
           isinstance(reconstructed_features_from_compressor_output[0], torch.Tensor):
            reconstructed_features = reconstructed_features_from_compressor_output[0]
        else:
            reconstructed_features = reconstructed_features_from_compressor_output
           
        features_for_backbone = reconstructed_features
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
    network_type="FiLMedNetworkWithSharedStem"
):
    network = get_model(
        model_name=network_type,
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
    shared_stem_config, 
    task_probability_model_config, 
    compression_module_config,
    backbone_config,
    reconstruction_layer_for_backbone_config=None, 
    analysis_config_parent=None, 
    network_type="FiLMedHFactorizedPriorCompressionModule" 
):
    
    network = get_model(
        model_name=network_type, 
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
