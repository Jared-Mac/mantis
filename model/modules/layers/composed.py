"""
This module is intended for composed or more complex layer structures that
might not fit neatly into other categories. Currently, it primarily contains a
`MockLayer`.

Layers from this module can be used as placeholders or for specific experimental
setups within the broader network architectures defined in `model/network.py`
or as components in custom analysis/synthesis networks.

Key classes provided by the module:
    - MockLayer: An identity layer that can be used as a placeholder or for
      testing purposes. It is registered as a layer, model, analysis network,
      and synthesis network for flexibility.
"""
from compressai.layers import ResidualBlock
from sc2bench.models.layer import register_layer_class
from torch import Tensor, nn
from torch import nn
from timm.models.layers import to_2tuple
from torchdistill.models.registry import register_model_class

from model.modules.layers.transf import BasicSwinStage, Detokenizer, PatchMerging, Tokenizer
from model.modules.module_registry import register_analysis_network, register_synthesis_network


@register_layer_class
@register_model_class
@register_analysis_network
@register_synthesis_network
class MockLayer(nn.Identity):
    """
        Convenience class for experiments without image reconstruction
    """

    def __init__(self, **kwargs):
        super(MockLayer, self).__init__()
