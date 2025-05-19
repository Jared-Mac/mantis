from collections import OrderedDict
from functools import partial

import torch
from compressai.layers import GDN
from torch import nn
from functools import partial
from timm.models.vision_transformer import init_weights_vit_timm, Block
from timm.models.layers import trunc_normal_, to_2tuple

from model.modules.layers.conv import ConvGDNBlock, ResidualBlockWithStride
from model.modules.layers.preconfigured import get_layer_preconfiguration
from model.modules.layers.transf import HybridSwinStage
from model.modules.module_registry import register_analysis_network
from .layers.film import FiLMGenerator, FiLMLayer

class AnalysisNetwork(nn.Module):
    def __init__(self):
        super(AnalysisNetwork, self).__init__()

    @torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher


@register_analysis_network
class AnalysisNetworkCNN(AnalysisNetwork):
    def __init__(self, latent_channels, block_params=None):
        super(AnalysisNetwork, self).__init__()
        gdn_blocks = []

        if not block_params:
            block_params = [
                (3, latent_channels * 4, 5, 2, 2),
                (latent_channels * 4, latent_channels * 2, 5, 2, 3),
                (latent_channels * 2, latent_channels, 2, 1, 0),
            ]
        for in_channels, out_channels, kernel_size, stride, padding in block_params:
            gdn_blocks.append(ConvGDNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ))

        self.layers = nn.Sequential(*gdn_blocks)

    def forward(self, x):
        return self.layers(x)


@register_analysis_network
class QuantizableSimpleAnalysisNetwork2(AnalysisNetwork):
    def __init__(self,
                 target_channels,
                 in_ch=3,
                 in_ch1=64,
                 in_ch2=96,
                 **kwargs):
        super(QuantizableSimpleAnalysisNetwork2, self).__init__()
        self.rb1 = ResidualBlockWithStride(in_ch=in_ch, out_ch=in_ch1, activation=nn.ReLU, stride=2)
        self.rb2 = ResidualBlockWithStride(in_ch=in_ch1, out_ch=in_ch2, activation=nn.ReLU, stride=2)
        self.rb3 = ResidualBlockWithStride(in_ch=in_ch2, out_ch=target_channels, activation=nn.ReLU, stride=2)

    def forward(self, x):
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        return x
    
@register_analysis_network
class TaskConditionedFiLMedAnalysisNetwork(AnalysisNetwork):
    def __init__(self, 
                 input_channels_from_stem, # Number of channels from SharedInputStem
                 latent_channels,          # Desired output channels for the latent space (y)
                 block_configs,            # List of configs for each main block
                                           # e.g., [{'out_channels': N, 'kernel': K, 'stride': S, 'padding': P, 'apply_film': True/False}, ...]
                 film_cond_dim,            # Dimensionality of the conditioning signal from task probability model
                 film_generator_hidden_dim=None): # Optional: hidden dim for the FiLMGenerator MLP
        super().__init__()
        self.film_cond_dim = film_cond_dim

        self.main_blocks = nn.ModuleList()
        self.film_generators = nn.ModuleDict()
        self.film_layers = nn.ModuleDict()

        current_ch = input_channels_from_stem
        for i, b_config in enumerate(block_configs):
            # Using ConvGDNBlock as example, you can make this configurable too
            block = ConvGDNBlock(
                in_channels=current_ch,
                out_channels=b_config['out_channels'],
                kernel_size=b_config['kernel_size'],
                stride=b_config['stride'],
                padding=b_config['padding'],
                bias=False
            )
            self.main_blocks.append(block)
            current_ch = b_config['out_channels']

            if b_config.get('apply_film', False):
                if self.film_cond_dim is None:
                    raise ValueError("film_cond_dim must be provided if apply_film is true in any block_config")
                
                gen_key = f"film_gen_{i}"
                layer_key = f"film_layer_{i}"
                
                self.film_generators[gen_key] = FiLMGenerator(
                    cond_dim=self.film_cond_dim,
                    num_features=current_ch, # Modulate the output of this block
                    hidden_dim=film_generator_hidden_dim
                )
                self.film_layers[layer_key] = FiLMLayer()
        
        # Ensure final output has `latent_channels`
        if current_ch != latent_channels:
            self.final_projection = nn.Conv2d(current_ch, latent_channels, kernel_size=1)
        else:
            self.final_projection = nn.Identity()

    def forward(self, x_from_stem, conditioning_signal=None): # x_from_stem is output of SharedInputStem
        output = x_from_stem
        for i, block_module in enumerate(self.main_blocks):
            output = block_module(output)
            
            gen_key = f"film_gen_{i}"
            if gen_key in self.film_generators:
                if conditioning_signal is not None:
                    gamma, beta = self.film_generators[gen_key](conditioning_signal)
                    output = self.film_layers[f"film_layer_{i}"](output, gamma, beta)
                else:
                    # Recommended: If FiLM is integral, raise error. If optional, could skip.
                    # For now, we assume if configured, signal should be there for FiLM to apply.
                    # Or, you could have a mode where it passes through if signal is None.
                    print(f"Warning: TaskConditionedFiLMedAnalysisNetwork expects a conditioning_signal for block {i} but received None. FiLM not applied.")
        
        output = self.final_projection(output)
        return output

