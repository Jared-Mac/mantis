# In a new file, e.g., model/modules/common_blocks.py
# Or at the top of model/modules/analysis.py if you prefer
from torch import nn
# Example: using ConvBlock3x3 from your layers
from .layers.conv import ConvBlock3x3 # Adjust path if common_blocks.py is outside modules

class SharedInputStem(nn.Module):
    def __init__(self, in_channels=3, initial_out_channels=64, num_stem_layers=2, final_stem_channels=128):
        super().__init__()
        layers = []
        current_channels = in_channels
        
        # First layer
        layers.append(ConvBlock3x3(current_channels, initial_out_channels, stride=2, activation=nn.LeakyReLU))
        current_channels = initial_out_channels
        
        # Intermediate stem layers (optional)
        for _ in range(1, num_stem_layers -1): # If num_stem_layers >= 2
            layers.append(ConvBlock3x3(current_channels, current_channels, stride=2, activation=nn.LeakyReLU))
        
        # Final stem layer to reach final_stem_channels
        if num_stem_layers > 1 :
             layers.append(ConvBlock3x3(current_channels, final_stem_channels, stride=2, activation=nn.LeakyReLU))
             self.stem_feature_channels = final_stem_channels
        elif num_stem_layers == 1: # Only one layer, initial_out_channels is the final
             self.stem_feature_channels = initial_out_channels
        else: # No stem layers, should not happen if we intend to have a stem
             self.stem_feature_channels = in_channels # Or raise error

        self.stem = nn.Sequential(*layers)

    def forward(self, x):
        return self.stem(x)

    def get_output_channels(self):
        return self.stem_feature_channels