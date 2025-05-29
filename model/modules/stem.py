# model/modules/stem.py
from torch import nn
from .layers.conv import ConvBlock3x3

class SharedInputStem(nn.Module):
    def __init__(self, in_channels=3, initial_out_channels=64, num_stem_layers=2, final_stem_channels=128):
        super().__init__()
        layers = []
        current_channels = in_channels
        
        # First layer
        layers.append(ConvBlock3x3(current_channels, initial_out_channels, stride=2, activation=nn.LeakyReLU))
        current_channels = initial_out_channels
        
        # Intermediate stem layers (optional)
        # If num_stem_layers = 2, this loop runs once for the second layer.
        # If num_stem_layers = 1, this loop doesn't run.
        for i in range(1, num_stem_layers):
            out_c = final_stem_channels if i == num_stem_layers - 1 else current_channels
            layers.append(ConvBlock3x3(current_channels, out_c, stride=2, activation=nn.LeakyReLU))
            current_channels = out_c
        
        if num_stem_layers == 1: # Only one layer, initial_out_channels is the final unless overridden
            if initial_out_channels != final_stem_channels and num_stem_layers == 1:
                 # This case means num_stem_layers=1 but initial and final are different.
                 # Add a projection or ensure config is logical.
                 # For now, assume if num_stem_layers=1, initial_out_channels IS the final_stem_channels.
                 # The logic above handles if num_stem_layers > 1, the last layer uses final_stem_channels.
                 # If num_stem_layers == 1, current_channels is initial_out_channels.
                 # We need to ensure it becomes final_stem_channels.
                 if initial_out_channels != final_stem_channels:
                    layers.append(nn.Conv2d(initial_out_channels, final_stem_channels, kernel_size=1)) # Projection
                    current_channels = final_stem_channels
            self.stem_feature_channels = current_channels
        elif num_stem_layers > 1:
            self.stem_feature_channels = final_stem_channels
        elif num_stem_layers == 0: # No stem
            self.stem_feature_channels = in_channels
        else: # Should not happen
            raise ValueError(f"Invalid num_stem_layers: {num_stem_layers}")


        self.stem = nn.Sequential(*layers)

    def forward(self, x):
        return self.stem(x)

    def get_output_channels(self):
        return self.stem_feature_channels