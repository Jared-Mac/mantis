import torch
import torch.nn as nn
from torchdistill.models.registry import register_model_class

# Basic Residual Block (can be a simpler conv block if ResNet is too complex for a start)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

@register_model_class
class MultiTailResNetBackbone(nn.Module):
    def __init__(self, input_features_dim, num_tails, per_tail_output_classes, block=BasicBlock, num_blocks=[2,2,2,2]): # num_blocks similar to ResNet18
        super().__init__()
        self.input_features_dim = input_features_dim # This is likely the number of channels from g_s
        self.num_tails = num_tails
        self.per_tail_output_classes = per_tail_output_classes

        # self.in_planes = input_features_dim # This was part of ResNet example, but for distinct tails, each starts with input_features_dim

        self.tails = nn.ModuleList()
        for _ in range(num_tails):
            # Each tail processes the *same* input from g_s, so in_planes for _make_layer is input_features_dim for all.
            
            # Small ResNet-like feature extractor for each tail
            tail_feature_extractor = nn.Sequential(
                self._make_layer(block, planes=64, num_blocks=num_blocks[0], stride=1, in_planes=self.input_features_dim), # planes can be configurable
                # Example: self._make_layer(block, 128, num_blocks[1], stride=2, in_planes=64 * block.expansion), # if deeper and chaining layers
                nn.AdaptiveAvgPool2d((1, 1)) # Pool to 1x1
            )
            # The output features of _make_layer with planes=64 and BasicBlock will be 64 * block.expansion
            tail_classifier = nn.Linear(64 * block.expansion, per_tail_output_classes)
            
            current_tail = nn.Sequential(
                tail_feature_extractor,
                nn.Flatten(),
                tail_classifier
            )
            self.tails.append(current_tail)

    # Utility from typical ResNet implementations
    def _make_layer(self, block, planes, num_blocks, stride, in_planes):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        # `in_planes` here is specific to this layer's construction
        current_in_planes = in_planes 
        for strd in strides:
            layers.append(block(current_in_planes, planes, strd))
            current_in_planes = planes * block.expansion
        # self.in_planes = planes * block.expansion # This line would be for a sequential ResNet, not for parallel tails starting from same input_features_dim
        return nn.Sequential(*layers)

    def forward(self, x_features, conditioning_signal=None):
        # x_features: output from the FiLM encoder's synthesis network (g_s)
        # conditioning_signal: In this design, it's not directly used by the backbone to route.
        # Instead, the backbone returns all tail outputs, and the main model or loss function uses
        # the conditioning_signal to select/weight the appropriate outputs.

        all_tail_outputs = []
        for tail in self.tails:
            all_tail_outputs.append(tail(x_features))
        
        # Return a dictionary keyed by tail index.
        output_dict = {f"tail_{i}": all_tail_outputs[i] for i in range(self.num_tails)}
        
        return output_dict
