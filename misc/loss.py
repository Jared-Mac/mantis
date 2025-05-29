# misc/loss.py
import os
from functools import partial

import timm
import torch
from pytorch_grad_cam import XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import Tensor, nn
from torchdistill.common.constant import def_logger
# This line is crucial:
from torchdistill.losses.registry import register_mid_level_loss # Removed register_loss_wrapper
from timm.models.layers import to_2tuple
from torch.nn import functional as F

# from saliency_maps.cam_prep.cam_patch import apply_multires_patches # Assuming this exists if uncommented
# model.modules.layers.transf might not be correct if this file is in misc
# Assuming Tokenizer is accessible or defined elsewhere if misc is standalone
# For now, let's comment it out if it's not directly used by the losses here
# from model.modules.layers.transf import Tokenizer 

logger = def_logger.getChild(__name__)


# Example of how Tokenizer might be defined if needed locally, or ensure correct import path
class Tokenizer(nn.Module):
    """
        Patch embed without Projection (From Image Tensor to Token TEnsor)
    """
    def __init__(self):
        super(Tokenizer, self).__init__()

    def forward(self, x):
        if x.dim() == 4: # B C H W
             x = x.flatten(2).transpose(1, 2)  # B H*W C
        elif x.dim() == 3: # B H*W C (already tokenized)
            pass
        else:
            logger.warning(f"Tokenizer received input with unexpected dim {x.dim()}")
        return x


@register_mid_level_loss 
class SalientPixelsBCELoss(nn.Module):
    def __init__(self, positive_trials, reduction='sum'):
        super(SalientPixelsBCELoss, self).__init__()
        self.positive_trials = positive_trials
        self.tokenizer = Tokenizer()
        self.bce = nn.BCELoss(reduction=reduction)

    def forward(self, input, target): # This forward signature implies it's wrapped or configured to receive specific input/target
        # If used as a mid-level loss directly, it would receive (student_io_dict, teacher_io_dict, targets_from_loader)
        # Let's assume it's wrapped by SimpleLossWrapper and `input` is `decision_scores`, `target` is `s_map_tuple`
        decision_scores = input # Assuming input is already decision_scores
        s_map = target # Assuming target is already s_map (potentially as part of a tuple)

        if isinstance(s_map, (list,tuple)) and len(s_map) > 1 and isinstance(s_map[1], torch.Tensor):
            s_map_tensor = s_map[1] # Assuming s_map is passed as (some_other_info, s_map_tensor)
        elif isinstance(s_map, torch.Tensor):
            s_map_tensor = s_map
        else:
            raise ValueError(f"Unexpected format for s_map target: {type(s_map)}")

        s_map_tokenized = self.tokenizer(s_map_tensor)
        soft_decisions = F.gumbel_softmax(decision_scores, hard=False)[:, :, 0:1]
        
        salient_pixels_idx = torch.argsort(s_map_tokenized, dim=1, descending=True)[:, :self.positive_trials]
        if salient_pixels_idx.dim() == 3 and salient_pixels_idx.shape[-1] == 1: # Ensure it's (B, k)
            salient_pixels_idx = salient_pixels_idx.squeeze(-1)

        trials = torch.zeros_like(s_map_tokenized)
        for idx in range(trials.size(0)):
            trials[idx][salient_pixels_idx[idx]] = 1.0 # Use 1.0 for float tensors
        return self.bce(soft_decisions, trials)


@register_mid_level_loss 
class MSELossWithPrecomputedCAMMapMultires(nn.Module):
    # ... (definition as before) ...
    def __init__(self,
                 alpha,
                 beta,
                 gamma,
                 tokenize=True):
        super(MSELossWithPrecomputedCAMMapMultires, self).__init__()
        self.tokenizer = Tokenizer() if tokenize else nn.Identity()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):        
        student_input_val = student_io_dict['compression_module']['input'] 
        if isinstance(student_input_val, (list, tuple)): 
             input_tensor, *s_maps = student_input_val
        else: 
            input_tensor = student_input_val # If only tensor is passed
            s_maps = [] # No s_maps provided directly with input

        if len(s_maps) < 2 : # Check if s_maps were provided
             # This indicates s_maps are expected to be found elsewhere or this loss is misused
             logger.warning("MSELossWithPrecomputedCAMMapMultires expects at least 2 saliency maps "
                            "from student_io_dict['compression_module']['input'] but found less.")
             # Fallback or error. For now, let's make dummy maps to avoid crashing, but this needs fixing in config/usage.
             s_map1 = torch.ones_like(input_tensor) 
             s_map2 = torch.ones_like(input_tensor)
        else:
            s_map1, s_map2 = s_maps[0], s_maps[1] # Take the first two if more are provided

        student_output1 = student_io_dict['compression_module']['output']
        student_output2 = student_io_dict['backbone.layers.0']['output']
        
        teacher_output1 = teacher_io_dict['layers.1']['output'] 
        teacher_output2 = teacher_io_dict['layers.2']['output'] 

        d1 = (student_output1 - teacher_output1).square().sum() * self.alpha
        
        # CAM map processing - ensure CAM map matches feature map dimensions or can be broadcasted
        processed_s_map1 = self.tokenizer(s_map1) # (B, HW, 1) if s_map1 was (B,1,H,W)
        d2_term = student_output1 - teacher_output1 # (B,C,H,W)
        
        # If student_output1 is (B,C,H,W) and processed_s_map1 is tokenized (B,HW,1)
        # we need to bring s_map1 back to spatial to multiply element-wise, or adapt features.
        # Assuming s_map1 is (B,1,H,W) or (B,H,W) initially.
        if s_map1.dim() == 4 and s_map1.shape[1] == 1: # B, 1, H, W
            d2_map_expanded = s_map1.expand_as(d2_term)
        elif s_map1.dim() == 3: # B, H, W (no channel dim)
            d2_map_expanded = s_map1.unsqueeze(1).expand_as(d2_term)
        else: # Fallback if shapes are already compatible or need specific handling
            d2_map_expanded = processed_s_map1 
        d2 = (d2_term.square() * d2_map_expanded).sum() * self.beta

        processed_s_map2 = self.tokenizer(s_map2)
        d3_term = student_output2 - teacher_output2
        if s_map2.dim() == 4 and s_map2.shape[1] == 1:
            d3_map_expanded = s_map2.expand_as(d3_term)
        elif s_map2.dim() == 3:
            d3_map_expanded = s_map2.unsqueeze(1).expand_as(d3_term)
        else:
            d3_map_expanded = processed_s_map2
        d3 = (d3_term.square() * d3_map_expanded).sum() * self.gamma
        
        return (d1 + d2 + d3) / (self.alpha + self.beta + self.gamma + 1e-9)


@register_mid_level_loss 
class MSELossWithPrecomputedCAMMapMultiresMixMaps(nn.Module):
    # ... (definition as before, ensure forward signature includes targets) ...
    def __init__(self,
                 alpha,
                 beta,
                 gamma,
                 tokenize=True):
        super(MSELossWithPrecomputedCAMMapMultiresMixMaps, self).__init__()
        self.tokenizer = Tokenizer() if tokenize else nn.Identity()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        student_input_val = student_io_dict['compression_module']['input']
        if isinstance(student_input_val, (list, tuple)):
             input_tensor, *s_maps = student_input_val
        else:
            input_tensor = student_input_val
            s_maps = []

        if len(s_maps) < 3:
             logger.warning("MSELossWithPrecomputedCAMMapMultiresMixMaps expects 3 saliency maps "
                            "but received less.")
             s_map0 = torch.ones_like(input_tensor)
             s_map1 = torch.ones_like(input_tensor)
             s_map2 = torch.ones_like(input_tensor)
        else:
            s_map0, s_map1, s_map2 = s_maps[0], s_maps[1], s_maps[2]


        student_output1 = student_io_dict['compression_module']['output']
        student_output2 = student_io_dict['backbone.layers.0']['output']
        
        map_for_l1 = torch.clamp(
            ((s_map1 + F.interpolate(s_map0, size=(s_map1.shape[2], s_map1.shape[3]), mode='bilinear', align_corners=False) + \
              F.interpolate(s_map2, size=(s_map1.shape[2], s_map1.shape[3]), mode='bilinear', align_corners=False)) / 3.0), min=0., max=1.5)
        map_for_l2 = torch.clamp(
            ((s_map2 + F.interpolate(s_map0, size=(s_map2.shape[2], s_map2.shape[3]), mode='bilinear', align_corners=False) + \
              F.interpolate(s_map1, size=(s_map2.shape[2], s_map2.shape[3]), mode='bilinear', align_corners=False)) / 3.0), min=0, max=1.5)
        
        teacher_output1 = teacher_io_dict['layers.1']['output'] 
        teacher_output2 = teacher_io_dict['layers.2']['output'] 
        
        d1 = (student_output1 - teacher_output1).square().sum() * self.alpha
        
        d2_term = student_output1 - teacher_output1
        d2_map_expanded = map_for_l1.expand_as(d2_term) 
        d2 = (d2_term.square() * d2_map_expanded).sum() * self.beta

        d3_term = student_output2 - teacher_output2
        d3_map_expanded = map_for_l2.expand_as(d3_term)
        d3 = (d3_term.square() * d3_map_expanded).sum() * self.gamma

        return (d1 + d2 + d3) / (self.alpha + self.beta + self.gamma + 1e-9)


@register_mid_level_loss 
class MSELossExtractClassToken(nn.MSELoss):
    # ... (definition as before) ...
    def __init__(self, reduction='mean'):
        super(MSELossExtractClassToken, self).__init__(reduction=reduction)

    def forward(self, input, target): 
        target = target[:, 1:] 
        return super().forward(input, target)


@register_mid_level_loss 
class BppLossOrig(nn.Module):
    # ... (definition as before, ensure forward signature includes targets) ...
    def __init__(self, entropy_module_path, input_sizes, reduction='mean'):
        super().__init__()
        self.entropy_module_path = entropy_module_path
        self.reduction = reduction
        self.input_h, self.input_w = to_2tuple(input_sizes)

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs): 
        entropy_module_dict = student_io_dict[self.entropy_module_path]
        
        eb_output = entropy_module_dict['output']
        if isinstance(eb_output, tuple) and len(eb_output) == 2:
            _, likelihoods = eb_output
        elif isinstance(eb_output, dict) and 'likelihoods' in eb_output:
            likelihoods = eb_output['likelihoods']
        else:
            raise ValueError(f"Unexpected output format from entropy_bottleneck at {self.entropy_module_path}")

        n = likelihoods.size(0) 
        if isinstance(likelihoods, dict) and 'y' in likelihoods: 
            likelihoods_y = likelihoods['y']
        elif isinstance(likelihoods, torch.Tensor): 
            likelihoods_y = likelihoods
        else:
            raise ValueError(f"Likelihoods from entropy_bottleneck are in an unexpected format: {type(likelihoods)}")

        if self.reduction == 'sum':
            bpp = -likelihoods_y.log2().sum()
        elif self.reduction == 'batchmean': 
            bpp = -likelihoods_y.log2().flatten(1).sum(1).mean() 
        elif self.reduction == 'mean': 
            bpp = -likelihoods_y.log2().sum() / (n * self.input_h * self.input_w)
        else:
            raise Exception(f"Reduction: {self.reduction} does not exist")
        return bpp

@register_mid_level_loss 
class BppLossOrigWithCAMEB(nn.Module):
    # ... (definition as before, ensure forward signature includes targets) ...
    def __init__(self, entropy_module_path, cam_entropy_module_path, input_sizes, reduction='mean', op='add'):
        super().__init__()
        self.entropy_module_path = entropy_module_path
        self.cam_entropy_module_path = cam_entropy_module_path
        self.reduction = reduction
        self.input_h, self.input_w = to_2tuple(input_sizes)
        if op == "add":
            self.f = lambda x, y: x + y
        elif op == 'add_avg':
            self.f = lambda x, y: ((x + y) + x) / 2
        else:
            raise ValueError

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        entropy_module_dict = student_io_dict[self.entropy_module_path]
        cam_entropy_module_dict = student_io_dict[self.cam_entropy_module_path]
        
        eb_output_latent = entropy_module_dict['output']
        if isinstance(eb_output_latent, tuple) and len(eb_output_latent) == 2: _, likelihoods_latent = eb_output_latent
        elif isinstance(eb_output_latent, dict): likelihoods_latent = eb_output_latent['likelihoods']['y'] 
        else: likelihoods_latent = eb_output_latent

        eb_output_cam = cam_entropy_module_dict['output']
        if isinstance(eb_output_cam, tuple) and len(eb_output_cam) == 2: _, likelihoods_cam = eb_output_cam
        elif isinstance(eb_output_cam, dict): likelihoods_cam = eb_output_cam['likelihoods']['y'] 
        else: likelihoods_cam = eb_output_cam
        
        n = likelihoods_latent.size(0)
        if self.reduction == 'sum':
            bpp = self.f(-likelihoods_latent.log2().sum(dim=1), -likelihoods_cam.log2().squeeze()).sum()
        elif self.reduction == 'batchmean':
            bpp = self.f(-likelihoods_latent.log2().sum(dim=1), -likelihoods_cam.log2().squeeze()).sum() / n
        elif self.reduction == 'mean':
            bpp = self.f(-likelihoods_latent.log2().sum(dim=1), -likelihoods_cam.log2().squeeze()).sum() / (
                    n * self.input_h * self.input_w)
        else:
            raise Exception(f"Reduction: {self.reduction} does not exist")
        return bpp

@register_mid_level_loss 
class MSELossMultiresCAM(nn.Module):
    # ... (definition as before, ensure forward signature includes targets) ...
    @staticmethod
    def reshape_transform(tensor, height=7, width=7): # Assuming default ViT patch size from 224x224
        # This needs to be robust if tensor shape doesn't match expected token lengths
        # For example, if tensor is (B, N, C)
        B, N, C_ = tensor.shape
        # Try to infer height/width assuming square patches, or use provided defaults
        # This is a heuristic and might need adjustment based on actual model structure
        if N == 56*56: height, width = 56,56
        elif N == 28*28: height, width = 28,28
        elif N == 14*14: height, width = 14,14
        elif N == 7*7: height, width = 7,7
        # else: use default height, width or raise error

        result = tensor.reshape(B, height, width, C_)
        result = result.permute(0, 3, 1, 2).contiguous() # B, C, H, W
        return result

    def __init__(self,
                 alpha1,
                 alpha2,
                 alpha3,
                 cam_device, # Device for CAM model and its inputs
                 map_mode=None):
        super(MSELossMultiresCAM, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        # Ensure CAM model is on the specified device
        self.cam_model_device = torch.device(cam_device if torch.cuda.is_available() else "cpu")
        model_for_cam = timm.create_model(model_name='swin_s3_tiny_224', pretrained=True).to(self.cam_model_device)
        model_for_cam.eval() # Set CAM model to eval mode

        target_layers = [
            model_for_cam.layers[2].blocks[-1].norm1, # Example target layer
            model_for_cam.layers[3].blocks[-1].norm1  # Example target layer
        ]
        cam = XGradCAM(model=model_for_cam,
                       target_layers=target_layers,
                       use_cuda=(self.cam_model_device.type == 'cuda'), # Match use_cuda with device
                       reshape_transform=self.reshape_transform) # Use class's static method

        # Assuming apply_multires_patches is defined elsewhere and works with the CAM object
        # from saliency_maps.cam_prep.cam_patch import apply_multires_patches # Placeholder
        # self.cam = apply_multires_patches(cam) 
        self.cam = cam # Using basic CAM for now
        self.tokenizer = Tokenizer()
        self.f = nn.Identity()


    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        student_input_val = student_io_dict['compression_module']['input']
        if isinstance(student_input_val, (list, tuple)):
             input_tensor, *s_maps_if_any = student_input_val
        else:
             input_tensor = student_input_val
        
        student_output1 = student_io_dict['compression_module']['output']
        student_output2 = student_io_dict['backbone.layers.0']['output']
        
        main_task_labels = targets[0] if isinstance(targets, (list,tuple)) else targets
        cam_targets_for_gradcam = [ClassifierOutputTarget(label.item()) for label in main_task_labels]
        
        # Ensure input_tensor is on the same device as the CAM model
        cam_input = input_tensor.detach().clone().to(self.cam_model_device)
        
        # Grad-CAM produces one map per target layer.
        # If self.cam is apply_multires_patches, it might return a list of maps.
        # If self.cam is the raw XGradCAM, it returns a single map (or one per layer if multiple target_layers).
        # For this example, let's assume XGradCAM with 2 target layers returns a list of 2 maps.
        raw_cam_maps = self.cam(input_tensor=cam_input, targets=cam_targets_for_gradcam) # list of [B,H,W] maps

        if not isinstance(raw_cam_maps, list) or len(raw_cam_maps) < 2:
            logger.error(f"XGradCAM did not return expected list of 2 maps. Got: {type(raw_cam_maps)}")
            # Create dummy maps on the correct device to prevent crashing
            map1 = torch.ones_like(student_output1[:, 0, :, :]).unsqueeze(1).to(student_output1.device) # B,1,H,W
            map2 = torch.ones_like(student_output2[:, 0, :, :]).unsqueeze(1).to(student_output2.device) # B,1,H,W
        else:
            map1 = torch.from_numpy(raw_cam_maps[0]).unsqueeze(1).to(student_output1.device) # B,1,H,W
            map2 = torch.from_numpy(raw_cam_maps[1]).unsqueeze(1).to(student_output2.device) # B,1,H,W
        
        # Resize CAM maps to match feature map sizes if necessary
        map1_resized = F.interpolate(map1, size=student_output1.shape[2:], mode='bilinear', align_corners=False)
        map2_resized = F.interpolate(map2, size=student_output2.shape[2:], mode='bilinear', align_corners=False)

        teacher_output1 = teacher_io_dict['layers.1']['output']
        teacher_output2 = teacher_io_dict['layers.2']['output']
        
        d1 = (student_output1 - teacher_output1).square().sum() * self.alpha1
        
        d2_term = student_output1 - teacher_output1
        d2_map_expanded = map1_resized.expand_as(d2_term)
        d2 = (d2_term.square() * d2_map_expanded).sum() * self.alpha2
        
        d3_term = student_output2 - teacher_output2
        d3_map_expanded = map2_resized.expand_as(d3_term)
        d3 = (d3_term.square() * d3_map_expanded).sum() * self.alpha3
        
        # self.cam.model.zero_grad() # Already in no_grad context during validation/loss computation
        return d1 + d2 + d3


@register_mid_level_loss 
class MSELossWithPrecomputedCAMMapSingle(nn.Module):
    # ... (definition as before, ensure forward signature includes targets) ...
    @staticmethod
    def _mult_avg(d, s_map):
        return (d + MSELossWithPrecomputedCAMMapSingle._mult(d, s_map)) / 2.0 # Add 2.0 for float division

    @staticmethod
    def _mult(d, s_map): 
        return d * s_map

    @staticmethod
    def _weighted_mult_avg(d, s_map, weight_d, weight_s):
        return (weight_d * d + weight_s * MSELossWithPrecomputedCAMMapSingle._mult(d, s_map)) / (weight_d + weight_s + 1e-9)

    @staticmethod
    def _identity(d, s_map):
        return d

    def __init__(self,
                 mode='bilinear', # Changed from bicubic to bilinear, more common for CAMs
                 tokenize=True,
                 interpol_to=None):
        super(MSELossWithPrecomputedCAMMapSingle, self).__init__()
        self.tokenizer = Tokenizer() if tokenize else nn.Identity()
        self.mode = mode
        self.interpol_to = to_2tuple(interpol_to) if interpol_to else None


    def forward(self, input_features_and_cam_map: tuple, target_features: Tensor): 
        # Expects input to be a tuple (features_from_student, cam_map_for_student)
        # Target is features_from_teacher (or ground truth if not distillation)
        input_features, cam_map = input_features_and_cam_map
        
        if self.interpol_to and (cam_map.shape[2:] != self.interpol_to):
            cam_map = F.interpolate(cam_map, size=self.interpol_to, mode=self.mode, align_corners=False if self.mode != 'nearest' else None)
        
        d = (input_features - target_features).square()
        
        # Process cam_map: tokenize if needed, then ensure it's broadcastable to d
        processed_cam_map = self.tokenizer(cam_map) # This might turn (B,1,H,W) into (B,HW,1)
        
        # If d is (B,C,H,W) and processed_cam_map is tokenized (B,HW,1)
        # We need the spatial cam_map (B,1,H,W) or (B,H,W) for element-wise multiplication
        if d.dim() == 4 and processed_cam_map.dim() == 3 :
            # Use the original cam_map (spatial) before tokenization for expansion
            if cam_map.dim() == 4 and cam_map.shape[1] == 1: # B,1,H,W
                 cam_map_spatial = cam_map
            elif cam_map.dim() == 3: # B,H,W
                 cam_map_spatial = cam_map.unsqueeze(1) # B,1,H,W
            else: # Fallback, this might error if shapes are incompatible
                 cam_map_spatial = processed_cam_map 
        elif d.dim() == 4 and processed_cam_map.dim() == 4 : # Both spatial
            cam_map_spatial = processed_cam_map
        else: # Other cases, assume processed_cam_map is fine or needs other handling
            cam_map_spatial = processed_cam_map

        cam_map_expanded = cam_map_spatial.expand_as(d)
        d_ = MSELossWithPrecomputedCAMMapSingle._mult_avg(d, cam_map_expanded)
        return d_.sum()


@register_mid_level_loss 
class MultiLabelTaskRelevancyBCELoss(nn.Module):
    # ... (definition as before) ...
    def __init__(self, reduction='mean'):
        super(MultiLabelTaskRelevancyBCELoss, self).__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor: 
        return self.bce_with_logits(input, target)


@register_mid_level_loss # Changed from register_loss_wrapper
class IndexedSimpleLossWrapper(nn.Module):
    # ... (definition as before) ...
    def __init__(self, single_loss, params_config, extract_idx):
        super().__init__()
        # single_loss is now the criterion config dict, instantiate it here
        # Or, expect it to be an instantiated nn.Module passed from get_mid_level_loss
        if isinstance(single_loss, dict): # If YAML passes config for single_loss
            # This case is complex as get_mid_level_loss already handles instantiation.
            # This wrapper should ideally receive an *instantiated* mid-level loss.
            # For now, assume single_loss IS an instantiated nn.Module.
            # If not, the YAML structure for this wrapper needs to be clear.
            raise DeprecationWarning("IndexedSimpleLossWrapper expects an instantiated nn.Module for single_loss.")
        self.single_loss = single_loss 
        
        input_config = params_config['input']
        self.is_input_from_teacher = input_config['is_from_teacher']
        self.input_module_path = input_config['module_path']
        self.input_key = input_config['io'] # e.g., 'output' or 'output.main_output' if output is dict
        
        target_config = params_config.get('target', {}) 
        self.is_target_from_teacher = target_config.get('is_from_teacher', False)
        self.target_module_path = target_config.get('module_path', None)
        self.target_key = target_config.get('io', None) # e.g., 'output' or 'target[0]'
        self.uses_label_as_target = target_config.get('uses_label', False)

        self.extract_idx = extract_idx # This is the key for dicts or index for tuples

    @staticmethod
    def extract_value(io_dict_or_targets, path, key, extract_idx_from_key):
        # If io_dict_or_targets is the `targets` tuple from dataloader
        if path == '.' and key.startswith('target['): # Special handling for targets tuple
            try:
                target_idx = int(key.split('[')[-1].split(']')[0])
                base_target = io_dict_or_targets[target_idx]
                # Now, if extract_idx_from_key is meant to index *within* that base_target (if it's a dict/list)
                if isinstance(extract_idx_from_key, str) and isinstance(base_target, dict):
                    return base_target[extract_idx_from_key]
                # If extract_idx_from_key is not relevant or base_target is not a dict, return base_target
                return base_target
            except: # Fallback, or raise error
                 raise ValueError(f"Could not parse target key: {key} with targets: {io_dict_or_targets}")

        # Standard path and key processing for io_dict
        data_at_path_key = io_dict_or_targets[path][key]
        
        # If data_at_path_key is a dict and extract_idx_from_key is a string key for it
        if isinstance(extract_idx_from_key, str) and isinstance(data_at_path_key, dict):
            return data_at_path_key[extract_idx_from_key]
        # If data_at_path_key is a list/tuple and extract_idx_from_key is an int index for it
        elif isinstance(extract_idx_from_key, int) and isinstance(data_at_path_key, (list,tuple)):
            return data_at_path_key[extract_idx_from_key]
        
        # If extract_idx_from_key is None or doesn't match type, return the whole data_at_path_key
        return data_at_path_key

    def forward(self, student_io_dict, teacher_io_dict, targets_from_loader, *args, **kwargs):
        input_io_source = teacher_io_dict if self.is_input_from_teacher else student_io_dict
        # The self.input_key might itself contain the dict key, e.g. 'output.main_output'
        # If self.input_key is simple 'output', then self.extract_idx would be 'main_output'
        final_input_key_parts = self.input_key.split('.')
        base_input_key = final_input_key_parts[0]
        input_dict_subkey = final_input_key_parts[1] if len(final_input_key_parts) > 1 else self.extract_idx

        input_batch = self.extract_value(input_io_source,
                                         self.input_module_path, base_input_key, input_dict_subkey)
        
        if self.uses_label_as_target:
            # self.target_key should indicate which part of `targets_from_loader` to use, e.g., 'target[0]'
            # And self.extract_idx might further index into that if it's a dict/list.
            # For simplicity here, assume self.target_key like 'target[0]' is handled by extract_value
            target_batch = self.extract_value(targets_from_loader, '.', self.target_key, self.extract_idx if not isinstance(self.extract_idx, int) else None)
        elif self.target_module_path is not None and self.target_key is not None:
            target_io_source = teacher_io_dict if self.is_target_from_teacher else student_io_dict
            final_target_key_parts = self.target_key.split('.')
            base_target_key = final_target_key_parts[0]
            target_dict_subkey = final_target_key_parts[1] if len(final_target_key_parts) > 1 else self.extract_idx

            target_batch = self.extract_value(target_io_source,
                                              self.target_module_path, base_target_key, target_dict_subkey)
        else: 
            target_batch = targets_from_loader

        return self.single_loss(input_batch, target_batch, *args, **kwargs)

    def __str__(self):
        idx_str = self.extract_idx if self.extract_idx is not None else ""
        return f"Indexed({idx_str})[{self.single_loss.__str__()}]"