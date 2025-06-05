import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VIBLossStage1(nn.Module):
    """
    VIB rate loss for Stage 1 training.
    
    Computes the rate term from entropy bottleneck likelihoods.
    """
    
    def __init__(self, num_pixels_placeholder=256*256, **kwargs):
        super().__init__()
        self.num_pixels_placeholder = num_pixels_placeholder

    def forward(self, student_io_dict, target_dummy=None, **kwargs):
        """
        Compute VIB rate loss from student outputs.
        
        Args:
            student_io_dict: Dictionary containing z_likelihoods
            target_dummy: Unused, for torchdistill compatibility
            
        Returns:
            Rate loss (bits per pixel)
        """
        # Extract z_likelihoods from student output
        if isinstance(student_io_dict, dict):
            if "z" in student_io_dict:
                z_likelihoods = student_io_dict["z"]
            # Handle nested structure from torchdistill wrapper if student_io_dict is like {'.': {'z_likelihoods': actual_dict}}
            elif '.' in student_io_dict and isinstance(student_io_dict['.'], dict) and \
                 list(student_io_dict['.'].keys())[0] == 'z_likelihoods' and \
                 "z" in student_io_dict['.']['z_likelihoods']:
                 z_likelihoods = student_io_dict['.']['z_likelihoods']["z"]
            elif isinstance(student_io_dict, dict) and list(student_io_dict.keys())[0] == 'z_likelihoods' and "z" in student_io_dict['z_likelihoods']:
                 z_likelihoods = student_io_dict['z_likelihoods']["z"] # If key is 'z_likelihoods'
            else:
                # Fallback for simpler direct dict passing or single key from wrapper
                # This part might need adjustment based on actual SimpleLossWrapper behavior
                # For now, assume it could be student_io_dict['z_likelihoods']['z'] or student_io_dict['.']['output']['z_likelihoods']['z']
                # For VIBLossStage1, the model output dict is {"z_likelihoods": {"z": z_likelihoods_tensor}}
                # So if student_io_dict is the 'z_likelihoods' part, then student_io_dict["z"] is correct.
                # If SimpleLossWrapper passes the entire model output dict under '.', then need to access further.
                # Let's assume SimpleLossWrapper passes the 'io' specified. If 'io' was 'z_likelihoods', then student_io_dict is {"z": tensor}
                if "z" in student_io_dict: #This was the most direct way it was defined
                    z_likelihoods = student_io_dict["z"]
                else:
                    raise ValueError(f"Could not extract z_likelihoods from student_io_dict: {student_io_dict}")

        else: # If student_io_dict is just the tensor itself
            z_likelihoods = student_io_dict
            
        # Compute rate: -log2(P(z)) summed over all elements
        if z_likelihoods.ndim >= 2:
            batch_size = z_likelihoods.shape[0]
            # Rate per pixel: sum(-log2 P(z)) / (B * H * W)
            # Add epsilon for numerical stability with log2
            rate = torch.sum(-torch.log2(z_likelihoods + 1e-10)) / (batch_size * self.num_pixels_placeholder)
        else:
            # Fallback for unexpected shapes
            rate = torch.sum(-torch.log2(z_likelihoods + 1e-10))
            
        return rate


class VIBLossStage2(VIBLossStage1):
    """
    VIB rate loss for Stage 2 training (z_film).
    
    Inherits from Stage 1 but handles z_film_likelihoods.
    """
    
    def forward(self, student_io_dict, target_dummy=None, **kwargs):
        # Handle z_film key specifically
        # Based on MantisStage2 output: {"z_film_likelihoods": {"z_film": z_film_likelihoods_tensor}}
        if isinstance(student_io_dict, dict) and "z_film" in student_io_dict:
             z_film_likelihoods_for_super = {"z": student_io_dict["z_film"]} 
        elif isinstance(student_io_dict, dict) and "z_film_likelihoods" in student_io_dict and "z_film" in student_io_dict["z_film_likelihoods"]:
             # If the whole model output dict key "z_film_likelihoods" is passed
             z_film_likelihoods_for_super = {"z": student_io_dict["z_film_likelihoods"]["z_film"]}
        elif isinstance(student_io_dict, torch.Tensor): # If only the tensor is passed
             z_film_likelihoods_for_super = {"z": student_io_dict}
        else:
            raise ValueError(f"Could not extract z_film_likelihoods from student_io_dict: {student_io_dict}")
            
        return super().forward(z_film_likelihoods_for_super, target_dummy, **kwargs)


class MultiTaskDownstreamLoss(nn.Module):
    """
    Multi-task downstream loss for Stage 2 training.
    
    Computes task-specific losses only for active tasks.
    """
    
    def __init__(self, num_tasks, task_loss_configs):
        """
        Initialize multi-task loss.
        
        Args:
            num_tasks: Number of tasks
            task_loss_configs: List of loss configurations for each task
                Example: [{'type': 'CrossEntropyLoss', 'params': {}}, ...]
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.task_loss_fns = nn.ModuleList()
        
        for config in task_loss_configs:
            if config['type'] == 'CrossEntropyLoss':
                self.task_loss_fns.append(nn.CrossEntropyLoss(**config.get('params', {})))
            elif config['type'] == 'MSELoss':
                self.task_loss_fns.append(nn.MSELoss(**config.get('params', {})))
            elif config['type'] == 'BCEWithLogitsLoss': # Added for completeness, though not used directly here
                self.task_loss_fns.append(nn.BCEWithLogitsLoss(**config.get('params', {})))
            else:
                raise ValueError(f"Unsupported loss type: {config['type']}")

    def forward(self, downstream_outputs_list, y_downstream_list, active_task_mask):
        """
        Compute multi-task downstream loss.
        
        Args:
            downstream_outputs_list: List of model outputs for each task
            y_downstream_list: List of ground truth labels for each task (already processed by collate_fn)
            active_task_mask: Boolean tensor (B, num_tasks) indicating active tasks
            
        Returns:
            Average loss over active tasks
        """
        total_loss = torch.tensor(0.0, device=active_task_mask.device, dtype=torch.float32)
        num_active_tasks_contributing = 0 # Renamed for clarity

        for k in range(self.num_tasks):
            if downstream_outputs_list[k] is None: # Should not happen if model always outputs
                continue

            # Select samples for which task k is active
            active_samples_mask_k = active_task_mask[:, k].bool() 
            if not active_samples_mask_k.any():
                continue
                
            output_k = downstream_outputs_list[k][active_samples_mask_k]
            # y_downstream_list[k] is already a tensor of shape (B) with ignore_index
            target_k = y_downstream_list[k][active_samples_mask_k] 
            
            if output_k.numel() > 0 and target_k.numel() > 0:
                # Ensure target_k doesn't only contain ignore_index if loss doesn't handle it well
                # CrossEntropyLoss handles ignore_index correctly.
                task_k_loss = self.task_loss_fns[k](output_k, target_k)
                if not torch.isnan(task_k_loss) and not torch.isinf(task_k_loss): # Avoid adding NaN/Inf loss
                    total_loss += task_k_loss
                    num_active_tasks_contributing += 1
        
        # Average over the number of active tasks that contributed a valid loss
        if num_active_tasks_contributing > 0:
            return total_loss / num_active_tasks_contributing
        else:
            # Return 0 loss if no tasks were active or all resulted in NaN/Inf
            # This prevents backward pass with NaN if this is the only loss component.
            return torch.tensor(0.0, device=active_task_mask.device, dtype=torch.float32, requires_grad=True)


class MultiTaskCriterionWrapper(nn.Module):
    """
    Wrapper for multi-task loss to integrate with torchdistill.
    
    Handles extraction of downstream outputs and targets from
    torchdistill's data structures.
    """
    
    def __init__(self, criterion, **kwargs):
        super().__init__()
        self.criterion = criterion

    def forward(self, student_io_dict, teacher_io_dict, targets, supp_dict=None, **kwargs):
        """
        Extract data and compute multi-task loss.
        
        Args:
            student_io_dict: Student model outputs from torchdistill
            teacher_io_dict: Teacher model outputs (unused)
            targets: Target dictionary containing task labels
            supp_dict: Supplementary dictionary (unused)
            
        Returns:
            Multi-task loss value
        """
        # Extract student model outputs
        if '.' in student_io_dict and 'output' in student_io_dict['.']:
            student_model_output = student_io_dict['.']['output']
        else:
            student_model_output = student_io_dict
            
        downstream_outputs_list = student_model_output['downstream_outputs']
        
        y_downstream_list = targets['Y_downstream'] # Already collated list of tensors
        active_task_mask = targets['Y_task']
        
        return self.criterion(downstream_outputs_list, y_downstream_list, active_task_mask)


class TaskDetectorLoss(nn.Module):
    """
    Binary cross-entropy loss for task detection, using BCEWithLogitsLoss for numerical stability with AMP.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        # Use BCEWithLogitsLoss as recommended for AMP safety
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(**kwargs)
        
    def forward(self, p_task_logits, y_task):
        """
        Compute task detection loss.
        
        Args:
            p_task_logits: Predicted task logits (B, num_tasks) - output from TaskDetector (no sigmoid)
            y_task: Ground truth task labels (B, num_tasks)
            
        Returns:
            BCEWithLogitsLoss between predictions and ground truth
        """
        # Ensure y_task is float, as expected by BCEWithLogitsLoss
        return self.bce_with_logits_loss(p_task_logits, y_task.float())


class CombinedMantisLoss(nn.Module):
    """
    Combined loss for MANTiS Stage 2 training.
    
    Combines task detector loss, downstream task losses, and rate loss.
    """
    
    def __init__(self, task_detector_loss_weight=1.0, downstream_loss_weight=1.0, 
                 rate_loss_weight=0.01, num_tasks=5, task_loss_configs=None):
        super().__init__()
        
        self.task_detector_loss_weight = task_detector_loss_weight
        self.downstream_loss_weight = downstream_loss_weight
        self.rate_loss_weight = rate_loss_weight
        
        self.task_detector_loss_fn = TaskDetectorLoss() # Uses BCEWithLogitsLoss now
        self.downstream_loss_fn = MultiTaskDownstreamLoss(num_tasks, task_loss_configs or [])
        self.rate_loss_fn = VIBLossStage2()
        
    def forward(self, model_output, targets):
        """
        Compute combined MANTiS loss.
        
        Args:
            model_output: Dictionary from MantisStage2.forward()
                          model_output['task_predictions'] should now be logits.
            targets: Dictionary containing ground truth labels
            
        Returns:
            Combined loss and individual loss components
        """
        # Task detector loss (expects logits)
        task_loss = self.task_detector_loss_fn(
            model_output['task_predictions'], # These are now logits
            targets['Y_task']
        )
        
        # Downstream task losses
        downstream_loss = self.downstream_loss_fn(
            model_output['downstream_outputs'],
            targets['Y_downstream'], 
            targets['Y_task']
        )
        
        # Rate loss
        rate_loss = self.rate_loss_fn(model_output['z_film_likelihoods'])
        
        # Combined loss
        total_loss = (self.task_detector_loss_weight * task_loss + 
                     self.downstream_loss_weight * downstream_loss + 
                     self.rate_loss_weight * rate_loss)
        
        return {
            'total_loss': total_loss,
            'task_detector_loss': task_loss,
            'downstream_loss': downstream_loss,
            'rate_loss': rate_loss
        } 