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
            else:
                # Handle nested structure from torchdistill wrapper
                z_likelihoods = student_io_dict[list(student_io_dict.keys())[0]]
        else:
            z_likelihoods = student_io_dict
            
        # Compute rate: -log2(P(z)) summed over all elements
        if z_likelihoods.ndim >= 2:
            batch_size = z_likelihoods.shape[0]
            # Rate per pixel: sum(-log2 P(z)) / (B * H * W)
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
        if isinstance(student_io_dict, dict) and "z_film" in student_io_dict:
            z_film_likelihoods = {"z": student_io_dict["z_film"]}
        else:
            z_film_likelihoods = student_io_dict
            
        return super().forward(z_film_likelihoods, target_dummy, **kwargs)


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
            elif config['type'] == 'BCEWithLogitsLoss':
                self.task_loss_fns.append(nn.BCEWithLogitsLoss(**config.get('params', {})))
            else:
                raise ValueError(f"Unsupported loss type: {config['type']}")

    def forward(self, downstream_outputs_list, y_downstream_list, active_task_mask):
        """
        Compute multi-task downstream loss.
        
        Args:
            downstream_outputs_list: List of model outputs for each task
            y_downstream_list: List of ground truth labels for each task
            active_task_mask: Boolean tensor (B, num_tasks) indicating active tasks
            
        Returns:
            Average loss over active tasks
        """
        total_loss = torch.tensor(0.0, device=active_task_mask.device, dtype=torch.float32)
        num_active_tasks = 0

        for k in range(self.num_tasks):
            if downstream_outputs_list[k] is None:
                continue

            # Select samples for which task k is active
            active_samples_mask_k = active_task_mask[:, k].bool()
            if not active_samples_mask_k.any():
                continue
                
            output_k = downstream_outputs_list[k][active_samples_mask_k]
            target_k = y_downstream_list[k][active_samples_mask_k]
            
            if output_k.numel() > 0 and target_k.numel() > 0:
                task_k_loss = self.task_loss_fns[k](output_k, target_k)
                total_loss += task_k_loss
                num_active_tasks += 1
        
        # Average over the number of active tasks
        if num_active_tasks > 0:
            return total_loss / num_active_tasks
        else:
            return total_loss


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
        # Assuming the structure is student_io_dict['.']['output'] from forward_hook
        if '.' in student_io_dict and 'output' in student_io_dict['.']:
            student_model_output = student_io_dict['.']['output']
        else:
            # Fallback: assume student_io_dict is the model output directly
            student_model_output = student_io_dict
            
        downstream_outputs_list = student_model_output['downstream_outputs']
        
        # Extract targets
        y_downstream_list = targets['Y_downstream']
        active_task_mask = targets['Y_task']
        
        return self.criterion(downstream_outputs_list, y_downstream_list, active_task_mask)


class TaskDetectorLoss(nn.Module):
    """
    Binary cross-entropy loss for task detection.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.bce_loss = nn.BCELoss(**kwargs)
        
    def forward(self, p_task, y_task):
        """
        Compute task detection loss.
        
        Args:
            p_task: Predicted task probabilities (B, num_tasks)
            y_task: Ground truth task labels (B, num_tasks)
            
        Returns:
            BCE loss between predictions and ground truth
        """
        return self.bce_loss(p_task, y_task.float())


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
        
        self.task_detector_loss = TaskDetectorLoss()
        self.downstream_loss = MultiTaskDownstreamLoss(num_tasks, task_loss_configs or [])
        self.rate_loss = VIBLossStage2()
        
    def forward(self, model_output, targets):
        """
        Compute combined MANTiS loss.
        
        Args:
            model_output: Dictionary from MantisStage2.forward()
            targets: Dictionary containing ground truth labels
            
        Returns:
            Combined loss and individual loss components
        """
        # Task detector loss
        task_loss = self.task_detector_loss(
            model_output['task_predictions'], 
            targets['Y_task']
        )
        
        # Downstream task losses
        downstream_loss = self.downstream_loss(
            model_output['downstream_outputs'],
            targets['Y_downstream'], 
            targets['Y_task']
        )
        
        # Rate loss
        rate_loss = self.rate_loss(model_output['z_film_likelihoods'])
        
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