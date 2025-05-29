# phased_training_box.py
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from torchdistill.common import module_util, yaml_util
from torchdistill.common.constant import def_logger
from torchdistill.core.interfaces.box import TrainingBoxInterface # Using interface for clarity
from torchdistill.datasets import util as dataset_util
from torchdistill.optim import get_optimizer, get_scheduler
from torchdistill.losses.util import get_criterion
from torchdistill.models.registry import get_model as get_torchdistill_model # avoid clash
from torchdistill.common.main_util import load_ckpt


logger = def_logger.getChild(__name__)

class PhasedTrainingBox(TrainingBoxInterface):
    """
    Manages training across multiple phases, each with potentially different
    trainable parameters, optimizers, schedulers, and loss criteria.
    """
    def __init__(
        self,
        teacher_model: torch.nn.Module, # Initial teacher for phase 1
        student_model: torch.nn.Module,
        dataset_dict: dict,
        train_config: dict, # Contains phase1, phase2, etc.
        device: torch.device,
        device_ids: list = None,
        distributed: bool = False,
        lr_factor: float = 1.0, # Not used if LRs are in YAML per phase
        accelerator=None, # For Hugging Face Accelerate, not used here
    ):
        super().__init__()
        self.teacher_model_initial = teacher_model # Used for first phase if needed
        self.student_model = student_model
        self.dataset_dict = dataset_dict
        self.full_train_config = train_config
        self.device = device
        self.device_ids = device_ids
        self.distributed = distributed
        self.accelerator = accelerator # Not used in this impl

        self.current_phase_key = None
        self.current_phase_config = None
        self.current_phase_epochs_done = 0
        self.total_epochs_across_phases = 0
        self.stage_number = 0 # Tracks current phase number (1, 2, 3...)
        self.stage_key_prefix = "phase" # As in phase1, phase2...

        self.optimizer: Optimizer = None
        self.lr_scheduler: _LRScheduler = None
        self.criterion: torch.nn.Module = None
        self.train_data_loader: DataLoader = None
        self.val_data_loader: DataLoader = None
        self.teacher_model: torch.nn.Module = None # Current teacher for the active phase

        self._calculate_total_epochs()
        self.advance_to_next_stage() # Initialize for the first phase


    def _calculate_total_epochs(self):
        self.total_epochs_across_phases = 0
        phase_num = 1
        while True:
            phase_key = f"{self.stage_key_prefix}{phase_num}"
            if phase_key in self.full_train_config:
                self.total_epochs_across_phases += self.full_train_config[phase_key]["num_epochs"]
                phase_num += 1
            else:
                break
        logger.info(f"Total epochs across all phases: {self.total_epochs_across_phases}")
        if self.total_epochs_across_phases == 0:
            raise ValueError("No phases found or num_epochs not set in any phase.")


    @property
    def num_epochs(self): # Total epochs for the main training loop
        return self.total_epochs_across_phases

    def _setup_phase(self, phase_key: str):
        logger.info(f"Setting up for phase: {phase_key}")
        self.current_phase_key = phase_key
        self.current_phase_config = self.full_train_config[phase_key]
        self.current_phase_epochs_done = 0 # Reset for the new phase

        # 1. Set trainable modules
        self._set_trainable_modules()

        # 2. Setup Optimizer and Scheduler
        self._setup_optimizer_and_scheduler()

        # 3. Setup Criterion
        criterion_config = self.current_phase_config["criterion"]
        self.criterion = get_criterion(criterion_config, self.student_model, self.teacher_model)
        logger.info(f"Criterion for {phase_key}: {self.criterion}")

        # 4. Setup Teacher model for the current phase (if specified)
        # The initial teacher_model_initial is used if phase1 needs it and no specific teacher defined for phase1
        if "teacher_model" in self.current_phase_config: # Phase-specific teacher
            teacher_config = self.current_phase_config["teacher_model"]
            self.teacher_model = get_torchdistill_model(teacher_config['name'], **teacher_config['params'])
            if 'ckpt' in teacher_config and teacher_config['ckpt']:
                load_ckpt(teacher_config['ckpt'], model=self.teacher_model, strict=True)
            self.teacher_model = self.teacher_model.to(self.device)
            if self.teacher_model:
                for param in self.teacher_model.parameters():
                    param.requires_grad = False
                self.teacher_model.eval()
            logger.info(f"Loaded teacher model for phase {phase_key}: {teacher_config['name']}")
        elif self.stage_number == 1 and self.teacher_model_initial: # Use initial teacher for phase 1
            self.teacher_model = self.teacher_model_initial
            logger.info(f"Using initial teacher model for phase {phase_key}")
        else:
            self.teacher_model = None
            logger.info(f"No teacher model for phase {phase_key}")


        # 5. Setup DataLoaders
        train_loader_config = self.current_phase_config["train_data_loader"]
        val_loader_config = self.current_phase_config.get("val_data_loader", None)
        
        loader_configs = [train_loader_config]
        if val_loader_config:
            loader_configs.append(val_loader_config)
        
        data_loaders = dataset_util.build_data_loaders(
            self.dataset_dict, loader_configs, self.distributed
        )
        self.train_data_loader = data_loaders[0]
        self.val_data_loader = data_loaders[1] if val_loader_config and len(data_loaders) > 1 else None
        logger.info(f"Train Dataloader for {phase_key} built. Num batches: {len(self.train_data_loader)}")
        if self.val_data_loader:
            logger.info(f"Val Dataloader for {phase_key} built. Num batches: {len(self.val_data_loader)}")
            
        # 6. Handle DDP wrapping for student model (if not already handled outside)
        # torchdistill typically handles this in the main script or via accelerator.
        # If PhasedTrainingBox is the primary DDP manager for the student model:
        if self.distributed and not isinstance(self.student_model, torch.nn.parallel.DistributedDataParallel):
             self.student_model = torch.nn.parallel.DistributedDataParallel(self.student_model, device_ids=[self.device.index]) # Assumes single device per process

    def _set_trainable_modules(self):
        student_model_without_ddp = self.student_model.module if module_util.check_if_wrapped(self.student_model) else self.student_model
        
        # First, freeze all parameters
        for param in student_model_without_ddp.parameters():
            param.requires_grad = False

        # Then, unfreeze specified modules
        trainable_module_paths = self.current_phase_config.get("trainable_modules", [])
        if not trainable_module_paths:
            logger.warning(f"No 'trainable_modules' specified for phase {self.current_phase_key}. All student model params will remain frozen unless optimizer re-enables them.")
            # This might be intentional if only, e.g., a teacher is being trained, or if the optimizer targets specific params not through module names.
            # However, for student training, this list is crucial.
            # For safety, if empty, one might choose to make all params trainable or raise an error.
            # Here, we'll assume the optimizer will handle it or it's intentional.

        for path in trainable_module_paths:
            module = module_util.get_module(student_model_without_ddp, path)
            if module is None:
                logger.warning(f"Module path '{path}' not found in student model for phase {self.current_phase_key}. Skipping.")
                continue
            for param in module.parameters():
                param.requires_grad = True
            logger.info(f"Made module '{path}' trainable for phase {self.current_phase_key}")

        # Log which parameters are trainable
        num_trainable_params = 0
        for name, param in student_model_without_ddp.named_parameters():
            if param.requires_grad:
                num_trainable_params += param.numel()
                # logger.debug(f"  Trainable: {name}")
        logger.info(f"Total trainable parameters in student model for phase {self.current_phase_key}: {num_trainable_params}")


    def _setup_optimizer_and_scheduler(self):
        trainable_params = [p for p in self.student_model.parameters() if p.requires_grad]
        
        if not trainable_params:
            logger.warning(f"No trainable parameters found for optimizer in phase {self.current_phase_key}. Optimizer will not be created.")
            self.optimizer = None
            self.lr_scheduler = None
            return

        optimizer_config = self.current_phase_config["optimizer"]
        self.optimizer = get_optimizer(trainable_params, **optimizer_config)
        logger.info(f"Optimizer for {self.current_phase_key}: {self.optimizer}")

        scheduler_config = self.current_phase_config.get("scheduler", None)
        if scheduler_config:
            # Adjust T_max for CosineAnnealingLR to be num_epochs in the current phase
            if scheduler_config['type'] == 'CosineAnnealingLR' and 'params' in scheduler_config:
                 scheduler_config['params']['T_max'] = self.current_phase_config['num_epochs']
            self.lr_scheduler = get_scheduler(self.optimizer, **scheduler_config)
            logger.info(f"LR Scheduler for {self.current_phase_key}: {self.lr_scheduler}")
        else:
            self.lr_scheduler = None

    def advance_to_next_stage(self):
        self.stage_number += 1
        next_phase_key = f"{self.stage_key_prefix}{self.stage_number}"
        if next_phase_key in self.full_train_config:
            self._setup_phase(next_phase_key)
            return True
        else:
            logger.info("All training phases completed.")
            return False

    def pre_epoch_process(self, epoch, **kwargs):
        """
        epoch: global epoch number across all phases
        """
        # Check if we need to transition to the next phase
        if self.current_phase_epochs_done >= self.current_phase_config["num_epochs"]:
            if not self.advance_to_next_stage():
                # This should ideally be caught by the main loop's epoch limit
                raise RuntimeError("Tried to advance past the final phase.")
        
        self.student_model.train()
        if self.teacher_model: # Teacher is always in eval mode during student training
            self.teacher_model.eval()

        if self.distributed and hasattr(self.train_data_loader.sampler, 'set_epoch'):
            # Use the epoch relative to the start of the current phase for the sampler
            # Or, if sampler expects global epoch, adjust accordingly.
            # Torchdistill's default behavior for DistributedSampler usually uses the global epoch.
            current_phase_start_epoch = 0
            for i in range(1, self.stage_number):
                current_phase_start_epoch += self.full_train_config[f"{self.stage_key_prefix}{i}"]["num_epochs"]
            epoch_in_phase = epoch - current_phase_start_epoch
            self.train_data_loader.sampler.set_epoch(epoch_in_phase) # Or just `epoch` if global is fine


    def forward_process(self, sample_batch, targets, supp_dict=None, **kwargs):
        # Forward pass through student (and teacher if used by criterion)
        # The criterion from torchdistill (GeneralizedCustomLoss) handles IO dictionary creation
        # It expects the model to return a dictionary if multiple outputs are needed by different loss terms.
        
        # student_model is already DDP wrapped if distributed
        student_output = self.student_model(sample_batch)

        # teacher_output is only needed if criterion uses it (e.g. distillation)
        teacher_output = None
        if self.teacher_model and self.criterion.uses_teacher_output(): # Add uses_teacher_output to your criterion if needed
            teacher_output = self.teacher_model(sample_batch)
            
        # For GeneralizedCustomLoss, targets are passed directly.
        # The loss function itself will unpack student_output (if dict) and targets based on its config.
        # It also manages an internal IO_dict for student and teacher.
        # For GCL, student_io_dict and teacher_io_dict are built internally using hooks.
        # The call should be: loss = self.criterion(student_output_for_loss, teacher_output_for_loss, targets_for_loss)
        # With GCL, it's simpler:
        loss = self.criterion(student_output, teacher_output, targets)
        return loss

    def post_forward_process(self, loss, **kwargs):
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            if self.accelerator is not None: # Not used here
                self.accelerator.backward(loss)
            else:
                loss.backward()
            
            # Optional: Gradient clipping
            # if self.current_phase_config.get("grad_clip_norm"):
            #     torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.current_phase_config["grad_clip_norm"])

            self.optimizer.step()

        # Step-wise scheduler (if any, most are epoch-wise)
        # if self.lr_scheduler and self.current_phase_config.get("scheduler_step_at", "epoch") == "step":
        #     self.lr_scheduler.step()


    def post_epoch_process(self, epoch, metrics=None, **kwargs):
        """
        epoch: global epoch number
        metrics: dictionary of validation metrics for this epoch
        """
        if self.lr_scheduler:
            scheduler_step_at = self.current_phase_config.get("scheduler", {}).get("step_at", "epoch")
            if scheduler_step_at == "epoch":
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if metrics is None: # Check if primary metric is available
                         logger.warning("ReduceLROnPlateau scheduler needs metrics to step, but none provided.")
                    else:
                        self.lr_scheduler.step(metrics) # Expects a scalar metric
                else:
                    self.lr_scheduler.step()
        
        self.current_phase_epochs_done += 1
        
        # Handle entropy bottleneck update if specified for the current phase
        eb_update_epoch = self.current_phase_config.get("epoch_to_update_entropy_bottleneck", None)
        if eb_update_epoch is not None and self.current_phase_epochs_done >= eb_update_epoch:
             if hasattr(self.student_model, 'update') and callable(getattr(self.student_model, 'update')):
                logger.info(f"Updating student model (entropy bottleneck) at end of phase epoch {self.current_phase_epochs_done}")
                self.student_model.update(force=True) # force=True typically re-estimates CDFs
             # Reset or mark as updated to avoid repeated calls if eb_update_epoch is not precise
             # This logic might need refinement based on how often .update() should be called.
             # For now, assume it's called once when current_phase_epochs_done >= eb_update_epoch.
             # If it should be called *every* epoch *after* eb_update_epoch, remove this line:
             self.current_phase_config["epoch_to_update_entropy_bottleneck"] = None # Mark as done for this phase

    def clean_modules(self):
        # Placeholder if any cleanup is needed
        pass