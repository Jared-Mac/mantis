# train_3phase.py
import argparse
import datetime
import os
import sys
import time
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Ensure project root is in sys.path for custom module imports
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Project root added to sys.path: {project_root}")
    # Import custom modules after path setup
    from misc.util import load_model # Assuming this is your custom model loader
    from misc.datasets.registry import get_dataset # Your custom dataset registry
    from misc.eval import get_eval_metric # Your eval metrics
    from misc.loss import MultiLabelTaskRelevancyBCELoss, BppLossOrig # Register custom losses

    # Import PhasedTrainingBox
    from phased_training_box import PhasedTrainingBox

except ImportError as e:
    print(f"Error importing custom modules: {e}. Check sys.path and module locations.")
    sys.exit(1)


from torchdistill.common import file_util, yaml_util, module_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import (
    is_main_process,
    init_distributed_mode,
    set_seed,
    save_on_master,
    load_ckpt # torchdistill's load_ckpt
)
from torchdistill.core.distillation import get_distillation_box # Will use our custom box
from torchdistill.core.training import get_training_box
from torchdistill.datasets import util
from torchdistill.eval.classification import compute_accuracy
from torchdistill.misc.log import setup_log_file, SmoothedValue, MetricLogger
from torchdistill.models.official import get_vision_model
from torchdistill.models.registry import get_model as get_torchdistill_model # To avoid clash

logger = def_logger.getChild(__name__)

def get_parser():
    parser = argparse.ArgumentParser(description="Phased Training with torchdistill")
    parser.add_argument(
        "--config",
        default="config/mantis/cifar-100/resnet18_3phase.yaml",
        help="YAML config file path",
    )
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--log", help="log file path")
    parser.add_argument("--run_log_dir", help="TensorBoard log directory path", default="runs")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--seed", type=int, default=42, help="seed in random number generator")
    parser.add_argument("--test_only", action="store_true", help="only test the models")
    parser.add_argument(
        "--student_model_ckpt", help="student model checkpoint file path to resume training"
    )
    parser.add_argument("--output_dir", default="output", help="path to save outputs")
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    return parser

def load_datasets(dataset_config):
    logger.info("Loading datasets")
    dataset_dict = {}
    for key, item_config in dataset_config.items():
        logger.info(f"Loading dataset: {key}")
        # item_config is {'name': 'unique_dataset_name', 'type': 'DatasetClass', 'splits': {...}}
        # We need to create instances for each split.
        if 'splits' in item_config:
            for split_name, split_data_config in item_config['splits'].items():
                dataset_id = split_data_config['dataset_id']
                # Construct the config for get_dataset
                # Ensure 'type' is at the top level for get_dataset to find it.
                dataset_instance_config = {
                    'type': item_config['type'], # e.g., 'LabelChunkedTaskDataset'
                    'params': split_data_config['params']
                }
                logger.info(f"  Instantiating split: {dataset_id} using type: {item_config['type']}")
                dataset_dict[dataset_id] = get_dataset(dataset_instance_config)
                logger.info(f"  Loaded {dataset_id} with {len(dataset_dict[dataset_id]) if hasattr(dataset_dict[dataset_id], '__len__') else 'iterable'} samples.")
        else: # Direct dataset definition without splits (e.g. cifar100_original)
            dataset_id = item_config.get('dataset_id', key) # Use key as id if not specified
            dataset_instance_config = {
                'type': item_config['type'],
                'params': item_config.get('params', {})
            }
            # Add transform parsing if it's under transform_params directly
            if 'transform_params' in item_config:
                 dataset_instance_config['params']['transform'] = util.build_transform(item_config)

            dataset_dict[dataset_id] = get_dataset(dataset_instance_config)
            logger.info(f"  Loaded {dataset_id} with {len(dataset_dict[dataset_id]) if hasattr(dataset_dict[dataset_id], '__len__') else 'iterable'} samples.")


    return dataset_dict

def send_to_device(device, images, targets, supp_dict=None):
    images = images.to(device, non_blocking=True)
    if isinstance(targets, (list, tuple)):
        targets = [t.to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in targets]
    elif isinstance(targets, torch.Tensor):
        targets = targets.to(device, non_blocking=True)
    
    if supp_dict:
        for k, v in supp_dict.items():
            if isinstance(v, torch.Tensor):
                supp_dict[k] = v.to(device, non_blocking=True)
    return images, targets, supp_dict


def main(args):
    # Distributed mode setup
    init_distributed_mode(
        world_size=args.world_size, dist_url=args.dist_url
    )
    if args.seed is not None:
        set_seed(args.seed)

    # Config loading
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))

    # Logging setup
    log_file_path = args.log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    device = torch.device(args.device)
    
    # Dataset loading
    dataset_config = config["datasets"]
    dataset_dict = load_datasets(dataset_config)

    # Model loading
    student_model_config = config["models"]["student_model"]
    # Use your custom load_model or torchdistill's get_model
    student_model = get_torchdistill_model(student_model_config['name'], **student_model_config['params'])
    student_model = student_model.to(device)


    # For Phase 1, we need a teacher model
    teacher_model_config_p1 = config["models"].get("teacher_model_phase1", None)
    teacher_model_p1 = None
    if teacher_model_config_p1:
        teacher_model_p1 = get_torchdistill_model(teacher_model_config_p1['name'], **teacher_model_config_p1['params'])
        if 'ckpt' in teacher_model_config_p1 and teacher_model_config_p1['ckpt']:
            load_ckpt(teacher_model_config_p1['ckpt'], model=teacher_model_p1, strict=True)
        teacher_model_p1 = teacher_model_p1.to(device)
        for param in teacher_model_p1.parameters(): # Freeze teacher
            param.requires_grad = False
        teacher_model_p1.eval()


    if args.student_model_ckpt:
        load_ckpt(args.student_model_ckpt, model=student_model, strict=False) # strict=False if phases change model structure

    # PhasedTrainingBox instantiation
    # Note: The teacher_model argument here is for the *first* phase.
    # The box will handle teacher setup for subsequent phases based on YAML.
    phased_box = PhasedTrainingBox(
        teacher_model=teacher_model_p1, # This is for Phase 1
        student_model=student_model,
        dataset_dict=dataset_dict,
        train_config=config["train"], # Pass the full multi-phase train config
        device=device,
        device_ids=[0] if device.type == 'cuda' else None, # Adjust if using multiple GPUs non-distributed
        distributed=args.world_size > 1,
        lr_factor=1.0, # Assuming base LRs are in YAML
        accelerator=None # Not using Hugging Face Accelerate for now
    )
    
    # Test only mode
    if args.test_only:
        test_data_loader_config = config["test"]["test_data_loader"]
        # Build test data loader using torchdistill's utility
        # dataset_id must match one defined in dataset_dict
        test_data_loader, _ = util.build_data_loaders(
            dataset_dict, [test_data_loader_config], distributed=args.world_size > 1
        )
        # Implement a separate test evaluation function if needed
        # For now, let's assume validation logic can be reused or adapted
        logger.info("Test_only mode. Skipping training.")
        # evaluate_final_model(phased_box, test_data_loader, device, config["test"]["eval_metrics"])
        return

    # Training loop
    logger.info("Start training")
    start_time = time.time()
    best_val_metric = -1 # Placeholder for a primary validation metric to track

    for epoch in range(args.start_epoch, phased_box.num_epochs):
        current_phase_key = f"{phased_box.stage_key_prefix}{phased_box.stage_number}"
        current_phase_config = config["train"][current_phase_key]
        
        # Get DataLoaders for the current phase from PhasedTrainingBox
        # PhasedTrainingBox.setup (called by __init__ and advance_to_next_stage)
        # configures self.train_data_loader and self.val_data_loader
        train_loader = phased_box.train_data_loader
        val_loader = phased_box.val_data_loader
        
        if args.world_size > 1 and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Pre-epoch process (e.g., set model to train mode)
        # Pass train_loader for DistributedSampler.set_epoch if PhasedTrainingBox needs it
        phased_box.pre_epoch_process(epoch=epoch, train_data_loader=train_loader) 
        
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

        header = f"Phase: {phased_box.stage_number} Epoch: [{epoch + 1}/{phased_box.num_epochs}]"
        for images, targets, supp_dict in metric_logger.log_every(train_loader, config["train"]["log_freq"], header):
            start_iter_time = time.time()
            images, targets, supp_dict = send_to_device(device, images, targets, supp_dict)
            
            # torchdistill's pre_forward_process is usually a pass
            # phased_box.pre_forward_process(images, targets, supp_dict) 
            
            loss = phased_box.forward_process(sample_batch=images, targets=targets, supp_dict=supp_dict)
            
            # post_forward_process handles loss.backward(), optimizer.step(), optimizer.zero_grad()
            # and step-wise scheduler updates if configured
            phased_box.post_forward_process(loss) 
            
            # Log metrics
            # Assuming loss is a scalar tensor. If it's a dict, need to sum or extract primary.
            metric_logger.update(loss=loss.item(), lr=phased_box.optimizer.param_groups[0]["lr"])
            metric_logger.meters["img/s"].update(images.size(0) / (time.time() - start_iter_time))

        # Validation
        if val_loader:
            # Example validation (adapt based on your eval_metrics in YAML)
            # This is a simplified accuracy evaluation. Your actual eval might be more complex.
            # And should use methods from misc.eval
            eval_header = f"Validation Phase: {phased_box.stage_number} Epoch: [{epoch + 1}]"
            val_metric_logger = MetricLogger(delimiter="  ")
            
            # Set model to eval mode (handled by PhasedTrainingBox methods if called)
            if hasattr(phased_box.student_model, 'eval'): phased_box.student_model.eval()
            if phased_box.teacher_model and hasattr(phased_box.teacher_model, 'eval'): phased_box.teacher_model.eval() # Should already be in eval

            current_eval_metrics_config = current_phase_config.get("eval_metrics", ['accuracy'])
            # Store results for ReduceLROnPlateau or best model saving
            epoch_val_metrics = {} 

            with torch.no_grad():
                for images, targets, supp_dict in val_metric_logger.log_every(val_loader, config["train"]["log_freq"], eval_header):
                    images, targets, supp_dict = send_to_device(device, images, targets, supp_dict)
                    
                    # In validation, forward_process might still compute losses if criterion is used for metrics.
                    # Or, you might just need the model's output for accuracy.
                    # For simplicity, let's assume student_model.forward() gives logits for accuracy.
                    
                    # If 'accuracy' is an expected metric
                    if 'accuracy' in current_eval_metrics_config :
                        # Get student output directly for eval. student_model is already wrapped by DDP if needed.
                        student_output_dict = phased_box.student_model(images) # Assuming this returns the dict
                        
                        # Ensure student_output_dict is the dict from model.network.SplittableNetworkWithSharedStem
                        if isinstance(student_output_dict, dict) and 'main_output' in student_output_dict:
                             student_logits = student_output_dict['main_output']
                        else: # Fallback if model directly returns logits
                             student_logits = student_output_dict

                        # Assuming targets[0] is the main label for classification
                        main_labels = targets[0] if isinstance(targets, (list,tuple)) else targets
                        acc1, acc5 = compute_accuracy(student_logits, main_labels, topk=(1, 5))
                        val_metric_logger.meters['acc1'].update(acc1.item(), n=images.size(0))
                        val_metric_logger.meters['acc5'].update(acc5.item(), n=images.size(0))
            
            # Synchronize validation metrics if distributed
            if args.world_size > 1:
                val_metric_logger.synchronize_between_processes()

            logger.info(f"{eval_header} Acc@1: {val_metric_logger.acc1.global_avg:.3f} Acc@5: {val_metric_logger.acc5.global_avg:.3f}")
            epoch_val_metrics['acc1'] = val_metric_logger.acc1.global_avg
            # Add other metrics to epoch_val_metrics as needed

        # Post-epoch process (e.g., LR scheduler step, advance to next phase)
        # Pass validation metrics for schedulers like ReduceLROnPlateau
        phased_box.post_epoch_process(metrics=epoch_val_metrics.get('acc1', None)) # Example: use acc1 for scheduler
        
        # Save checkpoint
        if args.output_dir and is_main_process():
            ckpt_path = os.path.join(args.output_dir, f"phased_ckpt_epoch{epoch+1}.pt")
            save_on_master(
                {
                    "epoch": epoch + 1,
                    "student_model": phased_box.student_model.state_dict(),
                    "optimizer": phased_box.optimizer.state_dict(),
                    "lr_scheduler": phased_box.lr_scheduler.state_dict() if phased_box.lr_scheduler else None,
                    "config": config, # Save config for reproducibility
                    "args": args,
                    "current_phase": phased_box.stage_number,
                    "current_phase_epoch_end": phased_box.stage_end_epoch
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved to {ckpt_path}")
        
        # Update best_val_metric and save best model if current is better
        # This logic needs to be adapted based on the primary metric for "best"
        # current_val_metric_for_best_tracking = epoch_val_metrics.get('acc1', -1)
        # if current_val_metric_for_best_tracking > best_val_metric:
        #     best_val_metric = current_val_metric_for_best_tracking
        #     if args.output_dir and is_main_process():
        #         save_on_master(phased_box.student_model.state_dict(), os.path.join(args.output_dir, "best_student_model.pt"))
        #         logger.info("Best model updated based on Acc@1.")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)