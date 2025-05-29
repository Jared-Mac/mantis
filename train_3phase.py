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
import yaml # For saving config to checkpoint

# Ensure project root is in sys.path for custom module imports
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Import custom modules after path setup
    from misc.datasets.registry import get_dataset 
    from misc.eval import get_eval_metric 
    import misc.loss # noqa F401, To register custom losses
    import model.network # noqa F401, To register custom models/networks
    import model.modules.timm_models # noqa F401, to register timm model getters
    import model.modules.compressor # noqa F401, to register compression modules

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
    load_ckpt 
)
from torchdistill.datasets import util as dataset_util 
from torchdistill.eval.classification import compute_accuracy # Example metric
from torchdistill.misc.log import setup_log_file, SmoothedValue, MetricLogger
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
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch (global)")
    parser.add_argument("--current_phase", default=1, type=int, help="Phase to start/resume from")
    parser.add_argument("--epochs_done_in_current_phase", default=0, type=int, help="Epochs already done in current_phase if resuming")

    parser.add_argument("--seed", type=int, default=42, help="seed in random number generator")
    parser.add_argument("--test_only", action="store_true", help="only test the models")
    parser.add_argument(
        "--student_model_ckpt", help="student model checkpoint file path to resume training"
    )
    parser.add_argument("--output_dir", default="output", help="path to save outputs")
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    return parser

def load_datasets_main(dataset_config): # Renamed to avoid clash if you import from train_3phase itself
    logger.info("Loading datasets from main script")
    dataset_dict = {}
    for key, item_config in dataset_config.items():
        logger.info(f"Loading dataset spec: {key}")
        if 'splits' in item_config: # Wrappers like LabelChunkedTaskDataset
            for split_name, split_data_config in item_config['splits'].items():
                dataset_id = split_data_config['dataset_id']
                # The get_dataset in misc.datasets.registry should handle the full item_config
                # or be adapted to take split_data_config + item_config['type']
                # For now, construct the config get_dataset expects:
                current_split_config = {
                    'type': item_config['type'], # e.g. LabelChunkedTaskDataset
                    'params': split_data_config['params'] # Params for this split
                }
                 # Handle transforms specified at the item_config level for the wrapper
                if 'transform_params' in item_config: # This applies to the output of LabelChunked wrapper
                    current_split_config['transform_params'] = item_config['transform_params'].get(split_name)

                logger.info(f"  Instantiating split: {dataset_id} using type: {item_config['type']}")
                dataset_dict[dataset_id] = get_dataset(current_split_config)
                logger.info(f"  Loaded {dataset_id} ({type(dataset_dict[dataset_id])}) with {len(dataset_dict[dataset_id]) if hasattr(dataset_dict[dataset_id], '__len__') else 'iterable'} samples.")

        else: # Direct dataset definitions like CIFAR100_original
            dataset_id = item_config.get('dataset_id', key)
            # Pass the whole item_config for direct datasets
            dataset_dict[dataset_id] = get_dataset(item_config)
            logger.info(f"  Loaded {dataset_id} ({type(dataset_dict[dataset_id])}) with {len(dataset_dict[dataset_id]) if hasattr(dataset_dict[dataset_id], '__len__') else 'iterable'} samples.")
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
    init_distributed_mode(world_size=args.world_size, dist_url=args.dist_url)
    if args.seed is not None: set_seed(args.seed)

    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    
    log_file_path = args.log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    device = torch.device(args.device)
    
    dataset_config = config["datasets"]
    dataset_dict = load_datasets_main(dataset_config)

    student_model_config = config["models"]["student_model"]
    # Dynamically set output_cond_signal_dim for TaskProbabilityModel if dataset provides info
    # This requires the dataset to be loaded first and to have a method like get_task_info()
    # Example: (Assuming cifar100_5tasks_chunked/train is the primary task dataset)
    try:
        main_task_dataset_train_id = config["train"]["phase2"]["train_data_loader"]["dataset_id"] # Example path
        if main_task_dataset_train_id in dataset_dict and hasattr(dataset_dict[main_task_dataset_train_id], 'get_task_info'):
            task_info = dataset_dict[main_task_dataset_train_id].get_task_info()
            num_distinct_tasks = task_info.get("num_distinct_task_chunks_for_predictor")
            if num_distinct_tasks is not None:
                student_model_config["params"]["task_probability_model_config"]["params"]["output_cond_signal_dim"] = num_distinct_tasks
                logger.info(f"Dynamically set task_probability_model output_cond_signal_dim to: {num_distinct_tasks}")
    except KeyError as e:
        logger.warning(f"Could not find dataset or task info for dynamic task_probability_model_config: {e}. Using YAML value.")


    student_model = get_torchdistill_model(student_model_config['name'], **student_model_config['params'])
    student_model = student_model.to(device)

    teacher_model_p1 = None # Teacher model only for Phase 1 typically
    if "teacher_model_phase1" in config["models"]:
        teacher_model_config_p1 = config["models"]["teacher_model_phase1"]
        teacher_model_p1 = get_torchdistill_model(teacher_model_config_p1['name'], **teacher_model_config_p1['params'])
        if 'ckpt' in teacher_model_config_p1 and teacher_model_config_p1['ckpt'] and os.path.exists(teacher_model_config_p1['ckpt']):
            logger.info(f"Loading teacher checkpoint for phase 1 from: {teacher_model_config_p1['ckpt']}")
            load_ckpt(teacher_model_config_p1['ckpt'], model=teacher_model_p1, strict=True)
        elif 'ckpt' in teacher_model_config_p1 and teacher_model_config_p1['ckpt']:
             logger.warning(f"Teacher checkpoint for phase 1 specified but not found: {teacher_model_config_p1['ckpt']}")
        teacher_model_p1 = teacher_model_p1.to(device)
        for param in teacher_model_p1.parameters(): param.requires_grad = False
        teacher_model_p1.eval()
    
    start_epoch_global = args.start_epoch
    initial_phase_to_start = args.current_phase
    initial_epochs_done_in_phase = args.epochs_done_in_current_phase

    if args.student_model_ckpt:
        logger.info(f"Loading student model checkpoint from: {args.student_model_ckpt}")
        # load_ckpt might return more info like epoch, which could override args if needed
        ckpt_data = load_ckpt(args.student_model_ckpt, model=student_model, strict=False) # strict=False recommended for phased training
        if ckpt_data: # Assuming load_ckpt returns a dict if successful
            # Try to resume phase and epoch from checkpoint if not overridden by args
            if args.start_epoch == 0 and 'epoch' in ckpt_data : # global epoch
                 start_epoch_global = ckpt_data['epoch'] # This is global epoch
                 logger.info(f"Resuming global epoch from checkpoint: {start_epoch_global}")
            if args.current_phase == 1 and 'current_phase' in ckpt_data:
                 initial_phase_to_start = ckpt_data['current_phase']
                 logger.info(f"Resuming phase from checkpoint: {initial_phase_to_start}")
            # epochs_done_in_current_phase might be harder to get directly unless explicitly saved in that way
            # The PhasedTrainingBox will need to reconcile global_epoch with phase-specific epochs.


    phased_box = PhasedTrainingBox(
        teacher_model=teacher_model_p1, # Initial teacher, box can override later
        student_model=student_model,
        dataset_dict=dataset_dict,
        train_config=config["train"],
        device=device,
        device_ids=[0] if device.type == 'cuda' and args.world_size == 1 else None,
        distributed=args.world_size > 1,
        accelerator=None,
        initial_phase_num_to_start=initial_phase_to_start, # For resuming
        initial_epochs_done_in_phase=initial_epochs_done_in_phase # For resuming
    )
    
    if args.test_only:
        # ... (test_only logic as before) ...
        return

    logger.info("Start training")
    start_time = time.time()
    
    # The main loop iterates through GLOBAL epochs
    for epoch in range(start_epoch_global, phased_box.num_epochs):
        current_phase_key = phased_box.current_phase_key # PhasedBox manages this
        current_phase_config = phased_box.current_phase_config
        
        train_loader = phased_box.train_data_loader
        val_loader = phased_box.val_data_loader
        
        if args.world_size > 1 and hasattr(train_loader.sampler, 'set_epoch'):
            # Sampler usually expects the current epoch number (0 to N-1) for its internal shuffling logic
            train_loader.sampler.set_epoch(epoch) 

        phased_box.pre_epoch_process(epoch=epoch) 
        
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value:.0f}"))

        header = f"Phase: {phased_box.stage_number} (Epoch {phased_box.current_phase_epochs_done + 1}/{current_phase_config['num_epochs']}) Global Epoch: [{epoch + 1}/{phased_box.num_epochs}]"
        for images, targets, supp_dict in metric_logger.log_every(train_loader, config["train"]["log_freq"], header):
            start_iter_time = time.time()
            images, targets, supp_dict = send_to_device(device, images, targets, supp_dict)
            
            loss = phased_box.forward_process(sample_batch=images, targets=targets, supp_dict=supp_dict)
            
            phased_box.post_forward_process(loss) 
            
            metric_logger.update(loss=loss.item(), lr=phased_box.optimizer.param_groups[0]["lr"])
            metric_logger.meters["img/s"].update(images.size(0) / (time.time() - start_iter_time))

        epoch_val_metrics = {}
        if val_loader:
            eval_header = f"Validation Phase: {phased_box.stage_number} (Epoch {phased_box.current_phase_epochs_done + 1}) Global Epoch: [{epoch + 1}]"
            val_metric_logger = MetricLogger(delimiter="  ")
            
            student_model_eval = phased_box.student_model # student_model is already DDP-wrapped by the box if distributed
            if hasattr(student_model_eval, 'eval'): student_model_eval.eval()
            
            if phased_box.teacher_model and hasattr(phased_box.teacher_model, 'eval'): 
                phased_box.teacher_model.eval()

            current_eval_metrics_config = current_phase_config.get("eval_metrics", [])
            
            with torch.no_grad():
                for images, targets, supp_dict in val_metric_logger.log_every(val_loader, config["train"]["log_freq"], eval_header):
                    images, targets, supp_dict = send_to_device(device, images, targets, supp_dict)
                    
                    if 'accuracy' in current_eval_metrics_config :
                        student_output_val_dict = student_model_eval(images)
                        
                        if isinstance(student_output_val_dict, dict) and 'main_output' in student_output_val_dict:
                                student_logits_val = student_output_val_dict['main_output']
                        else: 
                                student_logits_val = student_output_val_dict

                        main_labels_val = targets[0] if isinstance(targets, (list,tuple)) else targets
                        acc1, acc5 = compute_accuracy(student_logits_val, main_labels_val, topk=(1, 5))
                        val_metric_logger.meters['acc1'].update(acc1.item(), n=images.size(0))
                        val_metric_logger.meters['acc5'].update(acc5.item(), n=images.size(0))
            
            if args.world_size > 1 and dist.is_initialized(): # Ensure dist is initialized
                val_metric_logger.synchronize_between_processes()

            if 'accuracy' in current_eval_metrics_config:
                logger.info(f"{eval_header} Acc@1: {val_metric_logger.acc1.global_avg:.3f} Acc@5: {val_metric_logger.acc5.global_avg:.3f}")
                epoch_val_metrics['acc1'] = val_metric_logger.acc1.global_avg
        
        metric_for_scheduler = epoch_val_metrics.get('acc1') # Pass acc1 for ReduceLROnPlateau
        phased_box.post_epoch_process(epoch=epoch, metrics=metric_for_scheduler) 
        
        if args.output_dir and is_main_process():
            ckpt_path = os.path.join(args.output_dir, f"phased_ckpt_epoch_global{epoch+1}_phase{phased_box.stage_number}.pt")
            file_util.make_parent_dirs(ckpt_path) 
            current_student_model_state = phased_box.student_model.module.state_dict() \
                if module_util.check_if_wrapped(phased_box.student_model) \
                else phased_box.student_model.state_dict()
            
            save_on_master(
                {
                    "epoch": epoch + 1, # Global epoch
                    "current_phase": phased_box.stage_number,
                    "epochs_done_in_current_phase": phased_box.current_phase_epochs_done, # After increment in post_epoch_process
                    "student_model": current_student_model_state,
                    "optimizer": phased_box.optimizer.state_dict() if phased_box.optimizer else None,
                    "lr_scheduler": phased_box.lr_scheduler.state_dict() if phased_box.lr_scheduler else None,
                    "config_yaml_str": yaml.dump(config), 
                    "args": vars(args), 
                    "best_val_metric": epoch_val_metrics.get('acc1', -1) 
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved to {ckpt_path}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if not os.path.exists(args.output_dir) and is_main_process():
        os.makedirs(args.output_dir)
    main(args)