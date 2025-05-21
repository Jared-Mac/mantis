import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchdistill.common.constant import def_logger
from torchdistill.eval.classification import compute_accuracy
from torchdistill.misc.log import MetricLogger

from misc.analyzers import check_if_analyzable
from misc.util import check_if_module_exits, compute_bitrate, compute_psnr, extract_entropy_bottleneck_module
from model.modules.compressor import CompressionModule

logger = def_logger.getChild(__name__)


class EvaluationMetric:
    def __init__(self,
                 eval_func,
                 init_best_val,
                 comparator):
        self.eval_func = eval_func
        self.best_val = init_best_val
        self.comparator = comparator

    def compare_with_curr_best(self, result) -> bool:
        return self.comparator(self.best_val, result)


@torch.inference_mode()
def evaluate_psnr(model,
                  data_loader,
                  device,
                  base_model=None,
                  log_freq=1000,
                  title=None,
                  header='Test:',
                  **kwargs) -> float:
    model.to(device)
    if title is not None:
        logger.info(title)

    if base_model:
        model = model.compression_module
        logger.info("Evaluating PSNR between head representation and recon ")
        base_model.layers = base_model.layers[:2]
        base_model.head = nn.Identity()
        base_model.norm = nn.Identity()
        base_model.pos_drop = nn.Identity()
        base_model.forward_head = lambda x: x
    else:
        base_model = nn.Identity()
        logger.info("Evaluating PSNR between image and recon")

    model.to(device)

    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    for image, _ in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        orig_repr = base_model(image)
        recon = model(image)
        psnr = compute_psnr(recon, orig_repr)
        batch_size = image.shape[0]
        metric_logger.meters['psnr'].update(psnr.item(), n=batch_size)

    psnr = metric_logger.psnr.global_avg
    logger.info(' * PSNR {:.4f}'.format(psnr))
    return metric_logger.psnr.global_avg


@torch.inference_mode()
def evaluate_accuracy(model,
                      data_loader,
                      device,
                      device_ids=None,
                      distributed=False,
                      log_freq=1000,
                      title=None,
                      header='Test:',
                      no_dp_eval=True,
                      accelerator=None,
                      include_top_5=False,
                      pre_compressor=None,
                      **kwargs) -> float:
    model.to(device)
    if not no_dp_eval:
        if distributed:
            model = DistributedDataParallel(model, device_ids=device_ids)
        elif device.type.startswith('cuda'):
            model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if pre_compressor:
            image = pre_compressor(image)
        output = model(image)
        acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    logger.info(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    if include_top_5:
        return top1_accuracy, top5_accuracy
    return top1_accuracy


@torch.inference_mode()
def evaluate_bpp(model,
                 data_loader,
                 device,
                 device_ids=None,
                 distributed=False,
                 log_freq=1000,
                 title=None,
                 header='Test:',
                 no_dp_eval=True,
                 test_mode=False,
                 extract_bottleneck=True,
                 **kwargs) -> float:
    model.to(device)
    if not no_dp_eval:
        if distributed:
            model = DistributedDataParallel(model, device_ids=device_ids)
        elif device.type.startswith('cuda'):
            model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    analyzable = False
    if test_mode:
        if check_if_analyzable(model):
            model.activate_analysis()
            analyzable = True
            logger.info("Analysis for Compressed Size activated")
        else:
            logger.warning("Requesting analyzing compressed size but model is not analyzable")

    model.eval()
    bottleneck_module = extract_entropy_bottleneck_module(model)
    has_hyperprior = False
    has_dual_hyperprior = False
    if check_if_module_exits(bottleneck_module, 'gaussian_conditional'):
        has_hyperprior = True
    if check_if_module_exits(bottleneck_module, 'entropy_bottleneck_spat') or check_if_module_exits(bottleneck_module,
                                                                                                    'gaussian_conditional_2'):
        has_dual_hyperprior = True
        has_hyperprior = False
    metric_logger = MetricLogger(delimiter='  ')
    for image, _ in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        # todo, can delete just use sc2 repo directly
        if isinstance(bottleneck_module, CompressionModule):
            _, likelihoods = bottleneck_module(image, return_likelihoods=True)
        else:
            likelihoods = bottleneck_module(image)["likelihoods"]
        if has_dual_hyperprior:
            likelihoods_y, likelihoods_z_1, likelihoods_z_2 = likelihoods.values()
            bpp_z_1, _ = compute_bitrate(likelihoods_z_1, image.shape)
            bpp_z_2, _ = compute_bitrate(likelihoods_z_2, image.shape)
            bpp_y, _ = compute_bitrate(likelihoods_y, image.shape)
            metric_logger.meters['bpp_y'].update(bpp_y.item(), n=image.size(0))
            metric_logger.meters['bpp_z_1'].update(bpp_z_1.item(), n=image.size(0))
            metric_logger.meters['bpp_z_2'].update(bpp_z_2.item(), n=image.size(0))
            bpp = bpp_z_1 + bpp_z_2 + bpp_y
        elif has_hyperprior:
            likelihoods_y, likelihoods_z = likelihoods.values()
            bpp_z, _ = compute_bitrate(likelihoods_z, image.shape)
            bpp_y, _ = compute_bitrate(likelihoods_y, image.shape)
            metric_logger.meters['bpp_z'].update(bpp_z.item(), n=image.size(0))
            metric_logger.meters['bpp_y'].update(bpp_y.item(), n=image.size(0))
            bpp = bpp_z + bpp_y
        else:
            bpp, _ = compute_bitrate(likelihoods["y"], image.shape)
        if analyzable:
            model(image)
        metric_logger.meters['bpp'].update(bpp.item(), n=image.size(0))
    metric_logger.synchronize_between_processes()
    avg_bpp = metric_logger.bpp.global_avg
    logger.info(' * Bpp {:.5f}\n'.format(avg_bpp))
    if has_dual_hyperprior:
        avg_bpp_z_1 = metric_logger.bpp_z_1.global_avg
        avg_bpp_z_2 = metric_logger.bpp_z_2.global_avg
        avg_bpp_y = metric_logger.bpp_y.global_avg
        logger.info(' * Bpp_z_1 {:.5f}\n'.format(avg_bpp_z_1))
        logger.info(' * Bpp_z_2 {:.5f}\n'.format(avg_bpp_z_2))
        logger.info(' * Bpp_y {:.5f}\n'.format(avg_bpp_y))
    elif has_hyperprior:
        avg_bpp_z = metric_logger.bpp_z.global_avg
        avg_bpp_y = metric_logger.bpp_y.global_avg
        logger.info(' * Bpp_z {:.5f}\n'.format(avg_bpp_z))
        logger.info(' * Bpp_y {:.5f}\n'.format(avg_bpp_y))
    if analyzable:
        model.summarize()
        model.deactivate_analysis()
    return avg_bpp


@torch.inference_mode()
def evaluate_filesize_and_accuracy(model,
                                   data_loader,
                                   device,
                                   device_ids,
                                   distributed,
                                   log_freq=1000,
                                   title=None,
                                   header='Test:',
                                   no_dp_eval=True,
                                   test_mode=False,
                                   use_hnetwork=False,
                                   **kwargs) -> float:
    model.to(device)
    if not no_dp_eval:
        if distributed:
            model = DistributedDataParallel(model, device_ids=device_ids)
        elif device.type.startswith('cuda'):
            model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    analyzable = False
    if test_mode:
        if check_if_analyzable(model):
            model.activate_analysis()
            analyzable = True
        else:
            logger.warning("Requesting analyzing compressed size but model is not analyzable")
    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(image)
        acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    avg_filesize = None
    if analyzable:
        avg_filesize = model.summarize()[0]
        model.deactivate_analysis()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    logger.info(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    return metric_logger.acc1.global_avg, avg_filesize


EVAL_METRIC_DICT = {
    "accuracy": EvaluationMetric(eval_func=evaluate_accuracy,
                                 init_best_val=0,
                                 comparator=lambda curr_top_val, epoch_val: epoch_val > curr_top_val),
    "bpp": EvaluationMetric(eval_func=evaluate_bpp,
                            init_best_val=float("inf"),
                            comparator=lambda curr_top_val, epoch_val: epoch_val < curr_top_val),
    'psnr': EvaluationMetric(eval_func=evaluate_psnr,
                             init_best_val=float("-inf"),
                             comparator=lambda curr_top_val, epoch_val: epoch_val > curr_top_val),

    "accuracy-and-filesize": EvaluationMetric(eval_func=evaluate_filesize_and_accuracy,
                                              init_best_val=None,
                                              comparator=None
                                              )
}

# Registry for new stateful metric classes
_EVAL_METRIC_NAME_CLASS_DICT = dict()

def register_eval_metric_class(cls):
    metric_name = cls.__name__
    if metric_name in _EVAL_METRIC_NAME_CLASS_DICT:
        logger.warning(f"Evaluation metric class {metric_name} already registered. Overwriting.")
    _EVAL_METRIC_NAME_CLASS_DICT[metric_name] = cls
    return cls

@register_eval_metric_class
class TaskPredictionAccuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.name = 'task_pred_acc'

    def update(self, model_output, targets):
        conditioning_signal = model_output["conditioning_signal"]
        task_detector_targets_one_hot = targets[1]

        predicted_task_indices = torch.argmax(conditioning_signal, dim=1)
        true_task_indices = torch.argmax(task_detector_targets_one_hot, dim=1)
        
        self.correct += (predicted_task_indices == true_task_indices).sum().item()
        self.total += task_detector_targets_one_hot.size(0)

    def compute(self):
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0

    def get_metric_dict(self):
        return {self.name: self.compute()}

    # Make it behave somewhat like EvaluationMetric for the existing loop if needed for transition
    # This eval_func would be called by the existing loop in main_classification_torchdistill.py
    # It would need to run a full evaluation pass.
    def eval_func(self, model, data_loader, device, **kwargs):
        self.reset()
        model.eval()
        with torch.no_grad():
            for image, target_tuple in data_loader: # Assuming target_tuple is (task_specific_label, task_detector_target_one_hot)
                image = image.to(device, non_blocking=True)
                # Ensure targets are also moved to device if they are tensors
                processed_targets = []
                for t_item in target_tuple:
                    if isinstance(t_item, torch.Tensor):
                        processed_targets.append(t_item.to(device, non_blocking=True))
                    else:
                        processed_targets.append(t_item) # Keep non-tensors as is

                model_output_dict = model(image) 
                self.update(model_output_dict, tuple(processed_targets))
        return self.compute()


@register_eval_metric_class
class AccuracyForChunk:
    def __init__(self, chunk_id):
        self.chunk_id = chunk_id
        self.correct = 0
        self.total = 0
        self.name = f'acc_chunk_{chunk_id}'

    def update(self, model_output, targets):
        backbone_outputs = model_output["backbone_outputs"]
        task_specific_labels = targets[0]
        task_detector_targets_one_hot = targets[1]

        true_task_indices = torch.argmax(task_detector_targets_one_hot, dim=1)
        chunk_mask = (true_task_indices == self.chunk_id)
        
        if chunk_mask.sum().item() > 0:
            current_tail_output = backbone_outputs.get(f'tail_{self.chunk_id}')
            if current_tail_output is None:
                logger.warning(f"Output for tail_{self.chunk_id} not found in backbone_outputs.")
                return

            chunk_logits = current_tail_output[chunk_mask]
            chunk_true_labels = task_specific_labels[chunk_mask]
            
            if chunk_logits.nelement() > 0:
                predicted_labels_for_chunk = torch.argmax(chunk_logits, dim=1)
                self.correct += (predicted_labels_for_chunk == chunk_true_labels).sum().item()
                self.total += chunk_true_labels.size(0)

    def compute(self):
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0
    
    def get_metric_dict(self):
        return {self.name: self.compute()}

    # Make it behave somewhat like EvaluationMetric for the existing loop
    def eval_func(self, model, data_loader, device, **kwargs):
        self.reset()
        model.eval()
        with torch.no_grad():
            for image, target_tuple in data_loader:
                image = image.to(device, non_blocking=True)
                processed_targets = []
                for t_item in target_tuple:
                    if isinstance(t_item, torch.Tensor):
                        processed_targets.append(t_item.to(device, non_blocking=True))
                    else:
                        processed_targets.append(t_item)
                
                model_output_dict = model(image)
                self.update(model_output_dict, tuple(processed_targets))
        return self.compute()


def get_eval_metric(metric_config_key, metric_config_value=None, **kwargs_passed_from_main_loop):
    # metric_config_key: e.g., "accuracy_chunk0" (the key from YAML eval_metrics list of dicts)
    # metric_config_value: e.g., {"type": "AccuracyForChunk", "params": {"chunk_id": 0}} (the value for that key)
    # kwargs_passed_from_main_loop: In current main script, this is empty when get_eval_metric is called at metric setup.
    #                               But the eval_func it returns is called with many kwargs.

    if metric_config_value is None: 
        metric_type = metric_config_key
        constructor_params = {}
    else:
        metric_type = metric_config_value.get('type', metric_config_key)
        constructor_params = metric_config_value.get('params', {})

    # Check new stateful metrics first
    if metric_type in _EVAL_METRIC_NAME_CLASS_DICT:
        MetricClass = _EVAL_METRIC_NAME_CLASS_DICT[metric_type]
        # Instantiate the stateful metric object
        metric_instance = MetricClass(**constructor_params)
        # Wrap it in EvaluationMetric to fit the existing main loop structure for now.
        # The eval_func of the stateful metric will run its own loop.
        return EvaluationMetric(eval_func=metric_instance.eval_func, # Use the new eval_func from the metric class
                                init_best_val=0 if 'acc' in metric_type.lower() else float('inf'), # Basic heuristic
                                comparator=lambda curr, new: new > curr if 'acc' in metric_type.lower() else new < curr) # Basic
    
    # Fallback to existing function-based EvaluationMetric wrappers in this file
    if metric_type in EVAL_METRIC_DICT:
        # This returns the EvaluationMetric wrapper with its predefined eval_func
        return EVAL_METRIC_DICT[metric_type]

    # Fallback to torchdistill's default get_eval_metric
    try:
        from torchdistill.eval import get_eval_metric as get_torchdistill_eval_metric
        # This is tricky. torchdistill's get_eval_metric might return a stateful metric or a simple function.
        # For now, assume it returns something that can be wrapped or is already an EvaluationMetric-like object.
        # This path might need more robust handling if torchdistill metrics are stateful and different.
        td_metric = get_torchdistill_eval_metric(metric_type, **constructor_params)
        if hasattr(td_metric, 'eval_func'): # If it's already an EvaluationMetric-like object
            return td_metric
        elif callable(td_metric): # If it's a function, wrap it
             return EvaluationMetric(eval_func=td_metric,
                                    init_best_val=0, comparator=lambda x,y: y>x) # generic wrapper
        logger.warning(f"Torchdistill metric '{metric_type}' might not be fully compatible with current wrapper.")
        return td_metric # Return as is if unsure
    except ValueError:
        raise ValueError(f'Unsupported metric type: {metric_type} (also not found in custom or torchdistill registries)')


