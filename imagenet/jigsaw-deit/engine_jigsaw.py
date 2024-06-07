# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
from collections import defaultdict

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import torch.nn.functional as F

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
                    
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs.sup, targets)
            if args.use_jigsaw:
                loss_jigsaw = F.cross_entropy(outputs.pred_jigsaw, outputs.gt_jigsaw) * args.lambda_jigsaw
                loss += loss_jigsaw

        loss_value = loss.item()
        loss_jigsaw_value = loss_jigsaw.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss_total=loss_value)
        metric_logger.update(loss_jigsaw=loss_jigsaw_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    # initialize dictionaries to hold predictions and labels for each class
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # to store predictions and actual labels
    all_predictions = []
    all_labels = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output.sup, target)

        acc1, acc5 = accuracy(output.sup, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        # store predictions and labels for class accuracy calculation
        _, predictions = torch.max(output.sup, 1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    for label, pred in zip(all_labels, all_predictions):
        # compute total correct predictions and total predictions
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1

    class_accuracies = [
        class_correct[cls] / class_total[cls] for cls in class_total]

    print("Class average accuracy: " + str(
                        100 * sum(class_accuracies) / len(class_accuracies)))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, 
                  losses=metric_logger.loss))

    return_keys = {k: meter.global_avg for
                   k, meter in metric_logger.meters.items()}

    return_keys["class_avg_acc"] = 100 * sum(class_accuracies) / len(class_accuracies)

    return return_keys
