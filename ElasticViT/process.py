# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import math
import operator
import time
import random
import torch
import torch.nn.functional as F
import numpy as np
from util import AverageMeter
from timm.utils.agc import adaptive_clip_grad
import torch.nn as nn
from torch.distributed import get_world_size, all_reduce, barrier

__all__ = ['train_one_epoch', 'validate', 'PerformanceScoreboard']

logger = logging.getLogger()


def bn_cal(model, train_loader, args, arch=None, num_batches=100, mixup_fn=None):
    model.eval()

    for _, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.training = True
            module.momentum = None

            module.reset_running_stats()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if batch_idx > num_batches:
            break

        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        if mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)

        model(inputs)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def eval_one_subnet(subnet, model, train_loader, val_loader, args, mixup_fn):
    if isinstance(subnet, str):
        subnet = model.module.arch_sampling(subnet)
    elif isinstance(subnet, (list, tuple)):
        model.module.set_arch(*subnet)
    else:
        raise NotImplementedError
    
    logger.info(f"Evaluating subnet: {subnet}")

    with torch.no_grad():
        logger.info("start batch-norm layer calibration...")
        bn_cal(model, train_loader, args, num_batches=64 *
               (256//args.dataloader.batch_size), mixup_fn=mixup_fn)
        logger.info("finish batch-norm layer calibration...")
    acc1_val, _, _ = validate(val_loader, model, None, 0, None, args, None, )
    return round(acc1_val, 2)


def update_meter(meter, loss, acc1, acc5, size, batch_time, world_size):

    barrier()
    r_loss = loss.clone().detach()
    all_reduce(r_loss)
    r_loss /= world_size

    meter['loss'].update(r_loss.item(), size)
    meter['batch_time'].update(batch_time)


def teacher_inference(teacher_model, inputs, T=0.2):
    with torch.no_grad():
        if isinstance(teacher_model, list):
            teacher_outputs_0 = (teacher_model[0](inputs) / T).softmax(dim=-1)
            teacher_outputs_1 = (teacher_model[1](inputs) / T).softmax(dim=-1)
            teacher_outputs = teacher_outputs_0 / 2. + teacher_outputs_1 / 2.
        else:
            teacher_outputs = teacher_model(inputs).softmax(dim=-1)

    return teacher_outputs


def get_bank_id(current_bank_id, bank_nums):
    last_bank_id = current_bank_id
    if last_bank_id == 0:
        next_step = [0, 0, 1]
    elif last_bank_id == bank_nums-1:
        next_step = [-1, 0, 0]
  #  elif last_bank_id==bank_nums-3 or last_bank_id==2:
  #          next_step=[-1, 0, 1, 0]
    else:
        next_step = [-1, 0, 1]

    bank_id = random.choice(next_step) + last_bank_id
    return bank_id


def compute_dist_loss(labels, outputs, teacher_outputs, criterion, distill_criterion, multi_teachers=False, ALPHA=.5):
    distill_loss = distill_criterion(outputs, teacher_outputs)

    if multi_teachers:
        return distill_loss
    else:
        return ALPHA * criterion(outputs, labels) + (1-ALPHA) * distill_loss


def train_one_epoch(train_loader, model, criterion, optimizer, lr_scheduler, epoch, monitors, args, modes, distillation_loss, teacher_model,
                    inplace_distillation_loss, mixup_fn, model_ema, record_one_epoch=False, hard_distillation=False, force_random=False, current_bank_id=0, direction=0, use_latency=False):
    meters = [{
        'loss': AverageMeter(),
        'batch_time': AverageMeter()
    } for _ in range(4)]

    total_sample = len(train_loader.sampler)
    batch_size = args.dataloader.batch_size

    steps_per_epoch = math.ceil(total_sample / batch_size)
    steps_per_epoch = torch.tensor(steps_per_epoch).to(args.device)
    all_reduce(steps_per_epoch)
    steps_per_epoch = int(steps_per_epoch.item() // get_world_size())

    if args.rank == 0:
        logger.info('Training: %d samples (%d per mini-batch)',
                    total_sample, batch_size)

    num_updates = epoch * steps_per_epoch
    seed = num_updates
    model.train()
    model_without_ddp = model.module
    flops_sampling_method = model_without_ddp.flops_sampling_method
    bank_nums = len(model_without_ddp.bank_flops_ranges)

    bank_update_steps = steps_per_epoch//4

    min_flops = getattr(args.search_space, 'min_flops', 0)
    limit_flops = getattr(args.search_space, 'limit_flops', 0)

    if teacher_model:
        if isinstance(teacher_model, list):
            for tm in teacher_model:
                tm.eval()
        else:
            teacher_model.eval()

    model_without_ddp.set_regularization_mode(head_drop_prob=0, module_drop_prob=0)  # according to NASVit
    biggest_outputs, teacher_outputs = None, None
    multi_teachers = isinstance(teacher_model, list)

    for batch_idx, (original_inputs, original_targets) in enumerate(train_loader):
        original_inputs = original_inputs.to(args.device)
        original_targets = original_targets.to(args.device)

        optimizer.zero_grad()
        seed = seed + 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if mixup_fn is not None:
            inputs, targets = original_inputs.clone(), original_targets.clone()
            inputs, targets = mixup_fn(inputs, targets)
        else:
            inputs, targets = original_inputs, original_targets

        if batch_idx == 0 and limit_flops != min_flops:
            model_without_ddp.bank_loss_update(mini_batch=(
                original_inputs, original_targets), world_size=args.world_size)

        if teacher_model is not None:
            teacher_outputs = teacher_inference(
                teacher_model=teacher_model, inputs=inputs)

        uniform_idx = 0  # for cyclic training
        uniform_logits_buffer, flops_list = [], []

        if flops_sampling_method == 'adjacent_step':
            current_bank_id = get_bank_id(current_bank_id, bank_nums)
            target_flops = model_without_ddp.bank_flops_ranges[current_bank_id]

        for idx, mode in enumerate(modes):
            start_time = time.time()
          #  print('step',idx,batch_idx,mode,current_bank_id)
            # note: here we turn off the bank
            arch = model_without_ddp.arch_sampling(mode=mode, random_subnet_idx=uniform_idx,
                                                   force_random=force_random, flops_list=flops_list, current_bank_id=current_bank_id)
            # if not use_latency:
            flops = model_without_ddp.compute_flops(arch=arch)

            # print('current mode',mode,flops)
            if mode == 'uniform':
                flops_list.append(flops)

            outputs = model(inputs)

            loss = None
            if (len(modes) == 1 or (len(modes) > 1 and mode == 'max')) and teacher_model is not None:
                loss = compute_dist_loss(labels=targets, outputs=outputs, teacher_outputs=teacher_outputs, criterion=criterion, distill_criterion=distillation_loss,
                                            multi_teachers=multi_teachers)

            if args.inplace_distillation:
                arch_from_bank = False

                if mode == 'max':
                    with torch.no_grad():
                        if not hard_distillation:
                            biggest_outputs = outputs.clone().detach().softmax(dim=-1)
                        else:
                            biggest_outputs, biggest_outputs_kd = outputs
                            biggest_outputs = biggest_outputs.clone().detach().softmax(dim=-1)
                            biggest_outputs_kd = biggest_outputs_kd.clone().detach().softmax(dim=-1)

                elif mode == 'uniform':
                    update_bank_loss = (
                        batch_idx + 1) % bank_update_steps == 0 and (batch_idx + 1) != bank_update_steps * 4

                    if limit_flops != min_flops:
                        # print('update bank arch',arch)
                        model_without_ddp.bank_arch_update(arch, mini_batch=(
                            original_inputs, original_targets), world_size=args.world_size, update_bank=update_bank_loss)
                    arch_from_bank = model_without_ddp.bank_sampling

                    if 'max' not in modes:
                        with torch.no_grad():
                            soft_logits = outputs.clone().detach()
                            uniform_logits_buffer.append(
                                soft_logits.unsqueeze(dim=2).softmax(dim=1))

                        soft_targets = teacher_outputs
                        loss = compute_dist_loss(labels=targets, outputs=outputs, teacher_outputs=soft_targets, criterion=criterion, distill_criterion=distillation_loss,
                                                 multi_teachers=multi_teachers)
                    else:
                        loss = inplace_distillation_loss(outputs, biggest_outputs)
                    uniform_idx += 1

                elif mode == 'min':
                    soft_targets_min = biggest_outputs if 'max' in modes else torch.cat(
                        uniform_logits_buffer, dim=2).mean(dim=2)
                    loss = inplace_distillation_loss(
                        outputs, soft_targets_min)
                else:
                    raise NotImplementedError

            if loss is None:
                raise NotImplementedError

            loss.backward()
            update_meter(meters[idx], loss, None, None, inputs.size(
                0), time.time() - start_time, args.world_size)

        adaptive_clip_grad(model.parameters(), 0.1, norm_type=2.0)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_value)
        optimizer.step()

        num_updates += 1

        if lr_scheduler is not None:
            lr_scheduler.step_update(
                num_updates=num_updates, metric=meters[-1]['loss'].avg)

        torch.cuda.synchronize()

        if model_ema is not None:
            model_ema.update(model)

        if args.rank == 0 and (batch_idx + 1) % args.log.print_freq == 0:
            for m in monitors:
                for idx, mode in enumerate(modes):
                    m.update(epoch, batch_idx + 1, steps_per_epoch, "Mode " + mode + ' Training', {
                        'Loss': meters[idx]['loss'],
                        'BatchTime': meters[idx]['batch_time'],
                        'LR': optimizer.param_groups[0]['lr'],
                        'GPU memory': round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
                    })
            logger.info(
                "--------------------------------------------------------------------------------------------------------------")
    if args.rank == 0:
        for idx, mode in enumerate(modes):
            logger.info('==> Mode [%s] Loss: %.3f',
                        mode, meters[idx]['loss'].avg)
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    if 'top1' in meters[-1].keys():
        return current_bank_id, direction, meters[-1]['top1'].avg, meters[-1]['top5'].avg, meters[-1]['loss'].avg
    else:
        return current_bank_id, direction, meters[-1]['loss'].avg


def validate(data_loader, model, criterion, epoch, monitors, args, modes=['max', 'uniform', 'uniform', 'min']):

    if modes is not None:
        meters = [{
            'loss': AverageMeter(),
            'top1': AverageMeter(),
            'top5': AverageMeter(),
            'batch_time': AverageMeter()
        } for _ in range(len(modes))]
    else:
        meter = {
            'loss': AverageMeter(),
            'top1': AverageMeter(),
            'top5': AverageMeter(),
            'batch_time': AverageMeter()
        }

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    # if args.rank == 0:
    #     logger.info('Validation: %d samples (%d per mini-batch)',
    #                 total_sample, batch_size)

    model.eval()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            inputs = inputs.cuda()
            targets = targets.cuda()
            start_time = time.time()

            if modes is not None:
                for idx, mode in enumerate(modes):
                    model.module.arch_sampling(mode=mode)
                    outputs = model(inputs)
                    if criterion is not None:
                        loss = criterion(outputs, targets)

                    acc1, acc5 = accuracy(
                        outputs.data, targets.data, topk=(1, 5))
                    update_meter(meters[idx], loss, acc1, acc5, inputs.size(
                        0), time.time() - start_time, args.world_size)
            else:
                outputs = model(inputs)
                acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                meter['top1'].update(acc1.item(), inputs.size(0))
                meter['top5'].update(acc1.item(), inputs.size(0))
                # update_meter(meter, 0, acc1, acc5, inputs.size(0), time.time() - start_time, args.world_size, a=True)
            
            if criterion is not None:
                if args.rank == 0:
                    if (batch_idx + 1) % args.log.print_freq == 0:
                        for m in monitors:
                            for idx, mode in enumerate(modes):
                                m.update(epoch, batch_idx + 1, steps_per_epoch, 'Mode ' + mode + ' Val', {
                                    'Loss': meters[idx]['loss'],
                                    'Top1': meters[idx]['top1'],
                                    'Top5': meters[idx]['top5'],
                                    'BatchTime': meters[idx]['batch_time'],
                                })
                            logger.info(
                                "--------------------------------------------------------------------------------------------------------------")

    if criterion is not None:
        if args.rank == 0:
            for idx, mode in enumerate(modes):
                logger.info('==> Mode [%s] Top1: %.3f    Top5: %.3f    Loss: %.3f', mode,
                            meters[idx]['top1'].avg, meters[idx]['top5'].avg, meters[idx]['loss'].avg)
        return meters[-1]['top1'].max, meters[-1]['top5'].max, meters[-1]['loss'].avg
    else:
        return meter['top1'].avg, meter['top5'].avg, meter['loss'].avg


def validate_single(data_loader, model, criterion, epoch, monitors, args):
    meters = [{
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'top5': AverageMeter(),
        'batch_time': AverageMeter()
    } for _ in range(1)]

    modes = ['subnet']

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            start_time = time.time()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            print(acc1)
            update_meter(meters[0], loss, acc1, acc5, inputs.size(
                0), time.time() - start_time, args.world_size)

    return meters[-1]['top1'].avg, meters[-1]['top5'].avg, meters[-1]['loss'].avg


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch
