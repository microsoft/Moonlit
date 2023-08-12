# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from modules.modeling.ops.lsq_plus import set_quant_mode
import modules.alphanet_training.utils.comm as comm

from .imagenet_eval import validate_one_subnet, log_helper


MIN_BATCHES_TO_CALIB = 160


def calibrate_bn_params(model: nn.Module, data_loader, num_batches, device):
    # reset running stats
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            m.momentum = None 
            m.reset_running_stats()

    with torch.no_grad():
        stacked_images = None
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device)
            if stacked_images is None:
                stacked_images = images
            elif len(stacked_images) < MIN_BATCHES_TO_CALIB:
                stacked_images = torch.concat([stacked_images, images], dim=0)
            else:
                model(stacked_images)
                stacked_images = None
            if i >= num_batches:
                break
    model.eval()

    # sync bn running stats
    if dist.is_initialized():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                dist.all_reduce(m.running_mean, op=dist.ReduceOp.SUM)
                dist.all_reduce(m.running_var, op=dist.ReduceOp.SUM)
                m.running_mean /= dist.get_world_size()
                m.running_var /= dist.get_world_size()
                dist.all_reduce(m.num_batches_tracked)


def validate(
    train_loader, 
    val_loader, 
    model, 
    criterion, 
    args, 
    logger,
    bn_calibration=True,
    eval_random_net=True
):
    supernet = model.module \
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    results = []
    top1_list, top5_list = [],  []
    with torch.no_grad():
        for i in range(2 + eval_random_net):
            if i == 0:
                supernet.set_max_subnet()
            elif i == 1:
                supernet.set_min_subnet()
            else:
                supernet.sample_active_subnet()

            subnet = supernet.get_active_subnet()
            subnet_cfg = subnet.config.as_dict()
            subnet.cuda(args.gpu)
            if args.quant_mode:
                set_quant_mode(subnet)
                
            if bn_calibration:
                subnet.eval()
                calibrate_bn_params(subnet, train_loader, args.post_bn_calibration_batch_num, device=torch.device('cuda', args.gpu))

            acc1, acc5, loss, flops, params = validate_one_subnet(
                val_loader, subnet, criterion, args, logger
            )
            top1_list.append(acc1)
            top5_list.append(acc5)

            summary = str({
                        'net_id': ['max_net', 'min_net', 'random_net'][i],
                        'mode': 'evaluate',
                        'epoch': getattr(args, 'curr_epoch', -1),
                        'acc1': acc1,
                        'acc5': acc5,
                        'loss': loss,
                        'flops': flops,
                        'params': params,
                        **subnet_cfg
            })

            if args.distributed and getattr(args, 'distributed_val', True):
                logger.info(summary)
                results += [summary]
            else:
                group = comm.reduce_eval_results(summary, args.gpu)
                results += group
                for rec in group:
                    logger.info(rec)
    return top1_list, top5_list


def validate_spec_subnet(train_loader, val_loader, model, criterion, args, logger, subnet_choice):
    supernet = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    supernet.set_active_subnet(subnet_choice)
    subnet = supernet.get_active_subnet()
    subnet.cuda(args.gpu)

    if args.quant_mode:
        set_quant_mode(subnet)

    subnet.eval()
    calibrate_bn_params(subnet, train_loader, args.post_bn_calibration_batch_num, device=torch.device('cuda', args.gpu))

    acc1, acc5, loss, flops, params = validate_one_subnet(
        val_loader, subnet, criterion, args, logger
    )

    return acc1, acc5, loss, flops, params