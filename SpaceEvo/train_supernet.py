# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# based on AlphaNet

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys
import operator
from datetime import date, datetime

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed


from modules.modeling.ops.lsq_plus import set_quant_mode
from modules.search_space.superspace import get_superspace, get_available_superspaces
from modules.modeling.supernet import Supernet
from modules.alphanet_training.data.data_loader import build_data_loader
from modules.alphanet_training.utils.config import setup
import modules.alphanet_training.utils.saver as saver
from modules.alphanet_training.utils.progress import AverageMeter, ProgressMeter, accuracy
import modules.alphanet_training.utils.comm as comm
import modules.alphanet_training.utils.logging as logging
from modules.alphanet_training.evaluate import supernet_eval
from modules.alphanet_training.solver import build_optimizer, build_lr_scheduler
import modules.alphanet_training.utils.loss_ops as loss_ops 

from copy import deepcopy
import numpy as np
import joblib 

# from sklearn.ensemble import RandomForestRegressor


parser = argparse.ArgumentParser(description='Supernet sandwich rule training.')
parser.add_argument('--superspace', choices=get_available_superspaces(), required=True, type=str)
parser.add_argument('--supernet_choice', type=str, help='candidate of superspace, e.g. 322223', default='322223')
parser.add_argument('--align_sample', action='store_true', help='all blocks in a stage share the same kwe values')
parser.add_argument('--config-file', default='supernet_training_configs/tmp_debug.yaml', type=str,
                    help='training configuration')
parser.add_argument('--quant_mode', action='store_true', help='lsq finetune')
parser.add_argument('--local_rank', default=-1, type=int)
# overwrite args
parser.add_argument('--batch_size_per_gpu', type=int, default=None)
parser.add_argument('--resume', action='store_true')


logger = logging.get_logger(__name__)


def build_args_and_env(run_args):

    assert run_args.config_file and os.path.isfile(run_args.config_file), 'cannot locate config file'
    args = setup(run_args.config_file)
    args.config_file = run_args.config_file
    args.superspace = run_args.superspace
    args.supernet_choice = run_args.supernet_choice
    args.align_sample = run_args.align_sample
    args.supernet_encoding = args.superspace + '-' + args.supernet_choice
    args.arch = args.supernet_encoding + f'-align{int(args.align_sample)}'
    args.exp_name = args.arch
    args.local_rank = run_args.local_rank

    # load config
    assert args.distributed and args.multiprocessing_distributed, 'only support DDP training'
    args.distributed = True

    args.models_save_dir = os.path.join(args.models_save_dir, args.exp_name)

    if comm.is_master_process():
        os.makedirs(args.models_save_dir, exist_ok=True)

        # backup config file
        saver.copy_file(args.config_file, '{}/{}'.format(args.models_save_dir, os.path.basename(args.config_file)))

    args.checkpoint_save_path = os.path.join(
        args.models_save_dir, f'checkpoint.pth'
    )
    args.logging_save_path = os.path.join(
        args.models_save_dir, f'stdout.log'
    )
    args.valid_acc_save_path = os.path.join(
        args.models_save_dir, f'valid_acc.log'
    )

    # === override args ===
    args.batch_size_per_gpu = run_args.batch_size_per_gpu or args.batch_size_per_gpu
    if run_args.resume:
        args.resume = run_args.resume
    if args.resume and not os.path.exists(str(args.resume)):
        args.resume = args.checkpoint_save_path

    # === modify args in LSQ QAT model ===
    args.quant_mode = run_args.quant_mode
    if args.quant_mode:
        args.resume = args.checkpoint_save_path
        args.checkpoint_save_path = os.path.join(args.models_save_dir, 'lsq.pth')
        args.logging_save_path = os.path.join(args.models_save_dir, 'lsq_stdout.log')
        args.valid_acc_save_path = os.path.join(args.models_save_dir, 'lsq_valid_acc.log')
        args.lr_scheduler.base_lr = args.lr_scheduler.base_lr / 10
        args.warmup_epochs = 3
        args.epochs = 50
        args.batch_size_per_gpu = min(args.batch_size_per_gpu, 64)
    return args


def main():
    run_args = parser.parse_args()
    args = build_args_and_env(run_args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # cudnn.deterministic = True
    # warnings.warn('You have chosen to seed training. '
    #                'This will turn on the CUDNN deterministic setting, '
    #                'which can slow down your training considerably! '
    #                'You may see unexpected behavior when restarting '
    #                'from checkpoints.')
    main_worker(args)


def main_worker(args):
    dist.init_process_group(
        backend=args.dist_backend, 
    )
    args.world_size = dist.get_world_size()
    args.gpu = args.local_rank  # local rank, local machine cuda id
    args.batch_size = args.batch_size_per_gpu
    args.batch_size_total = args.batch_size * args.world_size
    # rescale base lr
    args.lr_scheduler.base_lr = args.lr_scheduler.base_lr * (max(1, args.batch_size_total // 256))

    # set random seed, make sure all random subgraph generated would be the same
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed(args.seed)

    # Setup logging format.
    logging.setup_logging(args.logging_save_path, 'a')

    logger.info(f"Use GPU: {args.gpu}, world size {args.world_size}")

    # synchronize is needed here to prevent a possible timeout after calling
    # init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    args.rank = comm.get_rank() # global rank
    torch.cuda.set_device(args.gpu)

    # build model
    logger.info("=> creating model '{}'".format(args.arch))
    model = Supernet.build_from_str(args.supernet_encoding)
    model.align_sample = args.align_sample

    if args.quant_mode:
        set_quant_mode(model)

    model.zero_last_gamma()
    model.cuda(args.gpu)


    # use sync batchnorm
    if getattr(args, 'sync_bn', False):
        model.apply(
                lambda m: setattr(m, 'need_sync', True))

    model = comm.get_parallel_model(model, args.gpu) # local rank

    if comm.is_master_process():
        logger.info(model)

    criterion = loss_ops.CrossEntropyLossSmooth(args.label_smoothing).cuda(args.gpu)
    soft_criterion = loss_ops.AdaptiveLossSoft(args.alpha_min, args.alpha_max, args.iw_clip).cuda(args.gpu)
    # soft_criterion = loss_ops.KLLossSoft().cuda(args.gpu)
    if not getattr(args, 'inplace_distill', True):
        soft_criterion = None

    ## load dataset, train_sampler: distributed
    logger.info(f'Start loading data {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
    train_loader, val_loader, test_loader, train_sampler =  build_data_loader(args)
    if val_loader is None:
        val_loader = test_loader
        logger.info(f'Valid loader is None. Use test loader to do evalution. len {len(val_loader)}')
    else:
        logger.info(f'len train loader and val loader: {len(train_loader)} {len(val_loader)}')
    logger.info(f'Finish loading data {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')


    args.n_iters_per_epoch = len(train_loader)
    if args.debug:
        args.n_iters_per_epoch = args.debug_batches

    logger.info( f'building optimizer and lr scheduler, \
            local rank {args.gpu}, global rank {args.rank}, world_size {args.world_size}')
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
 
    # optionally resume from a checkpoint
    if args.resume:
        if args.quant_mode:
            saver.load_checkpoints(args, model, logger=logger)
            args.start_epoch = 0
        else:
            saver.load_checkpoints(args, model, optimizer, lr_scheduler, logger)

    logger.info(args)

    if comm.is_master_process() and not args.quant_mode:
        writer = SummaryWriter(args.models_save_dir)
    else:
        writer = None

    best_max_net_acc1 = -1
    max_net_acc1 = 0
    best_epoch = -1
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        args.curr_epoch = epoch
        logger.info('Training lr {}'.format(lr_scheduler.get_lr()[0]))

        # train for one epoch
        acc1, acc5 = train_epoch(epoch, model, train_loader, optimizer, criterion, args, \
                soft_criterion=soft_criterion, lr_scheduler=lr_scheduler, writer=writer)
        if writer:
            writer.add_scalar('Acc1/Train', acc1, epoch)

        # validate supernet model
        if epoch % args.valid_freq == 0 or epoch == args.epochs - 1:
            (max_net_acc1, min_net_acc1, random_net_acc1), _ = validate(
                train_loader, val_loader, model, criterion, args
            )
            if writer:
                writer.add_scalar('Acc1/Valid/MaxNet', max_net_acc1, epoch)
                writer.add_scalar('Acc1/Valid/MinNet', min_net_acc1, epoch)
                writer.add_scalar('Acc1/Valid/RandomNet', random_net_acc1, epoch)
                if comm.is_master_process():
                    with open(args.valid_acc_save_path, 'a') as f:
                        f.write(f'acc1 epoch {epoch} | max_net_acc1 {max_net_acc1:.4f} | min_net_acc1 {min_net_acc1:.4f} | random_net_acc1 {random_net_acc1:.4f}\n')
                 
        if comm.is_master_process():
            for _ in range(10): # try to save mutiple times because on itp this could fail
                try:
                    # save checkpoints
                    saver.save_checkpoint(
                        args.checkpoint_save_path, 
                        model,
                        optimizer,
                        lr_scheduler, 
                        args,
                        epoch,
                    )
                    # back up checkpoint in case of failure
                    if epoch % 10 == 0:
                        os.system(f'cp {args.checkpoint_save_path} {args.checkpoint_save_path}.bak')

                    if epoch % 50 == 0:
                        os.system(f'cp {args.checkpoint_save_path} {args.checkpoint_save_path.replace(".pth", f"_{epoch}.pth")}')

                    # save best model
                    if max_net_acc1 > best_max_net_acc1:
                        best_max_net_acc1 = max_net_acc1
                        best_epoch = epoch
                        os.system(f'cp {args.checkpoint_save_path} {args.checkpoint_save_path.replace(".pth", "_best.pth")}')

                except:
                    logger.info('Save checkpoint failed. Retry.')
                else:
                    break


def train_epoch(
    epoch, 
    model, 
    train_loader, 
    optimizer, 
    criterion, 
    args, 
    soft_criterion=None, 
    lr_scheduler=None,
    writer: SummaryWriter=None,
):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()

    num_updates = epoch * len(train_loader)

    for batch_idx, (images, target) in enumerate(train_loader):
        cur_losses = []

        # measure data loading time
        data_time.update(time.time() - end)
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # total subnets to be sampled
        num_subnet_training = max(2, getattr(args, 'num_arch_training', 2))
        optimizer.zero_grad()

        ### compute gradients using sandwich rule ###
        # step 1 sample the largest network, apply regularization to only the largest network
        model.module.set_max_subnet()
        model.module.set_dropout_rate(args.dropout) #dropout for supernet
        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        cur_losses.append(loss.item())
        with torch.no_grad():
            soft_logits = output.clone().detach()

        # step 2. sample the smallest network and several random networks
        sandwich_rule = getattr(args, 'sandwich_rule', True)
        model.module.set_dropout_rate(0)  #reset dropout rate
        for arch_id in range(1, num_subnet_training):
            if arch_id == num_subnet_training-1 and sandwich_rule:
                model.module.set_min_subnet()
            else:
                model.module.sample_active_subnet()

            # calcualting loss
            output = model(images)

            if soft_criterion:
                loss = soft_criterion(output, soft_logits)
            else:
                assert not args.inplace_distill
                loss = criterion(output, target)
            
            loss.backward()
            cur_losses.append(loss.item())

        #clip gradients if specfied
        if getattr(args, 'grad_clip_value', None):
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_value)

        optimizer.step()

        #accuracy measured on the local batch
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        if args.distributed:
            corr1, corr5, loss = acc1*args.batch_size, acc5*args.batch_size, loss.item()*args.batch_size #just in case the batch size is different on different nodes
            stats = torch.tensor([corr1, corr5, loss, args.batch_size], device=args.gpu)
            dist.barrier()  # synchronizes all processes
            dist.all_reduce(stats, op=torch.distributed.ReduceOp.SUM) 
            corr1, corr5, loss, batch_size = stats.tolist()
            acc1, acc5, loss = corr1/batch_size, corr5/batch_size, loss/batch_size
            losses.update(loss, batch_size)
            top1.update(acc1, batch_size)
            top5.update(acc5, batch_size)
        else:
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        num_updates += 1
        if lr_scheduler is not None:
            lr_scheduler.step()

        if batch_idx % args.print_freq == 0 and comm.is_master_process():
            progress.display(batch_idx, logger)

            # update writer
            if writer:
                global_step = epoch * len(train_loader) + batch_idx
                for i, loss in enumerate(cur_losses):
                    if i == 0: name = 'max_net'
                    elif i == len(cur_losses) - 1: name = 'min_net'
                    else: name = 'random_net'
                    writer.add_scalar(f'Train/loss/{name}', loss, global_step)
                writer.add_scalar('Train/learning_rate', lr_scheduler.get_lr()[0], global_step)

        if args.debug and batch_idx >= args.debug_batches:
            break
        
    return top1.avg, top5.avg


def validate(
    train_loader, 
    val_loader, 
    model, 
    criterion, 
    args, 
    distributed=True,
):
    return supernet_eval.validate(
        train_loader,
        val_loader, 
        model, 
        criterion,
        args,
        logger,
        bn_calibration=True,
    )


if __name__ == '__main__':
    main()


