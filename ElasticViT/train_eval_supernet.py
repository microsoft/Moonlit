# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from copy import deepcopy
import logging
import torch
import yaml
import os
from pathlib import Path
from process import train_one_epoch, PerformanceScoreboard, eval_one_subnet
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from models import build_supernet, MobileBlock, build_teachers
from util import load_data_dist, get_config, init_logger, load_checkpoint, save_checkpoint, ProgressMonitor, TensorBoardMonitor, AdaptiveLossSoft, SoftTargetCrossEntropy, SoftTargetCrossEntropyNoneSoftmax, init_distributed_training, setup_print, fix_random_seed
from timm.data import Mixup
import random
import numpy as np

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


def get_bank_id_direction(current_bank_id, bank_nums, direction=0):
    last_bank_id = current_bank_id
    if direction == 0:
        if last_bank_id == bank_nums-1:
            direction = 1
            return current_bank_id-1, direction
        else:
            return current_bank_id+1, direction
    else:
        if last_bank_id == 0:
            direction = 0
            return current_bank_id+1, direction
        else:
            return current_bank_id-1, direction


def main():
    script_dir = Path.cwd()
    args = get_config(default_file=script_dir / 'configs/final_3min_space.yaml')

    monitors = None
    assert args.training_device == 'gpu', 'NOT SUPPORT CPU TRAINING NOW'

    init_distributed_training(args)

    print(
            f'training on world_size {dist.get_world_size()}, rank {dist.get_rank()}, local_rank {args.local_rank}')

    fix_random_seed(seed=0)

    output_dir = script_dir / args.output_dir

    if args.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

        log_dir = init_logger(
            args.name, script_dir, script_dir / 'logging.conf', pre='master_node_' if args.rank == 0 else '')
        logger = logging.getLogger()

        with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
            yaml.safe_dump(args, yaml_file)

        pymonitor = ProgressMonitor(logger)
        tbmonitor = TensorBoardMonitor(logger, log_dir)
        monitors = [pymonitor, tbmonitor]

    assert args.rank >= 0, 'ERROR IN RANK'
    assert args.distributed

    setup_print(is_master=args.rank == 0)

    if args.rank == 0:
        print(args)

    scaled_linear_lr = args.lr * dist.get_world_size() * args.dataloader.batch_size / 512
    scaled_linear_min_lr = args.min_lr * \
        dist.get_world_size() * args.dataloader.batch_size / 512
    scaled_linear_warmup_lr = args.warmup_lr * \
        dist.get_world_size() * args.dataloader.batch_size / 512

    args.lr = scaled_linear_lr
    args.min_lr = scaled_linear_min_lr
    args.warmup_lr = scaled_linear_warmup_lr

    # ------------- model --------------

    model, lib_data_dir, sampling_rate_inc = build_supernet(args)

    model.arch_sampling(mode='max')
    start_epoch = 0
    bank_nums = len(model.bank_flops_ranges)

    # ------------- model EMA -------------
    model.cuda()
    # chen: decay, 2022.12.1, 0.99985->0.9999
    model_ema = ModelEma(model=model, decay=0.99985, device='', resume='')

    # ------------- handle the weight-decay ------------- (see the paper on why we disable weight decay for CNNs)
    if args.weight_decay > 0.:
        skip_list = []
        for module_name, module in model.named_modules():
            if "first_conv" in module_name or isinstance(module, MobileBlock):
                for name, params in module.named_parameters():
                    skip_list.append(f"{module_name}.{name}")

        def no_weight_decay():
            # print(skip_list)
            return skip_list

        setattr(model, "no_weight_decay", no_weight_decay)

    # ------------- optmizer -------------
    optimizer = create_optimizer(args, model)

    # ------------- auto resume -------------
    chkp_file = args.resume.path if (args.resume.path is not None and os.path.exists(args.resume.path)) else os.path.join(output_dir, args.name + '_checkpoint.pth.tar')
    if os.path.exists(chkp_file):
        print("load checkpoint from", chkp_file)
        model, start_epoch, _ = load_checkpoint(
            model, chkp_file=chkp_file, strict=True, lean=args.resume.lean, optimizer=optimizer if not args.eval else None)
        model_ema.ema = deepcopy(model)

        if start_epoch > 0:
            model.banks_prob += (sampling_rate_inc * int(start_epoch//100))
    else:
        assert not args.eval

    model_ema.ema.train()  # use training mode to track the running states
    model = DistributedDataParallel(
        model, device_ids=[args.local_rank], find_unused_parameters=True)

    # ------------- data --------------
    train_loader, val_loader, test_loader, training_sampler = load_data_dist(
        args.dataloader)

    mixup_fn = None
    if args.dataloader.aug.mixup:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=1000)

    # ------------- loss function (criterion) -------------
    if getattr(args, 'mixup', 0.) > 0.:
        criterion = SoftTargetCrossEntropyNoneSoftmax()
    else:
        criterion = LabelSmoothingCrossEntropy(args.smoothing)
    
    criterion = criterion.cuda()

    num_epochs = args.epochs
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    teacher_model = None
    distillation_loss = None
    pretrained_teacher_path = getattr(
        args, 'teacher_path', f'{lib_data_dir}/pre_trained_teacher_models')
    
    if args.distillation and not args.eval:
        teacher_model = build_teachers(args, pretrained_teacher_path)

        if args.distillation:
            distillation_loss = SoftTargetCrossEntropyNoneSoftmax().cuda()

    inplace_distillation_loss = None
    if args.inplace_distillation:
        if getattr(args, 'alpha_divergence', True):
            inplace_distillation_loss = AdaptiveLossSoft().cuda()
        else:
            inplace_distillation_loss = SoftTargetCrossEntropy().cuda()

    if start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.rank == 0:
        logger.info(('Optimizer: %s' % optimizer).replace(
            '\n', '\n' + ' ' * 11))
        logger.info('Total epoch: %d, Start epoch %d, Val cycle: %d',
                    num_epochs, start_epoch, args.val_cycle)
    perf_scoreboard = PerformanceScoreboard(args.log.num_best_scores)

    v_top1, v_top5, v_loss = 0, 0, 0
    current_bank_id, direction = 0, 0

    print(model)

    if args.eval:
        model_cfg = getattr(args, args.eval_model)
        top1_eval_acc = eval_one_subnet(
            subnet=model_cfg, model=model, train_loader=train_loader, val_loader=val_loader, args=args, mixup_fn=mixup_fn)

        if args.rank == 0:
            logging.info(
                f"[Eval mode] {args.eval_model} evaluation top-1 accuracy {top1_eval_acc} (%), FLOPs {round(model.module.compute_flops(), 2)} (M)")
            tbmonitor.writer.close() 
        return 
    
    for epoch in range(start_epoch, num_epochs):
        if (epoch + 1) % 100 == 0:
            model.module.banks_prob += sampling_rate_inc

        if args.distributed:
            training_sampler.set_epoch(epoch)

        if args.rank == 0:
            logger.info('>>>>>>>> Epoch %3d' % epoch)

        current_bank_id, direction, train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, lr_scheduler, epoch, monitors, args, modes=args.sampling_mode,
            teacher_model=teacher_model, distillation_loss=distillation_loss, inplace_distillation_loss=inplace_distillation_loss,
            mixup_fn=mixup_fn, model_ema=model_ema, current_bank_id=current_bank_id, direction=direction)

        current_bank_id = get_bank_id(current_bank_id, bank_nums)

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1)

        top1_eval_acc = eval_one_subnet(
            subnet='min', model=model, train_loader=train_loader, val_loader=val_loader, args=args, mixup_fn=mixup_fn)

        if args.rank == 0:
            logging.info(
                f"Evaluation accuracy (min subnet) [{epoch}/{num_epochs}] {top1_eval_acc}")
            print('history logged ids', model.module.history_ids)
            tbmonitor.writer.add_scalars(
                'Train Loss', {'train': train_loss}, epoch)
            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)

            save_checkpoint(epoch, 'supernet', model, {
                'top1': v_top1, 'top5': v_top5}, is_best, args.name, output_dir, optimizer=optimizer, model_ema=model_ema)

            if epoch % 10 == 0:
                save_checkpoint(epoch, 'supernet', model, {
                    'top1': v_top1, 'top5': v_top5}, False, args.name + f'_{epoch}epochs_', output_dir, optimizer=optimizer, model_ema=model_ema)

    if args.rank == 0:
        tbmonitor.writer.close() 
        logger.info('Program completed successfully ... exiting ...')


if __name__ == "__main__":
    main()
