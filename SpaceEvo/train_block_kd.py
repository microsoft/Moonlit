import argparse
from datetime import datetime
import os
import time
from typing import List

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torchvision
from tqdm import tqdm
from tensorboardX import SummaryWriter

from modules.training.dataset.imagenet_dataloader import build_imagenet_dataloader
from modules.training.lr_scheduler import get_lr_scheduler
from modules.training.optimizer import get_optimizer
from modules.training.dist_utils import DistributedAvgMetric, dist_print, is_master_proc, save_on_master
from modules.block_kd import get_efficientnet_teacher_model, BlockKDManager, StagePlusProj
from modules.search_space.superspace import get_superspace, get_available_superspaces
from modules.training.loss_fn import NSRLoss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dataset_path', default='<path_to_imagenet>', type=str, help='imagenet dataset path')
    parser.add_argument('--output_path', default='./checkpoints/block_kd/')
    parser.add_argument('--teacher_arch', default='efficientnet-b5', type=str)
    parser.add_argument('--superspace', choices=get_available_superspaces(), required=True, type=str)
    # training setting
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--learning_rate_list', default=[0.005, 0.01, 0.01, 0.01, 0.01, 0.005], nargs='+')
    parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine', 'step'])
    parser.add_argument('--lr_step_size', default=20, type=int, help='decrease lr every step-size steps, only for cosine lr scheduler')
    parser.add_argument('--lr_gamma', default=0.9, type=float, help='decrease lr by a factor of lr_gamma')
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'sgd_nesterov', 'adam'], type=str)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--augment', default='default', choices=['default', 'auto_augment_tf'])
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--grad_clip_value', default=1, help='gradient clip value')
    parser.add_argument('--resume', action='store_true', help='resume training from the last epoch')
    parser.add_argument('--stage_list', nargs='*', help='only train stages in this list')
    parser.add_argument('--valid_size', default=50000, help='size of validation dataset')
    # compare training settings
    parser.add_argument('--hw_list', nargs='+', default=[224])
    parser.add_argument('--loss_fn', default='nsr', choices=['mse', 'nsr'], help='the type of loss function')
    parser.add_argument('--inplace_distill_from_teacher', action='store_true', help='students in sandwich rule learn from teacher')
    # others
    parser.add_argument('--test_only', action='store_true', help='skip training. only evaluate.')
    parser.add_argument('--num_calib_batches', default=20, help='num batches to calibrate bn params')
    parser.add_argument('--manual_seed', default=0, type=int)
    parser.add_argument('--debug', action='store_true', help='debug mode, only train and eval a small number of batches')
    args = parser.parse_args()

    args.hw_list = [int(hw) for hw in args.hw_list]
    args.learning_rate_list = [float(lr) for lr in args.learning_rate_list]

    args.local_output_path = '.log'
    return args


def main():
    args = get_args()
    dist.init_process_group(backend='nccl')

    if is_master_proc():
        os.makedirs(args.output_path, exist_ok=True)
        os.makedirs(args.local_output_path, exist_ok=True)

    args.device = torch.device('cuda', args.local_rank)
    
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    
    dist_print(f'Loading dataset from {args.dataset_path}')
    train_dataloader, valid_dataloader, test_dataloader, train_sampler = build_imagenet_dataloader(
        dataset_path=args.dataset_path, 
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        distributed=dist.is_initialized(),
        augment=args.augment,
        num_workers=args.num_workers,
        valid_size=args.valid_size
    )
    eval_dataloader = valid_dataloader or test_dataloader

    teacher = get_efficientnet_teacher_model(args.teacher_arch)
    teacher.to(args.device)

    student_superspace = get_superspace(args.superspace)

    block_kd_manager = BlockKDManager(superspace=student_superspace, teacher=teacher)

    if is_master_proc():
        writer = SummaryWriter(args.local_output_path)
    else:
        writer = None

    start_time = time.time()
    for stage_name in block_kd_manager.stage_name_list:
        if not student_superspace.need_choose_block(stage_name):
            continue

        stages = block_kd_manager.get_stages(stage_name)

        for model_name, model in stages:
            if args.stage_list and model_name not in args.stage_list: # skip the stages not in args.stage_list
                continue
            dist_print(model)
            if is_master_proc():
                net_path = os.path.join(args.output_path, model_name, 'net.txt')
                os.makedirs(os.path.dirname(net_path), exist_ok=True)
                with open(net_path, 'w') as f:
                    f.write(str(model))
            if args.test_only:
                state_dict = torch.load(os.path.join(args.output_path, model_name, 'checkpoint.pth'), map_location='cpu')
                model.load_state_dict(state_dict['model'])
                model.to(args.device)
                model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device], find_unused_parameters=True)
                criterion = nn.MSELoss() if args.loss_fn == 'mse' else NSRLoss()
                evaluate_stage(stage_name, model_name, args, model, teacher, criterion, eval_dataloader, train_dataloader, state_dict['epoch'])
            else:
                train_stage(stage_name, args, model, teacher, model_name, train_dataloader, eval_dataloader, train_sampler, writer)

    end_time = time.time()
    time_s = end_time - start_time
    dist_print(f'Training time: {time_s/3600:.2f}h')


def train_stage(stage_name: str, args, model: nn.Module, teacher: nn.Module, model_name: str, train_dataloader, eval_dataloader, train_sampler, writer: SummaryWriter):
    model.to(args.device)
    teacher.to(args.device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device], find_unused_parameters=True)
    model_without_ddp = model.module

    parameters = model.parameters()
    optimizer = get_optimizer(args.optimizer, args.learning_rate_list[int(stage_name[-1])-1], parameters, args.momentum, args.weight_decay)
    lr_scheduler = get_lr_scheduler(args.lr_scheduler, optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma, 
                        epochs=args.num_epochs, T_max=len(train_dataloader) * args.num_epochs // args.lr_step_size)
    
    if args.loss_fn == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_fn == 'nsr':
        criterion = NSRLoss()
    else:
        raise ValueError(args.loss_fn)

    start_epoch = 0
    if args.resume:
        checkpoint_path = os.path.join(args.output_path, model_name, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=args.device)
            model.module.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            lr_scheduler.load_state_dict(state_dict['lr_schduler'])
            start_epoch = state_dict['epoch'] + 1
            dist_print(f'Load state_dict from {checkpoint_path}, start training from epoch {start_epoch}')
        else:
            dist_print(f'No checkpoint found at {checkpoint_path}, start training from epoch 0')

    log_path = os.path.join(args.local_output_path, model_name, 'train.log')
    remote_log_path = os.path.join(args.output_path, model_name, 'train.log')
    if is_master_proc():
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(os.path.dirname(remote_log_path), exist_ok=True)

    for epoch in range(start_epoch, args.num_epochs - start_epoch):
        train_sampler.set_epoch(epoch)
        model.train()
        teacher.train()
        # train one epoch
        with tqdm(total=len(train_dataloader), desc=f'train {model_name} epoch [{epoch} / {args.num_epochs}]', disable=not is_master_proc()) as t:
            for batch_idx, (image, label) in enumerate(train_dataloader):
                image = image.to(args.device)

                # sandwich rule
                losses = []
                for i in range(4):
                    # resize image
                    _mode = ['max', 'min', 'random', 'random'][i]
                    image_resized = resize_image(image, get_input_resolution(_mode, args.hw_list))
                    
                    with torch.no_grad():
                        if i == 0 or len(args.hw_list) > 1:
                            teacher_out, teacher_in = teacher.forward_to_stage(image_resized, stage_name)

                    # use the output of max stage to distill other smaller stages 
                    if i != 0 and not args.inplace_distill_from_teacher: 
                        model_without_ddp.set_max_sub_stage()
                        with torch.no_grad():
                            max_stage_out = model(teacher_in)
                    
                    # set student stage
                    if i == 0:
                        model_without_ddp.set_max_sub_stage()
                    if i == 1:
                        model_without_ddp.set_min_sub_stage()
                    if i > 1:
                        model_without_ddp.sample_active_sub_stage()

                    # forward and backward
                    student_out = model(teacher_in)
                    loss = criterion(student_out, teacher_out if i == 0 or args.inplace_distill_from_teacher else max_stage_out)
                    loss.backward()
                    losses.append(round(loss.item(), 4))

                # clip gradient
                if args.grad_clip_value:
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_value)

                optimizer.step()
                optimizer.zero_grad()

                t.set_postfix(losses=losses, lr=optimizer.param_groups[0]['lr'])
                t.update()

                if is_master_proc():
                    with open(log_path, 'a') as f:
                       f.write(f'{datetime.now()} | epoch {epoch} | batch {batch_idx}/{len(train_dataloader)} | losses {losses} | lr {optimizer.param_groups[0]["lr"]}\n')
                    if writer:
                        writer.add_scalar(f'{model_name}/train_loss/max', losses[0], epoch * len(train_dataloader) + batch_idx)
                        writer.add_scalar(f'{model_name}/train_loss/min', losses[1], epoch * len(train_dataloader) + batch_idx)
                        writer.add_scalar(f'{model_name}/train_loss/random1', losses[2], epoch * len(train_dataloader) + batch_idx)
                        writer.add_scalar(f'{model_name}/train_loss/random2', losses[3], epoch * len(train_dataloader) + batch_idx)
                        writer.add_scalar(f'{model_name}/train_learning_rates', optimizer.param_groups[0]['lr'], epoch * len(train_dataloader) + batch_idx)
                # cosine lr scheduler update per args.lr_step_size steps
                if args.lr_scheduler == 'cosine' and batch_idx and batch_idx % args.lr_step_size == 0: 
                    lr_scheduler.step()

                if args.debug and batch_idx >= 3: break

        # eval, update and checkpoint
        if args.lr_scheduler == 'step': # step lr scheduler update per epoch
            lr_scheduler.step()

        evaluate_stage(stage_name, model_name, args, model, teacher, criterion, eval_dataloader, train_dataloader, epoch, writer)
        checkpoint = {
            'epoch': epoch,
            'model': model.module.state_dict(),
            'lr_schduler': lr_scheduler.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        checkpoint_path = os.path.join(args.output_path, model_name, 'checkpoint.pth')
        save_on_master(checkpoint, checkpoint_path)
        if is_master_proc():
            os.system(f'cp {checkpoint_path} {checkpoint_path.replace("checkpoint.pth", f"model_{epoch}.pth")}')
            os.system(f'cp {args.local_output_path}/* {args.output_path}')
            os.system(f'cp {log_path} {remote_log_path}')
        if args.debug: break


def evaluate_stage(stage_name, model_name, args, model, teacher, criterion, eval_dataloader, train_dataloader, epoch, writer=None):
    model.eval()
    teacher.eval()
    models_to_eval = {'max': model.module.set_max_sub_stage, 'min': model.module.set_min_sub_stage, 'random': model.module.sample_active_sub_stage}
    losses = []
    for sub_stage_type, set_func in models_to_eval.items():
        loss_metric = DistributedAvgMetric()
        set_func()
        hw = get_input_resolution(sub_stage_type, args.hw_list)

        # calibrate bn running stats
        def sub_train_loader(num_batches):
            for i, (image, label) in enumerate(train_dataloader):
                if i < num_batches:
                    image = image.to(args.device)
                    image = resize_image(image, hw)
                    teacher_out, teacher_in = teacher.forward_to_stage(image, stage_name)
                    yield teacher_in, teacher_out
                else:
                    break
        calibrate_bn_params(model, sub_train_loader(args.num_calib_batches))

        # eval
        with tqdm(total=len(eval_dataloader), desc=f'eval {model_name+"_"+sub_stage_type}, epoch {epoch}', disable=not is_master_proc()) as t:
            for batch_idx, (image, label) in enumerate(eval_dataloader):
                image = image.to(args.device)
                image = resize_image(image, hw)
                with torch.no_grad():
                    teacher_out, teacher_in = teacher.forward_to_stage(image, stage_name)
                    student_out = model(teacher_in)
                    loss = criterion(student_out, teacher_out)
                loss_metric.update(loss, len(image))
                t.set_postfix(loss=round(loss_metric.local_avg, 4))
                t.update()

                if args.debug and batch_idx >= 3: break

        losses.append(round(loss_metric.avg, 4))

    if is_master_proc():
        log_path = os.path.join(args.output_path, model_name, 'eval.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(f'{datetime.now()} | epoch {epoch} | eval losses (max min random): {",".join([str(_) for _ in losses])}\n')
        if writer:
            writer.add_scalar(f'{model_name}/eval_loss/max', losses[0], epoch)
            writer.add_scalar(f'{model_name}/eval_loss/min', losses[1], epoch)
            writer.add_scalar(f'{model_name}/eval_loss/random', losses[2], epoch)


def get_input_resolution(mode, hw_list) -> int:
    if mode == 'max':
        target_hw = max(hw_list)
    if mode == 'min':
        target_hw = min(hw_list)
    if mode == 'random':
        target_hw = np.random.choice(hw_list)
    return target_hw


def resize_image(x, hw) -> torch.Tensor:
    if x.shape[-1] != hw:
        x = torch.nn.functional.interpolate(x, size=hw, mode='bicubic')
    return x


def calibrate_bn_params(model: nn.Module, data_loader):
    # reset running stats
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            m.momentum = None 
            m.reset_running_stats()

    with torch.no_grad():
        for teacher_in, _ in data_loader:
            model(teacher_in)

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


if __name__ == '__main__':
    main()