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

# from nn_meter.predictor.quantize_block_predictor import BlockLatencyPredictor
from modules.modeling.supernet import Stage
from modules.training.dataset.imagenet_dataloader import build_imagenet_dataloader
from modules.training.dist_utils import DistributedAvgMetric, dist_print, is_master_proc, save_on_master
from modules.block_kd import get_quant_efficientnet_teacher_model, get_efficientnet_teacher_model, BlockKDManager, StagePlusProj
from modules.search_space.superspace import get_superspace, get_available_superspaces
from modules.training.loss_fn import NSRLoss
from modules.modeling.common.flops_counter import count_net_flops_and_params
from modules.modeling.ops.lsq_plus import set_quant_mode
from modules.latency_predictor import LatencyPredictor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dataset_path', default='imagenet_path', type=str, help='imagenet dataset path')
    parser.add_argument('--output_path', default='./results/block_kd/lut/')
    parser.add_argument('--teacher_arch', default='efficientnet-b5', type=str)
    parser.add_argument('--teacher_checkpoint_path', default='checkpoints/block_kd/teacher_checkpoint/checkpoint.pth', type=str)
    parser.add_argument('--checkpoint_path', default='./checkpoints/block_kd/lut')
    parser.add_argument('--superspace', choices=get_available_superspaces(), required=True, type=str)
    parser.add_argument('--platform', required=True, help='latency predictor platform')
    # dataset setting
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=50, type=int)
    parser.add_argument('--valid_size', default=50000, help='size of validation dataset')
    parser.add_argument('--augment', default='default', choices=['default', 'auto_augment_tf'])
    parser.add_argument('--num_workers', default=8, type=int)
    # stage setting
    parser.add_argument('--stage_list', nargs='*', help='only train stages in this list')
    parser.add_argument('--width_window_filter', nargs='*',
                        help='If set, only evaluate stages with width window choice in this list.')
    parser.add_argument('--num_samples_per_stage', default=1000, type=int)
    # compare training settings
    parser.add_argument('--hw_list', nargs='+', default=[224])
    parser.add_argument('--loss_fn', default='nsr', choices=['mse', 'nsr'], help='the type of loss function')
    # others
    parser.add_argument('--num_calib_batches', default=4, type=int, help='num batches to calibrate bn params')
    parser.add_argument('--manual_seed', default=0, type=int)
    parser.add_argument('--debug', action='store_true', help='debug mode, only train and eval a small number of batches')
    parser.add_argument('--debug_batches', default=10, type=int)
    parser.add_argument('--fp32_mode', action='store_true')  # fp32 mode, for debug
    args = parser.parse_args()

    args.checkpoint_path = os.path.join(args.checkpoint_path, args.superspace)
    args.hw_list = [int(hw) for hw in args.hw_list]
    args.disable_ddp = args.local_rank == -1
    if args.local_rank == -1:
        args.local_rank = 0
    
    args.output_path = os.path.join(args.output_path, args.superspace)
    if args.fp32_mode:
        args.output_path = args.output_path + '_fp32'
    # logging to a local path instead of teamdrive because the latter is too slow
    # cp local log to teamdrive in the end
    args.local_log_path = '.log'
    os.makedirs(args.local_log_path, exist_ok=True)
    return args


def main():
    args = get_args()

    os.makedirs(args.output_path, exist_ok=True)

    if not args.disable_ddp:
        dist.init_process_group(backend='nccl')
    args.device = torch.device('cuda', args.local_rank)
    
    # torch.manual_seed(args.manual_seed)
    # torch.cuda.manual_seed_all(args.manual_seed)
    # np.random.seed(args.manual_seed)
    
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

    if not args.fp32_mode:
        teacher = get_quant_efficientnet_teacher_model(args.teacher_arch)
        state_dict = torch.load(args.teacher_checkpoint_path, map_location='cpu')
        teacher.load_state_dict(state_dict['model'])
        dist_print(f'Load teacher state_dict from {args.teacher_checkpoint_path}')
    else:
        teacher = get_efficientnet_teacher_model(args.teacher_arch)
    teacher.to(args.device)

    student_superspace = get_superspace(args.superspace)

    block_kd_manager = BlockKDManager(superspace=student_superspace, teacher=teacher)
    
    latency_predictor = LatencyPredictor(args.platform)

    try:
        for stage_name in block_kd_manager.stage_name_list:
            if not student_superspace.need_choose_block(stage_name):
                continue

            stages = block_kd_manager.get_stages(stage_name)

            for model_name, model in stages:
                if args.stage_list and model_name not in args.stage_list: # skip the stages not in args.stage_list
                    continue
                if not args.fp32_mode:
                    set_quant_mode(model)
                state_dict = torch.load(os.path.join(args.checkpoint_path, model_name, 'lsq.pth' if not args.fp32_mode else 'checkpoint.pth'), map_location='cpu')
                model.load_state_dict(state_dict['model'])
                model.to(args.device)
                if not args.disable_ddp:
                    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device], find_unused_parameters=True)
                    model_without_ddp = model.module 
                else:
                    model_without_ddp = model
                criterion = nn.MSELoss() if args.loss_fn == 'mse' else NSRLoss()
                
                for wwc in range(block_kd_manager.get_num_width_window_choices(stage_name)):
                    if args.width_window_filter and str(wwc) not in args.width_window_filter:
                        continue
                    
                    active_width_list = block_kd_manager.get_active_width_list(stage_name, wwc)
                    block_width_config = model_name + f'_{wwc}'
                    for i in range(args.num_samples_per_stage):
                        dist_print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), f'[{i}/{args.num_samples_per_stage}]')
                        kwe_list = model_without_ddp.sample_active_sub_stage(width_list=active_width_list)
                        kwe_list_str = '_'.join(['#'.join([str(v[0]), str(v[1]), str(v[2])]) for v in kwe_list])
                        # get loss
                        loss, resolution = evaluate_stage(stage_name, model_name, args, model, teacher, criterion, eval_dataloader, train_dataloader)
                        hw = block_kd_manager.get_hw(stage_name, resolution)
                        # get mflops and mparams
                        cin = model_without_ddp.in_proj.active_out_channels
                        input_shape = [1, cin, hw, hw]
                        latency = predict_stage_latency(model_without_ddp.stage, input_shape, latency_predictor)
                        mflops, mparams = count_net_flops_and_params(model_without_ddp.stage.get_active_sub_stage(cin), input_shape)
                        if is_master_proc():
                            with open(os.path.join(args.local_log_path, block_width_config + '.csv'), 'a') as f:
                                f.write(f'{kwe_list_str},{"x".join([str(v) for v in input_shape])},{loss:.4f},{mflops:.4f},{mparams:.4f},{latency:.4f}\n')

                    if is_master_proc():
                        os.system(f'cp {os.path.join(args.local_log_path, block_width_config + ".csv")} {os.path.join(args.output_path, block_width_config + ".csv")}')
    finally:
        if is_master_proc():
            os.system(f'cp {args.local_log_path}/* {args.output_path}')


def evaluate_stage(stage_name, model_name, args, model, teacher, criterion, eval_dataloader, train_dataloader):
    model.eval()
    teacher.eval()
    loss_metric = DistributedAvgMetric()
    resolution = get_input_resolution('random', args.hw_list)

    # calibrate bn running stats
    def sub_train_loader(num_batches):
        for i, (image, label) in enumerate(train_dataloader):
            if i < num_batches:
                image = image.to(args.device)
                image = resize_image(image, resolution)
                teacher_out, teacher_in = teacher.forward_to_stage(image, stage_name)
                yield teacher_in, teacher_out
            else:
                break
    calibrate_bn_params(model, sub_train_loader(args.num_calib_batches))

    # eval
    with tqdm(total=len(eval_dataloader), desc=f'eval {model_name}', disable=True) as t:
    # with tqdm(total=len(eval_dataloader), desc=f'eval {model_name}', disable=not is_master_proc()) as t:
        for batch_idx, (image, label) in enumerate(eval_dataloader):
            image = image.to(args.device)
            image = resize_image(image, resolution)
            with torch.no_grad():
                teacher_out, teacher_in = teacher.forward_to_stage(image, stage_name)
                student_out = model(teacher_in)
                loss = criterion(student_out, teacher_out)
                dist_print(loss)
            loss_metric.update(loss, len(image))
            t.set_postfix(loss=round(loss_metric.local_avg, 4))
            t.update()

            if args.debug and batch_idx >= args.debug_batches: break
    return loss_metric.avg, resolution


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


def predict_stage_latency(stage: Stage, input_shape, predictor: LatencyPredictor):
    rv = 0
    cin, hw = input_shape[1], input_shape[2]
    for block in stage.active_blocks:
        config = block.get_active_block_config(cin=cin)
        rv += predictor.predict_block(config, hw)
        cin = config.cout 
        hw //= config.stride
    return rv 


if __name__ == '__main__':
    main()