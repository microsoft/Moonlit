import argparse
import copy
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
from modules.block_kd import get_quant_efficientnet_teacher_model, get_efficientnet_teacher_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dataset_path', default='imagenet_path', type=str, help='imagenet dataset path')
    parser.add_argument('--output_path', default='./result/lsq_efficientnet_teacher')
    parser.add_argument('--teacher_arch', default='efficientnet-b5', type=str)
    # training setting
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--warmup_epochs', default=0, type=int)
    parser.add_argument('--train_batch_size', default=12, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine'])
    parser.add_argument('--lr_step_size', default=1, type=int, help='decrease lr every step-size steps')
    parser.add_argument('--lr_gamma', default=0.9, type=float, help='decrease lr by a factor of lr_gamma')
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'sgd_nesterov', 'adam'], type=str)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--augment', default='default', choices=['default', 'auto_augment_tf'])
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--grad_clip_value', default=None, help='gradient clip value')
    # others
    parser.add_argument('--test_only', action='store_true', help='skip training. only evaluate.')
    parser.add_argument('--manual_seed', default=0, type=int)
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    os.makedirs(args.output_path, exist_ok=True)

    dist.init_process_group(backend='nccl')
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
    )

    model = get_quant_efficientnet_teacher_model(name=args.teacher_arch)

    if is_master_proc():
        writer = SummaryWriter(args.output_path)
    else:
        writer = None

    start_time = time.time()

    if args.test_only:
        state_dict = torch.load(os.path.join(args.output_path, 'checkpoint.pth'), map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=False)
        model.to(args.device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device], find_unused_parameters=False)
        evaluate(args, model, test_dataloader, state_dict['epoch'])
    else:
        train(args, model, train_dataloader, test_dataloader, writer)

    end_time = time.time()
    time_s = end_time - start_time
    dist_print(f'Training time: {time_s/3600:.2f}h')


def train(args, model: nn.Module, train_dataloader, eval_dataloader, writer: SummaryWriter):
    model.to(args.device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device], find_unused_parameters=False)

    parameters = model.parameters()
    optimizer = get_optimizer(args.optimizer, args.learning_rate, parameters, args.momentum, args.weight_decay)
    lr_scheduler = get_lr_scheduler(args.lr_scheduler, optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma, 
                        epochs=args.num_epochs, T_max=len(train_dataloader) * args.num_epochs // args.lr_step_size)
    
    criterion = nn.CrossEntropyLoss() 
    
    start_epoch = 0

    log_path = os.path.join(args.output_path, 'train.log')
    if is_master_proc():
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    for epoch in range(start_epoch, args.num_epochs - start_epoch):
        model.train()
        # train one epoch
        with tqdm(total=len(train_dataloader), desc=f'train epoch [{epoch} / {args.num_epochs}]', disable=not is_master_proc()) as t:
            for batch_idx, (image, label) in enumerate(train_dataloader):
                image = image.to(args.device)
                label = label.to(args.device)
                output = model(image)

                loss = criterion(output, label)
                loss.backward()

                # clip gradient
                if args.grad_clip_value:
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_value)

                optimizer.step()
                optimizer.zero_grad()

                # accuracy
                acc1 = torch.sum(torch.eq(torch.argmax(output, dim=1), label)) / len(output) * 100
                t.set_postfix(loss=loss.item(), acc1=acc1.item(), lr=optimizer.param_groups[0]['lr'])
                t.update()

                if is_master_proc():
                    with open(log_path, 'a') as f:
                       f.write(f'{datetime.now()} | epoch {epoch} | batch {batch_idx}/{len(train_dataloader)} | loss {loss: .2f} | acc1 {acc1: .2f} | lr {optimizer.param_groups[0]["lr"]: .5f}\n')
                    scalar_idx = epoch * len(train_dataloader) + batch_idx
                    writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], scalar_idx)
                    writer.add_scalar(f'loss', loss, scalar_idx)
                    writer.add_scalar(f'acc1', acc1, scalar_idx)
                # cosine lr scheduler update per args.lr_step_size steps
                lr_scheduler.step()


        evaluate(args, model, eval_dataloader, epoch, writer)
        checkpoint = {
            'epoch': epoch,
            'model': model.module.state_dict(),
            'lr_schduler': lr_scheduler.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        checkpoint_path = os.path.join(args.output_path, 'checkpoint.pth')
        save_on_master(checkpoint, checkpoint_path)
        if is_master_proc():
            os.system(f'cp {checkpoint_path} {checkpoint_path.replace("checkpoint.pth", f"model_{epoch}.pth")}')


def evaluate(args, model, eval_dataloader, epoch, writer=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_metric = DistributedAvgMetric()
    acc1_metric = DistributedAvgMetric()
    with tqdm(total=len(eval_dataloader), desc=f'eval epoch {epoch}', disable=not is_master_proc()) as t:
        for batch_idx, (image, label) in enumerate(eval_dataloader):
            image = image.to(args.device)
            label = label.to(args.device)
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, label)
                acc1 = torch.sum(torch.eq(torch.argmax(output, dim=1), label)) / len(output) * 100
            loss_metric.update(copy.deepcopy(loss), len(image))
            acc1_metric.update(copy.deepcopy(acc1), len(image))
            t.set_postfix(loss=(round(loss.item(), 2), round(loss_metric.avg, 2)), 
                          acc1=(round(acc1.item(), 2), round(acc1_metric.avg, 2)))
            t.update()
 
    if is_master_proc():
        log_path = os.path.join(args.output_path, 'eval.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(f'{datetime.now()} | epoch {epoch} | loss {loss_metric.avg} | acc1 {acc1_metric.avg}\n')


if __name__ == '__main__':
    main()