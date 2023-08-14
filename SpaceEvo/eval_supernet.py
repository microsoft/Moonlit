# based on AlphaNet: https://github.com/facebookresearch/AlphaNet

import argparse
import os
import random
from datetime import date, datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm

from modules.modeling.ops.lsq_plus import set_quant_mode
from modules.search_space.superspace import get_superspace, get_available_superspaces
from modules.modeling.supernet import Supernet
from modules.alphanet_training.data.data_loader import build_data_loader
import modules.alphanet_training.utils.saver as saver
import modules.alphanet_training.utils.comm as comm
import modules.alphanet_training.utils.logging as logging
from modules.alphanet_training.evaluate import supernet_eval
from modules.latency_predictor import LatencyPredictor

from copy import deepcopy
import numpy as np
import joblib

# from sklearn.ensemble import RandomForestRegressor


parser = argparse.ArgumentParser(description='Evaluate supernet and subnet.')
parser.add_argument('--superspace', choices=get_available_superspaces(), required=True, type=str)
parser.add_argument('--supernet_choice', type=str, required=True, nargs='+',
                    help='candidate of superspace, e.g. 322223-011120, or specific supernet, e.g. spaceevo@pixel4')
parser.add_argument('--align_sample', action='store_true', help='all blocks in a stage share the same kwe values')
parser.add_argument('--mode', default='acc', choices=['acc', 'lat'], help='evaluate accuracy or latency')
parser.add_argument('--quant_mode', action='store_true', help='evaluate quantized net')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--batch_size_per_gpu', type=int, default=32)
parser.add_argument('--resume', default='result/supernet_training')
parser.add_argument('--seed', default=0)
parser.add_argument('--dataset_dir', default='imagenet_path')
parser.add_argument('--data_loader_workers_per_gpu', default=4, type=int)
parser.add_argument('--augment', default='auto_augment_tf')
parser.add_argument('--valid_size', default=0, type=int)
parser.add_argument('--post_bn_calibration_batch_num', default=32, type=int)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--debug_batches', default=3, type=int)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--subnet_choice', default=None, type=str, nargs='*')
args = parser.parse_args()

logger = logging.get_logger(__name__)
logging.setup_logging(None)

args.distributed = args.local_rank != -1
if args.local_rank == -1:
    args.local_rank = 0

# get checkpoint path
if args.mode == 'acc':
    args.supernet_choice = args.supernet_choice[0]
    args.arch = args.superspace + '-' + args.supernet_choice + f'-align{int(args.align_sample)}'
    args.exp_name = args.arch
    args.resume = os.path.join(args.resume, args.exp_name, 'checkpoint.pth' if not args.quant_mode else 'lsq.pth') 

def main():
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # cudnn.deterministic = True
    # warnings.warn('You have chosen to seed training. '
    #                'This will turn on the CUDNN deterministic setting, '
    #                'which can slow down your training considerably! '
    #                'You may see unexpected behavior when restarting '
    #                'from checkpoints.')

    if args.distributed:
        dist.init_process_group(
            backend='nccl',
        )
        args.world_size = dist.get_world_size()
    else:
        args.world_size = 1
    args.gpu = args.local_rank  # local rank, local machine cuda id
    args.batch_size = args.batch_size_per_gpu
    args.batch_size_total = args.batch_size * args.world_size

    # set random seed, make sure all random subgraph generated would be the same
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed(args.seed)


    logger.info(f"Use GPU: {args.gpu}, world size {args.world_size}")

    # synchronize is needed here to prevent a possible timeout after calling
    # init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    args.rank = comm.get_rank()  # global rank
    torch.cuda.set_device(args.gpu)

    if args.mode == 'acc':
        eval_acc()
    else:
        eval_lat()


def eval_acc():
    # build model
    model = Supernet.build_from_str(f'{args.superspace}-{args.supernet_choice}')
    model.align_sample = args.align_sample

    if args.quant_mode:
        set_quant_mode(model)

    model.cuda(args.gpu)
    # use sync batchnorm
    if getattr(args, 'sync_bn', False):
        model.apply(
            lambda m: setattr(m, 'need_sync', True))

    if args.distributed:
        model = comm.get_parallel_model(model, args.gpu)  # local rank
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # load dataset, train_sampler: distributed
    logger.info(f'Start loading data {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
    train_loader, val_loader, test_loader, train_sampler = build_data_loader(args)
    if val_loader is None:
        val_loader = test_loader
        logger.info(f'Valid loader is None. Use test loader to do evaluation. len {len(val_loader)}')
    else:
        logger.info(f'len train loader and val loader: {len(train_loader)} {len(val_loader)}')
    logger.info(f'Finish loading data {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

    # load checkpoints
    saver.load_checkpoints(args, model, logger=logger)

    if not args.subnet_choice:
        # validate supernet model
        (max_net_acc1, min_net_acc1), _ = validate(
            train_loader, val_loader, model, criterion, args
        )

        # predict latency
        platform = 'tflite27_cpu_int8' if 'mobile' in args.superspace else 'onnx_lut'
        latency_predictor = LatencyPredictor(platform)
        model_without_ddp.set_max_subnet()
        max_latency = latency_predictor.predict_subnet(model_without_ddp.get_active_subnet().config)
        model_without_ddp.set_min_subnet()
        min_latency = latency_predictor.predict_subnet(model_without_ddp.get_active_subnet().config)
        print(max_latency, min_latency)

        if comm.is_master_process():
            with open('./tmp_eval.csv', 'a') as f:
                f.write(
                    f'{args.arch},{args.start_epoch},{max_net_acc1:.2f},{min_net_acc1:.2f},{max_latency:.2f},{min_latency:.2f}\n')
    else:
        for subnet_choice in args.subnet_choice:
            acc1, acc5, loss, flops, params = supernet_eval.validate_spec_subnet(train_loader, val_loader, model,
                                                                                 criterion, args, logger, subnet_choice)
            if comm.is_master_process():
                with open('./tmp_eval_specnet.csv', 'a') as f:
                    f.write(
                        f'{args.superspace}-{args.supernet_choice}-{subnet_choice},{acc1:.2f},{flops:.2f},{params:.2f}\n')


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
        eval_random_net=False
    )


def eval_lat():
    platform = 'tflite27_cpu_int8' if 'mobile' in args.superspace else 'onnx_lut'
    latency_predictor = LatencyPredictor(platform)

    if not args.subnet_choice:
        latency_dict = {}
        for supernet_choice_str in args.supernet_choice:
            supernet = Supernet.build_from_str(f'{args.superspace}-{supernet_choice_str}')
            supernet.align_sample = args.align_sample

            latency_list = []
            for _ in tqdm(range(1000)):
                supernet.sample_active_subnet()
                subnet_config = supernet.get_active_subnet_config()
                latency_list.append(latency_predictor.predict_subnet(subnet_config))
            latency_dict[supernet_choice_str] = latency_list
        output_path = f'results/eval_supernet/{args.superspace}-latency_cdf.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        draw_latency_cdf(latency_dict, output_path)

    else:
        supernet = Supernet.build_from_str(f'{args.superspace}-{args.supernet_choice[0]}')
        supernet.align_sample = args.align_sample

        for subnet_choice in args.subnet_choice:
            supernet.set_active_subnet(subnet_choice)
            lat = latency_predictor.predict_subnet(supernet.get_active_subnet_config(), verbose=True)
            print(f'{subnet_choice} {lat:.2f}ms')


def draw_latency_cdf(latency_dict: Dict[str, List], output_path):
    for supernet_choice_str, latency_list in latency_dict.items():
        latency_list.sort()
        y = np.arange(len(latency_list)) / float(len(latency_list))
        plt.plot(latency_list, y, label=supernet_choice_str)
    plt.xlabel('int8 latency')
    plt.title(f'{args.superspace} latency cdf')
    plt.legend()
    # plt.xlim([5, 60])
    plt.savefig(output_path)


if __name__ == '__main__':
    main()
