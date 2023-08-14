# based on AlphaNet: https://github.com/facebookresearch/AlphaNet

import argparse
import os
import random
from datetime import date, datetime
import yaml

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from modules.modeling.ops.lsq_plus import set_quant_mode
from modules.modeling.supernet import Supernet
from modules.alphanet_training.data.data_loader import build_data_loader
import modules.alphanet_training.utils.saver as saver
import modules.alphanet_training.utils.comm as comm
import modules.alphanet_training.utils.logging as logging
from modules.alphanet_training.evaluate import supernet_eval

import numpy as np


# load search_result
SEARCH_RESULT_PATH = './data/search_result.yaml'
y = yaml.safe_load(open(SEARCH_RESULT_PATH, 'r').read())
specific_supernet_encoding_dict = y['specific_supernet_encoding_dict']
specific_subnet_encoding_dict = y['specific_subnet_encoding_dict']


parser = argparse.ArgumentParser(description='Evaluate supernet and subnet.')
parser.add_argument('--model_name', required=True, type=str,
                    help='specific network name to evaluate, '
                         'e.g. "spaceevo@pixel4" (supernet) or "SEQnet@vnni-A0" (subnet)')
parser.add_argument('--quant_mode', action='store_true', help='evaluate quantized net')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--batch_size_per_gpu', type=int, default=32)
parser.add_argument('--resume', default='./checkpoints/supernet_training')
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

# get supernet encoding
is_supernet = args.model_name.startswith('spaceevo@')
if is_supernet:
    supernet_name = args.model_name
    supernet_encoding = specific_supernet_encoding_dict[args.model_name]
    subnet_encoding = None
else:
    if 'pixel4' in args.model_name:
        supernet_name = 'spaceevo@pixel4'
    elif 'vnni' in args.model_name:
        supernet_name = 'spaceevo@vnni'
    else:
        raise ValueError(args.model_name)
    supernet_encoding = specific_supernet_encoding_dict[supernet_name]
    subnet_encoding = specific_subnet_encoding_dict[args.model_name][1]

# get checkpoint path
args.resume = os.path.join(args.resume, supernet_name, 'checkpoint.pth' if not args.quant_mode else 'lsq.pth')


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

    eval_accuracy()


def eval_accuracy():
    # build model
    model = Supernet.build_from_str(supernet_encoding)

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

    if is_supernet:
        # validate supernet model
        (max_net_acc1, min_net_acc1), _ = validate_supernet(train_loader, val_loader, model, criterion, args)

        if comm.is_master_process():
            print(f'{args.model_name}: max_net_acc={max_net_acc1:.2f}, min_net_acc={min_net_acc1:.2f}')
    else:
        acc1, acc5, loss, flops, params = supernet_eval.validate_spec_subnet(train_loader, val_loader, model,
                                                                             criterion, args, logger, subnet_encoding)
        if comm.is_master_process():
            print(f'{args.model_name}: acc1={acc1:.2f}, flops={flops:.2f} params={params:.2f}')


def validate_supernet(
        train_loader,
        val_loader,
        model,
        criterion,
        args,
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


if __name__ == '__main__':
    main()
