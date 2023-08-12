# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# based on AlphaNet

import argparse
import os
import sys
import random
from datetime import date, datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

FILE_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(FILE_DIR, '..'))
from modules.search_space.superspace import get_superspace, get_available_superspaces
from modules.modeling.supernet import Supernet
from modules.alphanet_training.data.data_loader import build_data_loader
import modules.alphanet_training.utils.saver as saver
import modules.alphanet_training.utils.comm as comm
import modules.alphanet_training.utils.logging as logging
from modules.alphanet_training.evaluate import supernet_eval
from modules.latency_predictor import LatencyPredictor
from modules.latency_benchmark.onnx.export_onnx import export_onnx_fix_batch, export_onnx
from modules.latency_benchmark.onnx.quant_onnx import onnx_quantize_static_pipeline

from onnx_tools.common import onnx_model_dir, latency_output_csv_path, get_model_name, get_input_subnets

parser = argparse.ArgumentParser(description='Supernet sandwich rule training.')
# parser.add_argument('--superspace', choices=get_available_superspaces(), required=True, type=str)
# parser.add_argument('--supernet_choice', type=str, help='candidate of superspace, e.g. 322223', default='322223')
# parser.add_argument('--subnet_choice', default=None, type=str, nargs='*')
parser.add_argument('--align_sample', action='store_true', help='all blocks in a stage share the same kwe values')
parser.add_argument('--quant_mode', action='store_true', )
parser.add_argument('--batch_size_per_gpu', type=int, default=32)
parser.add_argument('--resume', default='./debug/supernet_training')
parser.add_argument('--seed', default=0)
parser.add_argument('--dataset_dir', default='path_to_imagenet')
parser.add_argument('--data_loader_workers_per_gpu', default=4, type=int)
parser.add_argument('--augment', default='auto_augment_tf')
parser.add_argument('--valid_size', default=0, type=int)
parser.add_argument('--post_bn_calibration_batch_num', default=32, type=int)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--debug_batches', default=3, type=int)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--skip_weights', action='store_true')
args = parser.parse_args()

logger = logging.get_logger(__name__)
logging.setup_logging(None)


def main():
    args.batch_size = args.batch_size_per_gpu

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for superspace, supernet_choice, subnet_choice in get_input_subnets():
        print(superspace, supernet_choice)
        model = Supernet.build_from_str('-'.join([superspace, supernet_choice]))
        model.align_sample = args.align_sample

        if not args.skip_weights:
            # load dataset, train_sampler: distributed
            logger.info(f'Start loading data {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
            train_loader, val_loader, test_loader, train_sampler = build_data_loader(args)
            if val_loader is None:
                val_loader = test_loader
                logger.info(f'Valid loader is None. Use test loader to do evalution. len {len(val_loader)}')
            else:
                logger.info(f'len train loader and val loader: {len(train_loader)} {len(val_loader)}')
            logger.info(f'Finish loading data {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

            exp_name = superspace + '-' + supernet_choice + f'-align{int(args.align_sample)}'
            args.resume = os.path.join(args.resume, exp_name, 'checkpoint.pth' if not args.quant_mode else 'lsq.pth')
            saver.load_checkpoints(args, model, logger=logger)

        supernet = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

        supernet.set_active_subnet(subnet_choice)
        subnet = supernet.get_active_subnet()

        if not args.skip_weights:
            subnet.to('cuda')
            supernet_eval.calibrate_bn_params(subnet, train_loader, args.post_bn_calibration_batch_num, device='cuda')
            subnet.to('cpu')
        subnet.eval()

        model_name = get_model_name(superspace, supernet_choice, subnet_choice)
        input_shape = [1, 3, subnet.resolution, subnet.resolution]
        fp32_path = os.path.join(onnx_model_dir, f'{model_name}.onnx')
        int8_path = fp32_path.replace('.onnx', '_quant_int8.onnx')
        export_onnx_fix_batch(subnet, os.path.join(onnx_model_dir, f'{model_name}.onnx'), input_shape)
        onnx_quantize_static_pipeline(fp32_path, int8_path,
                                      per_channel=True)  # int8 latencies of per_channel and per_tensor differ a lot


if __name__ == '__main__':
    main()
