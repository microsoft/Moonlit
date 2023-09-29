# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import logging
import logging.config
import os
import time
from pathlib import Path
import munch
import yaml


def merge_nested_dict(d, other):
    new = dict(d)
    for k, v in other.items():
        if d.get(k, None) is not None and type(v) is dict:
            new[k] = merge_nested_dict(d[k], v)
        else:
            new[k] = v
    return new


def get_config(default_file):
    p = argparse.ArgumentParser(description='Learned Step Size Quantization')
    p.add_argument('config_file', metavar='PATH', nargs='+',
                   help='path to a configuration file')
    p.add_argument("--local_rank", default=0, type=int)
    p.add_argument("--gpu_no", default=0, type=int)
    p.add_argument("--flops", default=500, type=int)
    p.add_argument('--mixup', type=float, default=0.01,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    p.add_argument('--cutmix', type=float, default=0.01,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    p.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    p.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    p.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    p.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    p.add_argument('--bank_id', nargs='+', help='<Required> Set flag', type=int,)
    p.add_argument('--mode', type=str, choices=['min', 'bank_s', 'bank'], default='min')
    arg = p.parse_args()

    with open(default_file) as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    for f in arg.config_file:
        if not os.path.isfile(f):
            raise FileNotFoundError('Cannot find a configuration file at', f)
        with open(f) as yaml_file:
            c = yaml.safe_load(yaml_file)
            cfg = merge_nested_dict(cfg, c)
    args = munch.munchify(cfg)
    args.local_rank = arg.local_rank
    args.gpu_no = arg.gpu_no
    args.mixup = arg.mixup
    args.cutmix = arg.cutmix
    args.cutmix_minmax = arg.cutmix_minmax
    args.mixup_prob = arg.mixup_prob
    args.mixup_switch_prob = arg.mixup_switch_prob
    args.mixup_mode = arg.mixup_mode
    args.flops = arg.flops
    args.mode = arg.mode
    args.bank_id = arg.bank_id

    return args


def init_logger(experiment_name, output_dir, cfg_file=None, pre=''):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    exp_full_name = time_str if experiment_name is None else pre + experiment_name + '_' + time_str
    log_dir = output_dir / exp_full_name
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / (exp_full_name + '.log')
    logging.config.fileConfig(cfg_file, defaults={'logfilename': str(log_file)})
    logger = logging.getLogger()
    logger.info('Log file for this run: ' + str(log_file))
    return log_dir
