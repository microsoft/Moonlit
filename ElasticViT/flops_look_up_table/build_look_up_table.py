# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import pickle
import sys 

sys.path.append("..")
from util import select_min_arch, parse_supernet_configs
from models import build_supernet
from pathlib import Path
from utils import get_config
import random

random.seed(2023)
num_of_subnets = 150000

# 3min
# min_archs = {
#     100: (128, (16, 16, 16, 48, 80, 144, 160), (1, 2, 2, 1, 1, 1, 1), (1, 3, 3, 3, 3), (3, 3, 3, 3, 3), (2, 2, 2, 2), (3, 5, 9, 10), (1, 1, 1, 1), (1, 1, 1, 1), (2, 2, 2, 2)), 
#     300: (176, (16, 24, 32, 48, 96, 176, 224), (1, 3, 3, 2, 2, 2, 2), (1, 3, 3, 3, 3, 3, 3), (3, 3, 3, 3, 3, 3, 3), (2, 2, 3, 3, 2, 2, 2, 2), (3, 3, 6, 6, 11, 11, 14, 14), (1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1), (2, 2, 2, 2, 2, 2, 2, 2)), 
#     800: (224, (16, 24, 32, 64, 96, 192, 256), (1, 3, 3, 2, 2, 2, 2), (1, 3, 3, 3, 3, 3, 3), (3, 3, 3, 3, 3, 3, 3), (2, 2, 2, 2, 2, 2, 2, 2), (4, 4, 6, 6, 12, 12, 16, 16), (1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1), (2, 2, 2, 2, 2, 2, 2, 2))
# }

args = get_config(default_file='../configs/final_3min_space.yaml')

min_arch = select_min_arch(flops=args.flops, min_archs=args.search_space.min_archs)
parse_supernet_configs(args, min_arch=min_arch)

model, *_ = build_supernet(args)

archs = []
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(n_parameters/1e6)

print(model.arch_sampling('min'))
print(f'min model flops {model.compute_flops()} M')

print(model.arch_sampling('max'))
print(f'max model flops {model.compute_flops()} M')

flops_table = []
target_flops = args.flops
seed = 1
while len(flops_table) < num_of_subnets:
    arch = model.arch_sampling_engine(mode='uniform', mumate_prob=1., res_mutate_prob=1.)
    flops = model.compute_flops()

    if target_flops <= flops <= target_flops + 30 and arch not in flops_table:

        seed +=1
        random.seed(seed)
        print(arch)
        flops_table.append(arch)

        if len(flops_table) % 1000 == 0:
            with open(f"flops_{target_flops}.pkl", 'wb') as f:
                pickle.dump(flops_table, f)