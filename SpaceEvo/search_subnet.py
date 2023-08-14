from __future__ import annotations
import argparse
import collections
import copy
from datetime import datetime
import errno
from functools import partial
import logging
import os
import random
import string
import time
from typing import List, Union

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from modules.training.dataset.imagenet_dataloader import build_imagenet_dataloader
from modules.search_space.superspace import get_superspace, get_available_superspaces
from modules.modeling.supernet import Supernet, SubnetChoiceConfig
from modules.modeling.network import QNetwork, QNetworkConfig
from modules.modeling.ops.lsq_plus import set_quant_mode
from modules.alphanet_training.evaluate.supernet_eval import calibrate_bn_params
from modules.alphanet_training.evaluate.imagenet_eval import validate_one_subnet
from modules.latency_predictor import LatencyPredictor

# ignore broken pipe error
from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
signal(SIGPIPE, SIG_IGN)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dataset_path', default='imagenet_path', type=str, help='imagenet dataset path')
    parser.add_argument('--output_path', default='./results/search_subnet/')
    parser.add_argument('--valid_size', default=10000, help='size of validation dataset')

    # --- supernet configs ----
    parser.add_argument('--superspace', choices=get_available_superspaces(), required=True, type=str)
    parser.add_argument('--supernet_choice', required=True, type=str, help='block choice of supernet, e.g. 032220')
    parser.add_argument('--checkpoint_path', default='results/teamdrive/supernet_training')
    # --- search configs ---
    parser.add_argument('--latency_constraint', type=int, required=True)
    parser.add_argument('--latency_delta', default=5, type=int)
    parser.add_argument("--arch_mutate_prob", default=0.3, type=float)
    parser.add_argument("--resolution_mutate_prob", default=0.5, type=float)
    parser.add_argument("--population_size", default=512, type=int)
    parser.add_argument("--max_time_budget", default=20, type=int)
    parser.add_argument("--parent_ratio", default=0.5, type=float)
    parser.add_argument("--mutation_size", default=128, type=float)
    parser.add_argument('--crossover_size', default=128, type=int)
    # --- other configs ---
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_calib_batches', default=20, type=int, help='num batches to calibrate bn params')
    parser.add_argument('--manual_seed', default=0, type=int)
    parser.add_argument('--debug', action='store_true', help='debug mode, only train and eval a small number of batches')
    parser.add_argument('--debug_batches', default=10)
    parser.add_argument('--verbose', action='store_true')
    parser.set_defaults(verbose=True)
    parser.add_argument('--use_testset', action='store_true')  # For upper-bound reference.
    parser.add_argument('--max_crossover_try', default=50, type=int, help='try crossover for max this times to avoid deadlock.')
    parser.add_argument('--test_top_num', default=10)
    args = parser.parse_args()
    
    args.supernet_encoding = args.superspace + '-' + args.supernet_choice
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.supernet_encoding + '-align0', 'lsq.pth')
    args.output_path = os.path.join(args.output_path, args.supernet_encoding)
    args.exp_name = f'{args.latency_constraint}ms' + ('_testset' if args.use_testset else '')
    args.log_path = os.path.join(args.output_path, args.exp_name + '.log') 
    args.local_log_path = os.path.join(args.exp_name + '_' + ''.join(random.choices(string.ascii_lowercase, k=6)) + '.log')
    args.print_freq = 10
    args.disable_dist = args.local_rank == -1
    if args.disable_dist:
        args.local_rank = 0
    args.distributed = not args.disable_dist
    args.device = torch.device('cuda', args.local_rank)
    args.gpu = args.device

    args.is_master_proc = args.local_rank == 0
    if args.debug:
        args.population_size = 10
        args.max_time_budget = 2
        args.parent_ratio = 0.5
        args.crossover_size = 3
        args.mutation_size = 2
    return args


args = get_args()

if args.is_master_proc:
    logging.basicConfig(level=logging.INFO, filename=args.local_log_path, format='%(asctime)s :: %(levelname)s :: %(message)s')

def log(text):
    try:
        if args.is_master_proc:
            logging.info(text)
            print(datetime.now().strftime('(%H:%M:%S)'), text)
            
    except IOError as e:
        if e.errno == errno.EPIPE:
            pass

def main():
    try:
        if args.local_rank == 0:
            os.makedirs(args.output_path, exist_ok=True)

        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        np.random.seed(args.manual_seed)

        if not args.disable_dist:
            dist.init_process_group(backend='nccl')

        # --- get supernet ---
        supernet = Supernet.build_from_str(args.supernet_encoding)
        state_dict = torch.load(args.checkpoint_path, map_location=args.device)
        patched_state_dict = {}
        for k, v in state_dict['state_dict'].items():
            patched_state_dict[k.replace('module.', '')] = v
        supernet.load_state_dict(patched_state_dict)
        log(f'Load state_dict from {args.checkpoint_path}, epoch {state_dict["epoch"]}')
        set_quant_mode(supernet)

        # --- build dataloader ---
        train_dataloader, valid_dataloader, test_dataloader, train_sampler = build_imagenet_dataloader(
            dataset_path=args.dataset_path, 
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            distributed=dist.is_initialized(),
            valid_size=args.valid_size,
            num_workers=8,
            augment='auto_augment_tf'
        )

        # --- search ---
        platform = 'tflite27_cpu_int8' if 'mobile' in args.superspace else 'onnx_lut'
        latency_predictor = LatencyPredictor(platform)
        accuracy_evaluator = AccuracyEvaluator(partial(evaluate, train_dataloader=train_dataloader, eval_dataloader=valid_dataloader if not args.use_testset else test_dataloader))

        evolution_finder = EvolutionFinder(supernet, latency_predictor, accuracy_evaluator)
        best_valid, best_info = evolution_finder.run_evolution_search()

        # --- print histroy
        log('===== History =====')
        evolution_finder.history.print()

        # --- best valid accuracy
        log('===== Best Valid Acc =====')
        log(str(best_valid))
        # --- test best arch ---
        log('===== Testing Best Arch =====')
        indv_list = evolution_finder.history.indv_list
        indv_list.sort(reverse=True)
        for indv in indv_list[:args.test_top_num]:
            log('-' * 20)
            log(f'Testing {indv}')
            supernet.set_active_subnet(indv.arch)
            test_acc1, flops, params = evaluate(supernet.get_active_subnet(), train_dataloader, test_dataloader)
            log(f'Test Acc1: {test_acc1}')

    finally:
        if args.is_master_proc:
            os.system(f'cp {args.local_log_path} {args.log_path}')


def evaluate(subnet: QNetwork, train_dataloader, eval_dataloader):
    subnet.to(args.device)
    if args.distributed:
        subnet = torch.nn.parallel.DistributedDataParallel(subnet, [args.device])
    subnet.eval()
    set_quant_mode(subnet)
    calibrate_bn_params(subnet, train_dataloader, args.num_calib_batches, args.device)
    acc1, *_, flops, params = validate_one_subnet(eval_dataloader, subnet, torch.nn.CrossEntropyLoss(), args, slient=True)
    return round(acc1, 3), flops, params


class Individual:

    def __init__(self, arch, acc, latency, flops, params) -> None:
        self.arch = arch
        self.acc = acc
        self.latency = latency
        self.flops = flops
        self.params = params

    def __lt__(self, other):
        return self.acc < other.acc
    
    def __str__(self) -> str:
        return f'arch:{self.arch}-acc:{self.acc:.2f}-latency:{self.latency:.2f}-mflops:{self.flops:.2f}-params:{self.params:.2f}'

    def __eq__(self, other) -> bool:
        if isinstance(other, Individual):
            return self.arch == other.arch
        if isinstance(other, SubnetChoiceConfig):
            return self.arch == other
        raise ValueError(other)


class History:

    def __init__(self) -> None:
        self.indv_list = []
        self.arch_list = []

    def clear(self):
        self.indv_list = []
        self.arch_list = [] 

    def __contains__(self, other: Union[Individual, SubnetChoiceConfig]):
        if isinstance(other, Individual):
            return other.arch in self.arch_list
        if isinstance(other, SubnetChoiceConfig):
            return other in self.arch_list
        raise ValueError()

    def print(self):
        for i, indv in enumerate(self.indv_list):
            log(f'indv-{i}: {indv}')

    def add(self, v: Individual):
        self.indv_list.append(v)
        self.arch_list.append(v.arch)


class AccuracyEvaluator:

    def __init__(self, eval_func) -> None:
        self.eval_func = eval_func

    def eval(self, subnet):
        return self.eval_func(subnet)


class EvolutionFinder:

    def __init__(self, supernet: Supernet, latency_predictor: LatencyPredictor, accuracy_evaluator) -> None:
        self.supernet = supernet 
        self.latency_predictor = latency_predictor
        self.accuracy_evaluator = accuracy_evaluator
        self.history = History()

    def _get_valid_indv(self, subnet_choice):
        subnet_config = self.supernet.get_active_subnet_config()
        latency = self.latency_predictor.predict_subnet(subnet_config)
        if latency <= args.latency_constraint and latency >= args.latency_constraint - args.latency_delta:
            subnet = self.supernet.get_active_subnet()
            accuracy, flops, params = self.accuracy_evaluator.eval(subnet)
            indv = Individual(subnet_choice, accuracy, latency, flops, params)
            self.history.add(indv)
            return True, indv
        else:
            return False, None
    
    def random_valid_sample(self):
        while True:
            subnet_choice = self.supernet.sample_active_subnet()
            while subnet_choice in self.history:
                subnet_choice = self.supernet.sample_active_subnet()

            is_valid, indv = self._get_valid_indv(subnet_choice)
            if is_valid:
                return indv 

    def mutate_valid_sample(self, subnet_choice):
        assert subnet_choice in self.history
        while True:
            sc = self.supernet.mutate(subnet_choice, args.arch_mutate_prob, args.resolution_mutate_prob)
            while sc in self.history:
                sc = self.supernet.mutate(subnet_choice, args.arch_mutate_prob, args.resolution_mutate_prob)

            is_valid, indv = self._get_valid_indv(sc)
            if is_valid:
                return indv 

    def crossover_valid_sample(self, sc1, sc2):
        assert sc1 in self.history and sc2 in self.history
        assert sc1 != sc2
        for _ in range(args.max_crossover_try):
            subnet_choice = self.supernet.crossover(sc1, sc2)
            if subnet_choice in self.history:
                continue
            is_valid, indv = self._get_valid_indv(subnet_choice)
            if is_valid:
                return indv 
        return None 
        
    def run_evolution_search(self):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        self.history.clear()
        mutation_numbers = args.mutation_size
        parents_size = int(round(args.parent_ratio * args.population_size))

        best_valids = [-100]
        population = []  # list of indvs
        best_info = None
        if args.verbose:
            log('=' * 100)
            log("Generate random population...")
        for i in range(args.population_size):
            population.append(self.random_valid_sample())
            if args.verbose:
                log(f'[population {i:3d}]: {population[-1]}')

        if args.verbose:
            log('=' * 100)
            log("Start Evolution...")
        # After the population is seeded, proceed with evolving the population.
        with tqdm(
            total=args.max_time_budget,
            desc="Searching with constraint (%s)" % args.latency_constraint,
            disable=(not args.verbose),
        ) as t:
            for i in range(args.max_time_budget):
                parents = sorted(population, reverse=True)[:parents_size]
                acc = parents[0].acc
                t.set_postfix({"acc": parents[0].acc})
                if args.verbose:
                    log('='*50)
                    log(f"[Iter {i + 1} Best]: {parents[0]}")

                if acc > best_valids[-1]:
                    best_valids.append(acc)
                    best_info = parents[0]
                else:
                    best_valids.append(best_valids[-1])

                population = parents

                for j in range(mutation_numbers):
                    idx = np.random.randint(parents_size)
                    par_sample = population[idx].arch
                    # Mutate
                    indv = self.mutate_valid_sample(par_sample)
                    population.append(indv)

                    log(f'*** mutate {j} ***')
                    log(population[idx])
                    log(indv)

                for j in range(args.crossover_size):
                    idx1, idx2 = np.random.choice(parents_size, 2, replace=False)
                    idx1, idx2 = sorted([idx1, idx2])
                    par_sample1 = population[idx1].arch
                    par_sample2 = population[idx2].arch
                    
                    log(f'*** crossover {j} ***')
                    log(population[idx1])
                    log(population[idx2])
                    
                    # Crossover
                    indv = self.crossover_valid_sample(par_sample1, par_sample2)
                    if indv is None:
                        log(f'Crossover reach max {args.max_crossover_try} trys. Mutate from par1 instead.')
                        indv = self.mutate_valid_sample(par_sample1)
                    population.append(indv)

                    log(indv)
                
                t.update(1)

                if args.verbose:
                    log('-' * 50)
                    log(f'Iter {i + 1} new population')
                    for j, indv in enumerate(population):
                        log(f'[indv {j:3d}]{indv}')

        return best_valids, best_info


if __name__ == '__main__':
    main()