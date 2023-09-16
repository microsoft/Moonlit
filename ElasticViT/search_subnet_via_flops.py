# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import torch
import yaml
import os
import pickle
import timm
import random
import numpy as np
from pathlib import Path
from process import bn_cal, train, validate, PerformanceScoreboard
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import distribute_bn
from timm.data import Mixup
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel
from models import build_supernet
from torch.distributed import get_rank
from util import get_config, load_checkpoint, load_data_dist, parse_supernet_configs, select_min_arch
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
# torch.backends.cudnn.enabled = False
# from nn_meter.predictor.transformer_predictor import BlockLatencyPredictor
# predictor = BlockLatencyPredictor("pixel6_lut")

def get_args_parser():
    parser = argparse.ArgumentParser(
        'elasticvit searching script', add_help=False)
    parser.add_argument('config_file', metavar='PATH', nargs='+',
                help='path to a configuration file')

    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--flops_limits', default=300, type=int)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
   
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser


class EvolutionSearcher(object):

    def __init__(self, model, model_without_ddp, train_loader, val_loader, test_loader, output_dir, cfg, mixup_fn, dim_mutation_prob=0.45, target_models=None):
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.max_epochs = 40
        self.select_num = 256
        self.population_num = 64
        self.m_prob = 0.2
        self.crossover_num = 128
        self.mutation_num = 128
        self.flops_limits = cfg.flops_limits + cfg.bound*2
        self.min_flops_limits = cfg.flops_limits
        self.target_limits = cfg.flops_limits
        self.bound = cfg.bound
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.dim_mutation_prob = dim_mutation_prob
        self.target_models = target_models

        if not os.path.exists(output_dir) and get_rank() == 0:
            os.makedirs(output_dir, exist_ok=True)
        self.s_prob = 0.4
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        # self.checkpoint_path = args.resume
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        # self.choices = choices
        self.conv_stage = cfg.conv_stage
        self.cfg = cfg
        self.train_loader = train_loader
        self.mixup_fn = mixup_fn
        self.fixed_head_dim = True
        self.archs = {}
        self.inp = torch.randn((5,3,224,224)).to(self.cfg.device)
        self.search_mode = 0 # 0 for stage_wise, 1 for layer_wise

    def save_checkpoint(self):
        info = {}
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        info['archs'] = self.archs
        checkpoint_path = os.path.join(self.output_dir, "checkpoint-{}.pth.tar".format(self.epoch))
        if get_rank() == 0:
            torch.save(info, checkpoint_path)
        print('save checkpoint to', checkpoint_path)

    def load_checkpoint(self):
        checkpoint_path = None
        for epoch in range(self.max_epochs-1, -1, -1):
            checkpoint_path = os.path.join(self.output_dir, "checkpoint-{}.pth.tar".format(epoch))
            if os.path.exists(checkpoint_path):
                info = torch.load(checkpoint_path)
                self.memory = info['memory']
                self.candidates = info['candidates']
                self.vis_dict = info['vis_dict']
                self.keep_top_k = info['keep_top_k']
                self.epoch = info['epoch']

                print('load checkpoint from', checkpoint_path)

                return True
        return False

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        
        # depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
        _res, channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale = cand

        stem_depths = depths[0]
        depths = depths[1:]

        stem_channels = channels[0]
        channels = channels[1:]

        conv_ratio = conv_ratio[stem_depths:]
        kr_size = kr_size[stem_depths:]

        conv_depths = sum(depths[:self.conv_stage])
        transformer_depths = sum(depths[self.conv_stage:])

        if _res > 256 or (_res == 160 and self.flops_limits >= 380):
            return False

        if conv_depths != len(kr_size) or conv_depths!= len(conv_ratio):
            return False
        
        if transformer_depths != len(mlp_ratio) or transformer_depths!= len(num_heads) or transformer_depths!= len(qk_scale) or transformer_depths!= len(v_scale) or transformer_depths!= len(window_size):
            return False
        
        transformer_depths_settings = depths[self.conv_stage:]
        
        for idx, s in enumerate(transformer_depths_settings):
            offset = sum(transformer_depths_settings[:idx])
            ws = window_size[offset:offset + s]
            res_idx = self.model_without_ddp.res_range.index(_res)
            stage_wise_res = self.model_without_ddp.feature_exactor[self.conv_stage + idx].res[res_idx]
            # stage_wise_res = self.model_without_ddp.stage_wise_res[self.conv_stage + idx][res_idx]
            for wss in ws:
                if wss > stage_wise_res or stage_wise_res%wss != 0:
                    print('c')
                    return False
            
            num_hds = num_heads[offset:offset + s]

            if not self.fixed_head_dim:
                heads_op = self.model_without_ddp.feature_exactor[self.conv_stage + idx].num_heads

                for head in num_hds:
                    if head not in heads_op:
                        return False
            else:
                for head in num_hds:
                    if head != channels[self.conv_stage+idx]//16:
                        num_heads[offset:offset + s] = [channels[self.conv_stage+idx]//16] * s
        
        # print(channels,depths, num_heads)
        _res, channels, depths, conv_ratio, kr_size, mlp_ratio, _, window_size, qk_scale, v_scale = cand
        self.model_without_ddp.set_arch(_res, channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale)
        # _aa = (_res, (0, 0, 0, 1, 1, 1), channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale, (False, True, True, False, False, False))
        # latency = predictor.get_latency(_aa, strides=[2,2,2,2,1,2])
        # n_parameters = self.model_without_ddp.get_sampled_params_numel(sampled_config)
        # info['params'] =  n_parameters / 10.**6

        p_arch = (_res, channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale)
        # latency = predictor.get_latency(p_arch)
        latency = self.model_without_ddp.compute_flops()
        info['latency'] = latency

        if info['latency'] > self.flops_limits:
            # print(latency, 'latency limit exceed')
            return False

        if info['latency'] < self.min_flops_limits:
            # print('under minimum latency limit')
            return False

        with torch.no_grad():
            bn_cal(self.model, self.train_loader, self.cfg, num_batches=64, mixup_fn=self.mixup_fn)
        
        acc1_test, _, _ = validate(self.test_loader, self.model, None, 0, None, self.cfg, None)

        print("rank:", self.cfg.rank, cand, info['latency'], acc1_test)
        
        acc1_val = acc1_test
        
        info['acc'] = acc1_val
        info['test_acc'] = acc1_test

        info['visited'] = True

        if self.epoch not in self.archs:
            self.archs[self.epoch] = []
        
        self.archs[self.epoch].append([cand, acc1_val, latency])

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random_cand(self):
        if self.target_models is None:
            arch = self.model_without_ddp.arch_sampling()
        else:
            arch = random.choice(self.target_models)
        return tuple(arch)

    def get_random(self, num, init_from_bank=False):
        if not init_from_bank:
            print('random select ........')
            
            cand_iter = self.stack_random_cand(self.get_random_cand)
            while len(self.candidates) < num:
                cand = next(cand_iter)
                if not self.is_legal(cand):
                    continue
                self.candidates.append(cand)
                print('random {}/{}'.format(len(self.candidates), num))
        else:
            target_flops = self.target_limits
            bank_id = self.model_without_ddp.bank_flops_ranges.index(target_flops)
            bank = self.model_without_ddp.banks[bank_id]
            arch_list, _ = self.model_without_ddp._get_bank_info(bank)
            iter_num = 0
            
            last_parents = num - len(arch_list)
            cand_iter = self.stack_random_cand(self.get_random_cand)
            while len(self.candidates) < last_parents:
                cand = next(cand_iter)
                if not self.is_legal(cand):
                    continue
                self.candidates.append(cand)
                print('random {}/{}'.format(len(self.candidates), num))
            
            for arch_idx in range(len(arch_list)):
                if iter_num >= num:
                    break

                arch = list(arch_list[arch_idx])

                for sub_dims_id in range(len(arch)):
                    if isinstance(arch[sub_dims_id], list):
                        arch[sub_dims_id] = tuple(arch[sub_dims_id])
                    
                arch = tuple(arch)

                # if arch not in self.vis_dict:
                #     self.vis_dict[arch] = {}
                if not self.is_legal(arch):
                    continue
                
                self.candidates.append(arch)
                print('random {}/{}, bank_arch_idx {}'.format(len(self.candidates), num, arch_idx))

                iter_num += 1

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def prob_mutation(choices):
            if random.random() < self.dim_mutation_prob:
                return random.choice(choices)
            else:
                return min(choices)

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            _res, channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale = cand
            
            first_conv_depths, first_conv_channels = depths[0], channels[0]

            depths, channels = depths[1:], channels[1:]
            conv_ratio = conv_ratio[first_conv_depths:]
            kr_size = kr_size[first_conv_depths:]
            
            random_s = random.random()

            # depth
            if random_s < s_prob:
                # _, _, new_depths, new_conv_ratio, new_kr_size, new_mlp_ratio, new_num_heads, new_window_size, new_qk_scale, new_v_scale = self.model_without_ddp.arch_sampling()
                _conv_ratio, _kr_size, _mlp_ratio, _num_heads, _windows_size, _qk_scale, _v_scale, _channels = [], [], [], [], [], [], [], []

                # _first_conv_depths = new_depths[0]
                # new_depths = new_depths[1:]
                # new_conv_ratio = new_conv_ratio[_first_conv_depths:]
                # new_kr_size = new_kr_size[_first_conv_depths:]
                new_depths = []
                for idx in range(len(self.model_without_ddp.depths)):
                    choices = [d for d in range(self.model_without_ddp.min_depths[idx], self.model_without_ddp.depths[idx])]
                    new_depths.append(prob_mutation(choices=choices))

                for stage_idx, new_depth in enumerate(new_depths):
                    if depths[stage_idx] < new_depth:
                        if stage_idx < self.conv_stage:
                            start, end = sum(depths[:stage_idx]), sum(depths[:stage_idx+1])
                            _conv_ratio += conv_ratio[start:end]
                            _kr_size += kr_size[start:end]

                            if self.search_mode == 0:
                                num_new_layers = new_depth - depths[stage_idx]
                                
                                _conv_ratio += num_new_layers * [conv_ratio[start:end][0]]
                                _kr_size += num_new_layers * [kr_size[start:end][0]]
                            else:
                                raise NotImplementedError
                        else:
                            start = sum(depths[:stage_idx]) - sum(depths[:self.conv_stage])
                            end = sum(depths[:stage_idx+1]) - sum(depths[:self.conv_stage])

                            _mlp_ratio += mlp_ratio[start:end]
                            _num_heads += num_heads[start:end]
                            _windows_size += window_size[start:end]
                            _qk_scale += qk_scale[start:end]
                            _v_scale += v_scale[start:end]

                            if self.search_mode == 0:
                                num_new_layers = new_depth - depths[stage_idx]

                                _mlp_ratio += num_new_layers * [mlp_ratio[start:end][0]]
                                _num_heads += num_new_layers * [num_heads[start:end][0]]
                                _windows_size += num_new_layers * [window_size[start:end][0]]
                                _qk_scale += num_new_layers * [qk_scale[start:end][0]]
                                _v_scale += num_new_layers * [v_scale[start:end][0]]
                            else:
                                raise NotImplementedError
                    else:
                        if stage_idx < self.conv_stage:
                            start = sum(depths[:stage_idx])
                            end = start + new_depth

                            _conv_ratio += conv_ratio[start:end]
                            _kr_size += kr_size[start:end]
                        
                        else:
                            start = sum(depths[:stage_idx]) - sum(depths[:self.conv_stage])
                            end = start + new_depth 

                            _mlp_ratio += mlp_ratio[start:end]
                            _num_heads += num_heads[start:end]
                            _windows_size += window_size[start:end]
                            _qk_scale += qk_scale[start:end]
                            _v_scale += v_scale[start:end]
                
                old_depths = depths
                mlp_ratio = _mlp_ratio
                num_heads = _num_heads
                window_size = _windows_size
                qk_scale = _qk_scale
                v_scale = _v_scale
                depths = new_depths
                conv_ratio = _conv_ratio
                kr_size = _kr_size
            
            mlp_ratio, num_heads, window_size, qk_scale, v_scale, depths, conv_ratio, kr_size = list(mlp_ratio), list(num_heads), list(window_size), list(qk_scale), list(v_scale), \
                list(depths), list(conv_ratio), list(kr_size)
            
            # conv ratio
            for i in range(self.conv_stage):
                if self.search_mode == 0:
                    random_s = random.random()
                    if random_s < m_prob:
                        ratio = prob_mutation(choices=self.model_without_ddp.conv_ratio)
                        for j in range(depths[i]):
                            idx = sum(depths[:i]) + j
                            conv_ratio[idx] = ratio
                else:
                    for j in range(depths[i]):
                        random_s = random.random()
                        if random_s < m_prob:
                            idx = sum(depths[:i]) + j
                            conv_ratio[idx] = prob_mutation(choices=self.model_without_ddp.conv_ratio)

            # kr_size
            for i in range(self.conv_stage):
                if self.search_mode == 0:
                    random_s = random.random()
                    if random_s < m_prob:
                        krs = prob_mutation(choices=[3, 5])

                        for j in range(depths[i]):
                            idx = sum(depths[:i]) + j
                            kr_size[idx] = krs
                else:
                    for j in range(depths[i]):
                        random_s = random.random()
                        if random_s < m_prob:
                            idx = sum(depths[:i]) + j
                            kr_size[idx] = prob_mutation(choices=[3, 5])

            # mlp_ratio
            for i in range(len(depths) - self.conv_stage):
                if self.search_mode == 0:
                    random_s = random.random()
                    if random_s < m_prob:
                        mlprt = prob_mutation(choices=self.model_without_ddp.mlp_ratio)
                        
                        for j in range(depths[self.conv_stage + i]):
                            idx = sum(depths[self.conv_stage:self.conv_stage + i]) + j
                            mlp_ratio[idx] = mlprt
                else:
                    for j in range(depths[self.conv_stage + i]):
                        random_s = random.random()
                        if random_s < m_prob:
                            idx = sum(depths[self.conv_stage:self.conv_stage + i]) + j
                            mlp_ratio[idx] = prob_mutation(choices=self.model_without_ddp.mlp_ratio)
            
            # qk_scale
            for i in range(len(depths) - self.conv_stage):
                if self.search_mode == 0:
                    random_s = random.random()
                    if random_s < m_prob:
                        qks = prob_mutation(choices=self.model_without_ddp.qk_scale[i])
                        
                        for j in range(depths[self.conv_stage + i]):
                            idx = sum(depths[self.conv_stage:self.conv_stage + i]) + j
                            qk_scale[idx] = qks
                else:
                    for j in range(depths[self.conv_stage + i]):
                        random_s = random.random()
                        if random_s < m_prob:
                            idx = sum(depths[self.conv_stage:self.conv_stage + i]) + j
                            qk_scale[idx] = prob_mutation(choices=self.model_without_ddp.qk_scale[i])
            
            # v_scale
            for i in range(len(depths) - self.conv_stage):
                if self.search_mode == 0:
                    random_s = random.random()
                    if random_s < m_prob:
                        vs = prob_mutation(choices=self.model_without_ddp.v_scale[i])
                        for j in range(depths[self.conv_stage + i]):
                            idx = sum(depths[self.conv_stage:self.conv_stage + i]) + j
                            v_scale[idx] = vs
                else:
                    for j in range(depths[self.conv_stage + i]):
                        random_s = random.random()
                        if random_s < m_prob:
                            idx = sum(depths[self.conv_stage:self.conv_stage + i]) + j
                            v_scale[idx] = prob_mutation(choices=self.model_without_ddp.v_scale[i])
            
            # channels
            channels = list(channels)
            for i in range(len(depths)):
                random_s = random.random()
                if random_s < m_prob:
                    choices = [c for c in range(self.model_without_ddp.min_channels[i], self.model_without_ddp.channels[i]+1, 8 if i > self.conv_stage else 16)]
                    while True:
                        channel = prob_mutation(choices=choices)

                        if i == 0 or channel >= channels[i-1]:
                            channels[i] = channel
                            break
            # num_heads
            for i in range(len(depths) - self.conv_stage):
                for j in range(depths[self.conv_stage + i]):
                    if self.fixed_head_dim:
                        idx = sum(depths[self.conv_stage:self.conv_stage + i]) + j
                        num_heads[idx] = channels[self.conv_stage + i]//16
                    else:
                        raise NotImplementedError
            
            random_s = random.random()

            if random_s < m_prob:
                _new_res = random.choice(self.model_without_ddp.res_range)

                if _new_res != _res:
                    _new_res_idx = self.model_without_ddp.res_range.index(_new_res)
                    
                    for i in range(len(depths) - self.conv_stage):
                        for j in range(depths[self.conv_stage + i]):
                            # random_s = random.random()
                            
                            idx = sum(depths[self.conv_stage:self.conv_stage + i]) + j

                            # stage_wise_res = self.model_without_ddp.feature_exactor[self.conv_stage + i].res[res_idx]
                            window_size[idx] = self.model_without_ddp.feature_exactor[self.conv_stage + i].windows_size[_new_res_idx][0]
            
            depths.insert(0, first_conv_depths)
            channels.insert(0, first_conv_channels)
            for _ in range(first_conv_depths):
                conv_ratio.insert(0, 1)
                kr_size.insert(0, 3)

            arch = (_res, tuple(channels), tuple(depths), tuple(conv_ratio), tuple(kr_size), tuple(mlp_ratio), tuple(num_heads), tuple(window_size), tuple(qk_scale), tuple(v_scale))
            print(arch)
            return arch

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            max_iters_tmp = 50

            while len(p1) != len(p2) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
            
            p1_res, p1_channels, p1_depths, p1_conv_ratio, p1_kr_size, p1_mlp_ratio, p1_num_heads, p1_window_size, p1_qk_scale, p1_v_scale = p1
            p2_res, p2_channels, p2_depths, p2_conv_ratio, p2_kr_size, p2_mlp_ratio, p2_num_heads, p2_window_size, p2_qk_scale, p2_v_scale = p2

            res = random.choice([p1_res, p2_res])
            channels = random.choice([p1_channels, p2_channels])
            depths = random.choice([p1_depths, p2_depths])
            depths_from_p1 = depths == p1_depths

            if depths_from_p1:
                new_depths = p1_depths
                old_depths = p2_depths
            else:
                new_depths = p2_depths
                old_depths = p1_depths
            
            def merge_micro_structure(p1_depths, p2_depths, micro_structure_p1, micro_structure_p2, depths_from_p1, conv=False):
                new_ms_tmp = random.choice([micro_structure_p1, micro_structure_p2])

                ms_from_p1 = new_ms_tmp == micro_structure_p1
                
                if (ms_from_p1 and depths_from_p1) or (not ms_from_p1 and not depths_from_p1):
                    return new_ms_tmp
                
                ms = []
                for i, (num_p1_depth, num_p2_depth) in enumerate(zip(p1_depths, p2_depths)):
                    # if i == 0:
                    #     continue

                    new_depth = num_p1_depth if depths_from_p1 else num_p2_depth
                    ori_depths = p1_depths if ms_from_p1 else p2_depths
                    
                    if conv:
                        start = sum(ori_depths[:i])
                        end = sum(ori_depths[:i]) + ori_depths[i]

                        if i == self.conv_stage+1:
                            break
                    else:
                        if i < self.conv_stage+1:
                            continue

                        start = sum(ori_depths[self.conv_stage+1:i])
                        end = sum(ori_depths[self.conv_stage+1:i]) + ori_depths[i]
                    
                    ele = new_ms_tmp[start:end][0]
                    ms += [ele]*new_depth
                
                return ms
            
            conv_ratio = merge_micro_structure(p1_depths, p2_depths, p1_conv_ratio, p2_conv_ratio, depths_from_p1, True)
            kr_size = merge_micro_structure(p1_depths, p2_depths, p1_kr_size, p2_kr_size, depths_from_p1, True)
            mlp_ratio = merge_micro_structure(p1_depths, p2_depths, p1_mlp_ratio, p2_mlp_ratio, depths_from_p1, False)

            if not self.fixed_head_dim:
                num_heads = merge_micro_structure(p1_depths, p2_depths, p1_num_heads, p2_num_heads, depths_from_p1, False)
            else:
                num_heads = []

                for i, depth in enumerate(new_depths[self.conv_stage+1:]):
                    num_heads += (depth * [channels[i+self.conv_stage+1]//16])
            
            window_size = merge_micro_structure(p1_depths, p2_depths, p1_window_size, p2_window_size, depths_from_p1, False)
            qk_scale = merge_micro_structure(p1_depths, p2_depths, p1_qk_scale, p2_qk_scale, depths_from_p1, False)
            v_scale = merge_micro_structure(p1_depths, p2_depths, p1_v_scale, p2_v_scale, depths_from_p1, False)

            crossover_arch = (res, tuple(channels), tuple(depths), tuple(conv_ratio), tuple(kr_size), tuple(mlp_ratio), tuple(num_heads), tuple(window_size), tuple(qk_scale), tuple(v_scale))
                
            return crossover_arch

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        resume = self.load_checkpoint()

        if not resume:
            self.get_random(512, init_from_bank=True)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['acc'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['acc'])

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 val acc = {}, Top-1 test acc = {}, latency = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['acc'], self.vis_dict[cand]['test_acc'], self.vis_dict[cand]['latency']))
                tmp_accuracy.append(self.vis_dict[cand]['acc'])
            self.top_accuracies.append(tmp_accuracy)

            mutation = self.get_mutation(self.select_num, self.mutation_num, self.m_prob, self.s_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            k = self.select_num
            self.candidates = (mutation + crossover + self.keep_top_k[k])
            self.keep_top_k[k] = []

            for key in self.keep_top_k.keys():
                self.keep_top_k[key] = []

            print(len(self.candidates))
            self.epoch += 1
            self.save_checkpoint()


def setup_print(is_master):
        
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def main():
    args = get_config('configs/final_3min_space.yaml')

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        args.distributed = False

    args.device = 'cuda:0'

    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        print(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', )

        print(f'searching on world_size {dist.get_world_size()}, rank {dist.get_rank()}, local_rank {args.local_rank}')

        # dist.barrier()
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        rank = args.local_rank
        dist.barrier()
    else:
        raise NotImplementedError

    current_id = 2012

    args.print_process = [0]

    setup_print(is_master=args.local_rank in args.print_process)
    print(args)

    # ------------- model --------------

    dim_mutation_prob = getattr(args, 'searching_mutation_prob', 0.45)

    lib_data_dir = getattr(args, 'lib_dir', '/mnt/data/')
    offline_models_dir = f"{lib_data_dir}/{getattr(args, 'offline_models_dir', 'EdgeDL/offline_models/ours/')}"
    min_models_path = f"{offline_models_dir}/min.pkl"

    parser = argparse.ArgumentParser(
        'elasticvit searching script', parents=[get_args_parser()])
    extra_cfgs = parser.parse_args()

    assert os.path.exists(min_models_path)
    with open(min_models_path, "rb") as f:
        min_models = pickle.load(f)

    target_models_path = f"{offline_models_dir}/flops_{extra_cfgs.flops_limits}.pkl" #load all possible target models from the flops look up table
    assert os.path.exists(target_models_path)
    with open(target_models_path, "rb") as f:
        target_models = pickle.load(f)

    min_model = select_min_arch(extra_cfgs.flops_limits, min_models)
    parse_supernet_configs(args, min_model) # search the flops >= min arch

    print("-"*30)
    print(min_model)
    print(args.search_space)
    print("-"*30)

    model = build_supernet(args)
    model, _, _ = load_checkpoint(model, args.checkpoint, strict=True)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    model.cuda()

    # ------------- data --------------
    train_loader, val_loader, test_loader, _ = load_data_dist(args.dataloader, searching_set=True)

    mixup_fn = None
    mixup_active = extra_cfgs.mixup > 0 or extra_cfgs.cutmix > 0. or extra_cfgs.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=extra_cfgs.mixup, cutmix_alpha=extra_cfgs.cutmix, cutmix_minmax=extra_cfgs.cutmix_minmax,
            prob=extra_cfgs.mixup_prob, switch_prob=extra_cfgs.mixup_switch_prob, mode=extra_cfgs.mixup_mode,
            label_smoothing=extra_cfgs.smoothing, num_classes=1000)
    
    seed = 0 + current_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.flops_limits = extra_cfgs.flops_limits

    print(args.flops_limits)
    print(len(train_loader.sampler))

    save_path = "/".join(args.checkpoint.split("/")[:-1]) + f"/search/result_{args.flops_limits}M"
    
    print(save_path)
    searcher = EvolutionSearcher(model=model, model_without_ddp=model.module, train_loader=train_loader, val_loader=val_loader, test_loader=val_loader, 
                                output_dir=save_path, cfg=args, mixup_fn=mixup_fn, dim_mutation_prob=dim_mutation_prob, target_models=target_models)
    searcher.search()
    return

if __name__ == "__main__":
    main()