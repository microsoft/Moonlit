# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import copy
import math
import random
import sys
import pickle
import numpy as np
import torch
import os
import torch.distributed as dist
import torch.nn as nn
from timm.utils import reduce_tensor

try:
    from torch.distributed import get_world_size
except:
    pass

from .cnn import SuperCNNLayer, conv_stem
from .common_ops import BNSuper1d, BNSuper2d, LNSuper
from .transformer import (ELinear, SuperAttention, SuperFFN,
                          SuperTransformerBlock)
from .sample_utils import mutate_dims, softmax
# from nn_meter.predictor.transformer_predictor import BlockLatencyPredictor
sys.setrecursionlimit(10000)


class FusedSuperNet(nn.Module):
    def __init__(self, input_size=(3), super_stem_channels=128, num_classes=1000,
                 stage=['C', 'C', 'C', 'T', 'T', 'T'], depth=[3, 3, 4, 6, 6, 6], conv_ratio=[6, 4], mlp_ratio=[2, 1], num_heads=[4, 6, 8], min_depth=[2, 2, 2, 2, 2, 3],
                 windows_size=[1, 7, 14], channels=[32, 48, 88, 128, 216, 352], min_channels=[24, 40, 80, 112, 192, 320], qk_scale=[[2.5, 2, 1.5, 1], [2, 1], [1]], v_scale=[[1, 0.6], [1, 0.6], [1]],
                 head='mbv3', classifer_head_dim=[960, 1280], head_dropout_prob=0.2, pre_norm=False, norm_layer='BN', res_range=[160, 192, 224, 256], downsampling=[True, True, True, False, True, True], act=nn.Hardswish, se=[False, True, True, False, False, False,],
                 classifier_mul=4, talking_head=[False, False, False, False, False, False, ], dw_downsampling=False, limit_flops=800, min_flops=50, sample_prob=1, head_dims=8, use_res_specific_RPN=False,
                 bank_step=100, bank_size=150, bank_sampling_rate=0.5, bank_sampling_method='weighted_prob', flops_bound=15, flops_sampling_method='random', model_sampling_method='preference',
                 use_small_bank=False, small_limit_flops=0, small_min_flops=0, small_step_size=0,
                 hard_distillation_head=False, lib_dir='./', big_bank_choice=[], big_bank_size=50, max_importance_comparision_num=5, use_latency=False, small_bank_choice=[], latency_bound=1.5
                 ):
        super().__init__()
        assert head in ['mbv3', 'normal']
        assert bank_sampling_method in ['weighted_prob', 'best', 'uniform']
        assert flops_sampling_method in [
            'random', 'adjacent', 'cyclic', 'adjacent_step']
        assert model_sampling_method in ['preference', 'uniform']

        self.flops_sampling_method = flops_sampling_method
        self.model_sampling_method = model_sampling_method

        self.input_size = input_size
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.stem_channels = super_stem_channels // 8
        self.channels = channels
        self.min_channels = min_channels
        self.qk_scale = qk_scale
        self.v_scale = v_scale
        self.mlp_ratio = mlp_ratio
        self.conv_ratio = conv_ratio
        self.history_ids = {}
        self.kr_size = [3, 5]
        super_stem_channels = [16, 24]
        super_stem_depths = [1, 2]
        self.super_stem_channels = super_stem_channels

        # self.conv_stem = conv_stem(super_stem_channels, act=act)

        self.first_conv = nn.Conv2d(in_channels=3, out_channels=max(
            self.super_stem_channels), stride=2, kernel_size=3, padding=1, bias=False)
        self.first_conv_bn = BNSuper2d(max(self.super_stem_channels))
        self.first_conv_act = act()
        self.sampled_first_conv_channels = -1

        self.min_arch = None
        self.arch_sampling_prob = sample_prob
        self.max_importance_comparision_num = max_importance_comparision_num

        self.head_dims = head_dims
        self.flops_bound = flops_bound
        self.use_latency = use_latency
        self.latency_bound = latency_bound
        self.small_bank_choice = small_bank_choice
        if not self.use_latency:
            self.bank_flops_ranges = [c for c in range(
                min_flops, limit_flops+1, bank_step)]
            if use_small_bank:
                small_bank = [c for c in range(
                    small_min_flops, small_limit_flops+1, small_step_size)]
                self.bank_flops_ranges = small_bank + self.bank_flops_ranges

            if len(big_bank_choice) > 0:
                self.bank_flops_ranges = self.bank_flops_ranges + big_bank_choice
        else:
            self.bank_flops_ranges = [c for c in self.small_bank_choice]
            if len(big_bank_choice) > 0:
                self.bank_flops_ranges = self.bank_flops_ranges+big_bank_choice
            self.flops_bound = self.latency_bound

        if lib_dir is not None:
            self.offline_archs = self.load_offline_models(lib_dir=lib_dir)

        self.banks_size = bank_size
        self.big_bank_size = big_bank_size
        self.banks_step = bank_step
        self.banks_prob = bank_sampling_rate
        self.ce_loss = nn.CrossEntropyLoss()
        self.bank_sampling = False
        self.bank_sampling_method = bank_sampling_method
        # self.predictor=BlockLatencyPredictor("pixel6_lut", mode="blockwise")

        self.banks = [[] for bank_flops in self.bank_flops_ranges]

        self.anchor_arch = None

        # anchor model
        self.current_bank_id = 0

        res = []
        for r in res_range:
            res.append(r//2)

        self.conv_stem = SuperCNNLayer(depth=2, min_depth=1, min_channels=min(super_stem_channels), out_channels=max(super_stem_channels),
                                       in_channels=max(super_stem_channels), ratio=[1], act=act, downsampling=False, res=res, se=False)

        self.stage = stage
        self.head_type = head
        self.classifer_head_dim = classifer_head_dim
        self.res_range = res_range

        self.hard_distillation_head = hard_distillation_head

        self.head_dropout_prob = head_dropout_prob

        self.input_res = -1

        feature_exactor = []
        in_channels = max(super_stem_channels)
        self.in_channels = in_channels
        self.last_channels = -1

        self.classifier_mul = classifier_mul

        channel_scaling_factor = 1.
        depths_offset = 0

        self.windows_size = []
        num_C = self.stage.count('C')
        self.num_conv_stage = num_C

        DISABLE_SWIN = 1

        for idx, s in enumerate(stage):
            out_channels = channels[idx]

            if s == 'C':
                if downsampling[idx]:
                    for i in range(len(res)):
                        res[i] //= 2

                feature_exactor.append(SuperCNNLayer(
                    depth=depth[idx]+depths_offset, min_depth=min_depth[idx]+depths_offset, min_channels=self.min_channels[idx], in_channels=in_channels,
                    out_channels=channels[idx], ratio=conv_ratio, act=act, downsampling=downsampling[idx], res=res, se=se[idx]))  # type: ignore
            else:
                if downsampling[idx]:
                    for i in range(len(res)):
                        res[i] = math.ceil(res[i]/2)

                _windows_size = [[DISABLE_SWIN] for _ in range(len(res))] #

                self.windows_size.append(_windows_size)

                feature_exactor.append(SuperTransformerBlock(
                    depth=depth[idx]+depths_offset, min_depth=min_depth[idx]+depths_offset, in_channels=in_channels, res=res, min_channels=self.min_channels[idx],
                    max_channels=out_channels, num_heads=num_heads[idx -
                                                                   num_C], windows_size=_windows_size, act=act,
                    mlp_ratio=mlp_ratio, pre_norm=pre_norm, qk_scale=qk_scale[idx-num_C], v_scale=v_scale[idx -
                                                                                                          num_C], downsampling=downsampling[idx], use_res_specific_RPN=use_res_specific_RPN,
                    norm_layer=BNSuper1d if norm_layer == 'BN' else LNSuper, talking_head=talking_head[idx], dw_downsampling=dw_downsampling, head_dims=head_dims))

            in_channels = out_channels
        self.feature_exactor = nn.ModuleList(feature_exactor)
        
        print(self.windows_size, self.stage)

        self.pre_head_0 = ELinear(
            channels[-1], channels[-1] * classifier_mul)
        self.pre_head_norm_0 = BNSuper1d(channels[-1] * classifier_mul)
        self.pre_head_act = act()
        self.pre_head_1 = ELinear(
            channels[-1] * classifier_mul, classifer_head_dim[1])

        self.classifer = nn.Linear(classifer_head_dim[1], num_classes)

    def load_offline_models(self, lib_dir):
        offline_archs = {}
        if not self.use_latency:
            for pf in self.bank_flops_ranges:
                path = f"{lib_dir}/flops_{pf}.pkl"
                with open(path, "rb") as f:
                    archs = pickle.load(f)
                print(f"load offline archs for {pf}M bank, num: {len(archs)}")
                offline_archs[pf] = archs
        else:
            for pf in self.bank_flops_ranges:
                path = f"{lib_dir}/latency_{pf}.pkl"
                with open(path, "rb") as f:
                    archs = pickle.load(f)
                print(f"load offline archs for {pf}ms bank, num: {len(archs)}")
                offline_archs[pf] = archs

        min_filename = f"{lib_dir}/min.pkl"
        with open(min_filename, "rb") as f:
            archs = pickle.load(f)
       #     print(archs)
        print(f"load min archs, num: {len(archs)}")
        offline_archs['min'] = archs

        return offline_archs

    def set_min_arch(self, arch):
        arch = tuple([tuple(a) if isinstance(a, list) else a for a in arch])
        self.min_arch = arch
        print('set_min_arch', self.min_arch)

    def _importance_comparison(self, arch, target_flops):
        if self.model_sampling_method == 'uniform':
            return -1

        try:
            anchor_arch_flops = 150 if min(
                self.bank_flops_ranges) >= 150 else 50
            bank_id = self.bank_flops_ranges.index(anchor_arch_flops)
        except:
            bank_id = 0
            # assert self.bank_flops_ranges[bank_id] >= 100

        if target_flops > self.bank_flops_ranges[bank_id]:
            anchor_arch, anchor_loss, anchor_id = self._get_subnet_from_bank(
                self.banks[bank_id])
        else:
            anchor_arch = self.arch_sampling(mode='min')

        _, anchor_channels, anchor_depths, _, _, _, _, _, _, _ = anchor_arch
        res, channels, depths, conv_ratios, kernel_sizes, mlp_ratio, num_heads, windows_size, qk_scale, v_scale = arch
        num_conv_stages = self.stage.count('C')

        tmp_channels = channels[:1 + num_conv_stages] + \
            anchor_channels[1 + num_conv_stages:]

        self.set_arch(res, tmp_channels, depths, conv_ratios, kernel_sizes,
                      mlp_ratio, num_heads, windows_size, qk_scale, v_scale)
        anchor_channels_flops = self.compute_flops(arch=(
            res, tmp_channels, depths, conv_ratios, kernel_sizes, mlp_ratio, num_heads, windows_size, qk_scale, v_scale))

        depths, conv_ratios, kernel_sizes, mlp_ratio, num_heads, windows_size, v_scale, qk_scale = list(depths), list(conv_ratios), \
            list(kernel_sizes), list(mlp_ratio), list(num_heads), list(
                windows_size), list(v_scale), list(qk_scale)

        def insert_ele(set_of_list, pos, eles):
            for list, ele in zip(set_of_list, eles):
                list.insert(pos, ele)

        def pop_ele(set_of_list, pos):
            for list in set_of_list:
                list.pop(pos)

        for stage_index, (anchor_d, d) in enumerate(zip(anchor_depths[1:], depths[1:])):
            reduced_layers = 0

            if stage_index < num_conv_stages:
                continue

            stage_index += 1

            if d < anchor_d:
                reduced_layers = anchor_d - d
                depths[stage_index] = anchor_d

                if stage_index >= num_conv_stages+1:
                    num_heads += [channels[stage_index] //
                                  self.head_dims] * depths[stage_index]
                    for _ in range(reduced_layers):
                        offset = sum(depths[num_conv_stages+1:stage_index])
                        insert_ele([mlp_ratio, windows_size, qk_scale, v_scale], pos=offset, eles=[
                                   mlp_ratio[offset], windows_size[offset], qk_scale[offset], v_scale[offset]])
                else:
                    for _ in range(reduced_layers):
                        offset = sum(depths[:stage_index])
                        insert_ele([conv_ratios, kernel_sizes], pos=offset, eles=[
                                   conv_ratios[offset], kernel_sizes[offset]])

            elif anchor_d < d:
                added_layers = d - anchor_d
                depths[stage_index] = anchor_d

                if stage_index >= num_conv_stages+1:
                    num_heads += [channels[stage_index] //
                                  self.head_dims] * depths[stage_index]
                    offset = sum(depths[num_conv_stages+1:stage_index])

                    for _ in range(added_layers):
                        pop_ele([mlp_ratio, windows_size,
                                qk_scale, v_scale], offset)
                else:
                    offset = sum(depths[:stage_index])
                    pop_ele([conv_ratios, kernel_sizes], offset)

        self.set_arch(res, channels, depths, conv_ratios, kernel_sizes,
                      mlp_ratio, num_heads, windows_size, qk_scale, v_scale)
        anchor_depths_flops = self.compute_flops(arch=(
            res, channels, depths, conv_ratios, kernel_sizes, mlp_ratio, num_heads, windows_size, qk_scale, v_scale))

        return anchor_channels_flops - anchor_depths_flops

    def bank_loss_update(self, mini_batch, world_size):
        images, labels = mini_batch
        world_size = get_world_size()

        for bank_id, bank in enumerate(self.banks):
            for arch_item in bank:
                arch, old_loss = arch_item

                self.set_arch(*arch)
                current_arch_flops = self.compute_flops(arch=arch)

                if current_arch_flops - self.bank_flops_ranges[bank_id] > 2*self.flops_bound:
                    print(
                        f'remove one subnet with {current_arch_flops}M FLOPs from {self.bank_flops_ranges[bank_id]}M FLOPs bank')
                    bank.remove(arch_item)
                    continue

                with torch.no_grad():
                    outputs = self.forward(images)
                    new_loss = -self.ce_loss(outputs, labels)

                dist.all_reduce(new_loss, op=dist.ReduceOp.SUM)
                new_loss /= world_size
                new_loss = new_loss.item()

                print(
                    f'arch old loss {old_loss}, new loss {new_loss}, batch size {images.size(0)}')
                arch_item[1] = new_loss

    def bank_arch_update(self, arch, mini_batch, world_size, update_bank=False):

        if not self.bank_sampling:
            bank_id = self.current_bank_id

            self.set_arch(*arch)
            flops = self.compute_flops(arch=arch)
            bank_flops = self.bank_flops_ranges[bank_id]
            if flops - bank_flops > self.flops_bound*2:
               # print(f'arch flops {flops}M out of {bank_flops}M bank')
                return

            world_size = get_world_size()

            images, labels = mini_batch
            with torch.no_grad():
                outputs = self.forward(images)
                if isinstance(outputs, (list, tuple)):
                    outputs = (outputs[0] + outputs[1]) / 2
                new_loss = -self.ce_loss(outputs, labels)
            new_loss = new_loss.clone().detach()
            dist.all_reduce(new_loss, op=dist.ReduceOp.SUM)
            new_loss /= world_size
            new_loss = new_loss.item()

            if update_bank:
                new_bank = []

                for arch_item in self.banks[bank_id]:
                    p_arch, _ = arch_item
                    self.set_arch(*p_arch)

                    with torch.no_grad():
                        outputs = self.forward(images)
                        if isinstance(outputs, (list, tuple)):
                            outputs = (outputs[0] + outputs[1]) / 2
                        p_loss = -self.ce_loss(outputs, labels)

                    dist.barrier()
                    dist.all_reduce(p_loss, op=dist.ReduceOp.SUM)

                    p_loss /= world_size

                    new_bank.append([p_arch, p_loss.item()])

                self.banks[bank_id] = new_bank

            current_bank = self.banks[bank_id]
            new_arch_item = [arch, new_loss]

            banks_size = self.banks_size if bank_flops < 500 else self.big_bank_size
            if len(current_bank) < banks_size:
                current_bank.append(new_arch_item)
            else:
                _, subnet_loss, subnet_id = self._get_subnet_from_bank(
                    current_bank, scheme='worst')

                if subnet_loss < new_loss:
                    current_bank[subnet_id] = new_arch_item
        # print('current banks',self.banks.keys())

    def _get_bank_info(self, bank):
        arch_list = [arch_item[0] for arch_item in bank]
        loss_list = [arch_item[1] for arch_item in bank]

        return arch_list, loss_list

    def _get_subnet_from_bank(self, bank, scheme='best'):
        arch_list, loss_list = self._get_bank_info(bank)
        # print('subnet bankid',bank,loss_list)

        selected_subnet_loss = max(
            loss_list) if scheme == 'best' else min(loss_list)
        selected_subnet_id = loss_list.index(selected_subnet_loss)
        selected_subnet_arch = arch_list[selected_subnet_id]

        return selected_subnet_arch, selected_subnet_loss, selected_subnet_id

    def _arch_bank_sampling(self, bank_id, update_loss=False):
        bank = self.banks[bank_id]

        arch_list, loss_list = self._get_bank_info(bank)

        if self.bank_sampling_method == 'weighted_prob':
            prob = softmax(loss_list, to_list=True)
            arch = arch_list[np.random.choice(
                [aid for aid in range(len(arch_list))], size=1, p=prob)[0]]
        
        elif self.bank_sampling_method == 'uniform':  # random sampling
            # need update the loss of this subnet?
            arch = random.choice(arch_list)

        else:  # select the best subnet
            arch = self._get_subnet_from_bank(bank)

        if update_loss:
            pass

        return arch

    def set_arch(self, res, channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale):
        # inp = 128//8
        num_conv_stages = self.stage.count('C')
        num_transformer_stages = len(self.stage) - self.stage.count('C')
        res_idx = self.res_range.index(res)
        self.input_res = res
        self.last_channels = channels[-1]

        stem_channels = channels[0]
        stem_depth = depths[0]
        stem_conv_ratio = conv_ratio[:stem_depth]
        stem_kr_size = kr_size[:stem_depth]

        channels = channels[1:]
        depths = depths[1:]
        conv_ratio = conv_ratio[stem_depth:]
        kr_size = kr_size[stem_depth:]

        self.sampled_first_conv_channels = stem_channels
        self.first_conv_bn.set_conf(self.sampled_first_conv_channels)
        # sampled_stem_depths, sampled_stem_channels, _, sampled_stem_kernel_sizes = self.conv_stem.arch_sampling(mode='min', sampled_res_idx=0, in_channels=self.sampled_first_conv_channels)

        # chen update 22/10/10
        self.conv_stem.set_conf([0 for _ in range(
            stem_depth)], sampled_layers=stem_depth, sampled_channels=stem_channels)

        for current_layer_idx, layer in enumerate(self.conv_stem.layers):
            if current_layer_idx >= stem_depth:
                break

            layer[0].set_conf(channels=stem_channels,
                              sampled_ratio=stem_conv_ratio[current_layer_idx],
                              sampled_kernel_size=stem_kr_size[current_layer_idx],
                              sampled_out_channels=stem_channels,
                              last_stage_sampled_channels=stem_channels,
                              sampled_res=0
                              )

        inp = self.sampled_first_conv_channels

        for idx, module in enumerate(self.feature_exactor):  # type: ignore
            if idx <= num_conv_stages - 1:
                module: SuperCNNLayer
                module.set_conf([0 for _ in range(
                    depths[idx])], sampled_layers=depths[idx], sampled_channels=channels[idx])  # type: ignore
                sampled_res = module.res[res_idx]
                module.sampled_res = sampled_res

                for current_layer_idx, layer in enumerate(module.layers):
                    if current_layer_idx >= depths[idx]:
                        break

                    conv_offset = 0
                    for j in range(idx):
                        conv_offset += depths[j]

                    layer[0].set_conf(channels=channels[idx],
                                      sampled_ratio=conv_ratio[conv_offset +
                                                               current_layer_idx],
                                      sampled_kernel_size=kr_size[conv_offset +
                                                                  current_layer_idx],
                                      sampled_out_channels=channels[idx],
                                      last_stage_sampled_channels=inp,
                                      sampled_res=sampled_res
                                      )
            else:
                module: SuperTransformerBlock
                module.set_conf(
                    sampled_layers=depths[idx], sampled_channels=channels[idx])
                sampled_res = module.res[res_idx]
                module.sampled_res = sampled_res  # feature map size

                for current_layer_idx, layer in enumerate(module.layers):
                    if current_layer_idx >= depths[idx]*2:
                        break

                    transformer_offset = 0
                    for j in range(idx-num_conv_stages):
                        transformer_offset += depths[num_conv_stages+j]

                    transformer_offset += current_layer_idx//2

                    if isinstance(layer, SuperAttention):
                        sampled_num_head = channels[idx]//self.head_dims
                        sampled_window_size = window_size[transformer_offset]
                        sampled_qk_scale = qk_scale[transformer_offset]
                        layer.sampled_res = sampled_res
                        sampled_v_scale = v_scale[transformer_offset]

                        if layer.layer_idx == 0:
                            if layer.conv_downsampling and not layer.use_dw_downsampling:
                                layer.reduction_norm.set_conf(
                                    activated_channels=channels[idx])

                        sampled_qkdim = int(
                            channels[idx]//sampled_num_head * sampled_v_scale)
                        sampled_vdim = int(
                            channels[idx]//sampled_num_head//sampled_qk_scale)
                        last_stage_sampled_channels = inp if current_layer_idx == 0 else -1
                        layer.set_conf(sampled_windows_size=sampled_window_size, sampled_num_heads=sampled_num_head,
                                       sampled_qk_scale=sampled_qk_scale, sampled_v_scale=sampled_v_scale, sampled_channels=channels[
                                           idx], last_stage_sampled_channels=last_stage_sampled_channels, sampled_res=sampled_res
                                       )
                    elif isinstance(layer, SuperFFN):
                        layer.set_conf(
                            activated_dim=channels[idx], ratio=mlp_ratio[transformer_offset])

            inp = channels[idx]

    def compute_flops(self, details=False, arch=None):
        if self.use_latency:
          #  print('arch',arch)
            return self.predictor.get_latency(arch)
        stem_flops = 0
        stem_flops += (
            3 * self.sampled_first_conv_channels * self.input_res//2 * self.input_res//2 * 9 +
            (self.sampled_first_conv_channels * self.input_res//2 * self.input_res//2 * self.conv_stem.layers[0][0].sampled_kernel_size**2 +
             self.sampled_first_conv_channels * self.sampled_first_conv_channels * self.input_res//2 * self.input_res//2) * self.conv_stem.sampled_layers
        )

        feature_exactor_flops = 0

        transformer_flops = 0
        num_conv_stages = self.stage.count('C')

        for idx, module in enumerate(self.feature_exactor):
            flops = module.compute_flops()

            feature_exactor_flops += flops

            if idx >= num_conv_stages:
                transformer_flops += flops

            if details:
                print(f"stage {idx} flops {flops/1e6}")

        classifer_flops = 0

        if self.head_type == 'mbv3':
            classifer_flops += (
                # type: ignore
                self.last_channels * self.last_channels * self.classifier_mul * (self.feature_exactor[-1].sampled_res*self.feature_exactor[-1].sampled_res) +
                self.last_channels * self.classifier_mul * self.classifer_head_dim[1] +
                self.classifer_head_dim[1] * self.num_classes
            )
        else:
            classifer_flops += (
                self.last_channels * self.num_classes
            )

        total = stem_flops + feature_exactor_flops + classifer_flops

        if details:
            print(
                f'conv flops {(total-transformer_flops)/1e6} (M), transformer flops {transformer_flops/1e6} (M), total flops {total/1e6} (M)')

        return total/1e6

    def set_regularization_mode(self, head_drop_prob=0.2, module_drop_prob=0.2, last_two_stage=True):
        self.head_dropout_prob = head_drop_prob
        num_conv_stages = self.stage.count('C')

        # type: ignore
        depths = sum(
            [module.sampled_layers for module in self.feature_exactor])
        current_depth = 0

        for idx, module in enumerate(self.feature_exactor):  # type: ignore
            if last_two_stage and idx < len(self.stage) - 2:
                # type: ignore
                current_depth += self.feature_exactor[idx].sampled_layers
                continue

            if idx <= num_conv_stages - 1:
                module: SuperCNNLayer

                for current_layer_idx, layer in enumerate(module.layers):
                    if current_layer_idx >= module.sampled_layers:
                        break

                    current_depth += 1

                    layer[0].drop_connect_prob = module_drop_prob * \
                        float(current_depth) / depths
            else:
                module: SuperTransformerBlock

                for current_layer_idx, layer in enumerate(module.layers):
                    if current_layer_idx >= module.sampled_layers*2:
                        break

                    if isinstance(layer, SuperAttention):
                        current_depth += 1

                    layer.drop_connect_prob = module_drop_prob * \
                        float(current_depth) / depths

    # convert the layer-wise sampling arch config to block wise
    def parse_arch(self, arch):
        num_conv_stages = self.stage.count('C')
        arch = [list(a) if isinstance(a, tuple) else a for a in arch]

        res, channels, depths, conv_ratios, kernel_sizes, mlp_ratio, num_heads, windows_size, qk_scale, v_scale = arch

        parse_conv_ratio, parse_kernel_size, parse_mlp_ratio, parse_qk_scale, parse_v_scale = [], [], [], [], []
        for stage_index, d in enumerate(depths[1:]):
            stage_index += 1

            if stage_index >= num_conv_stages+1:
                offset = sum(depths[num_conv_stages+1:stage_index])

                parse_mlp_ratio.append(mlp_ratio[offset])
                parse_qk_scale.append(qk_scale[offset])
                parse_v_scale.append(v_scale[offset])

            else:
                offset = sum(depths[:stage_index])

                parse_conv_ratio.append(conv_ratios[offset])
                parse_kernel_size.append(kernel_sizes[offset])

        return parse_conv_ratio, parse_kernel_size, parse_mlp_ratio, parse_qk_scale, parse_v_scale

    def arch_slicing(self, arch, min_arch, all_dim=False):
        num_conv_stages = self.stage.count('C')
        arch = [list(a) if isinstance(a, tuple) else a for a in arch]
        min_res, min_channels, min_depths, min_conv_ratios, min_kernel_sizes, min_mlp_ratio, _, _, min_qk_scale, min_v_scale = min_arch
        res, channels, depths, conv_ratios, kernel_sizes, mlp_ratio, num_heads, windows_size, qk_scale, v_scale = arch
        if res < min_res:
            res = min_res

        new_res_idx = self.res_range.index(res)

        for stage_index, (min_c, c) in enumerate(zip(min_channels[1:], channels[1:])):
            if c < min_c:
                channels[stage_index+1] = min_c

        num_heads, windows_size = [], []
        for stage_index, (min_d, d) in enumerate(zip(min_depths[1:], depths[1:])):
            reduced_layers = 0
            stage_index += 1

            if d < min_d:
                reduced_layers = min_d - d
                depths[stage_index] = min_d

            if stage_index >= num_conv_stages+1:
                offset = sum(depths[num_conv_stages+1:stage_index])
                min_offset = sum(min_depths[num_conv_stages+1:stage_index])

                windows_size += [self.feature_exactor[stage_index -
                                                      1].windows_size[new_res_idx][0]] * depths[stage_index]
                num_heads += [channels[stage_index] //
                              self.head_dims] * depths[stage_index]

                for _ in range(reduced_layers):
                    mlp_ratio.insert(offset, mlp_ratio[offset])
                    qk_scale.insert(offset, qk_scale[offset])
                    v_scale.insert(offset, v_scale[offset])

                if all_dim:
                    current_mlp_ratio_val, min_mlp_ratio_val = mlp_ratio[
                        offset], min_mlp_ratio[min_offset]
                    current_qk_scale_val, min_qk_scale_val = qk_scale[offset], min_qk_scale[min_offset]
                    current_v_scale_val, min_v_scale_val = v_scale[offset], min_v_scale[min_offset]
                    for idx in range(depths[stage_index]):
                        mlp_ratio[offset +
                                  idx] = max(current_mlp_ratio_val, min_mlp_ratio_val)
                        qk_scale[offset +
                                 idx] = max(current_qk_scale_val, min_qk_scale_val)
                        v_scale[offset +
                                idx] = max(current_v_scale_val, min_v_scale_val)
            else:
                offset = sum(depths[:stage_index])
                min_offset = sum(min_depths[:stage_index])

                for _ in range(reduced_layers):
                    conv_ratios.insert(offset, conv_ratios[offset])
                    kernel_sizes.insert(offset, kernel_sizes[offset])
                if all_dim:
                    current_conv_ratio_val, min_conv_ratio_val = conv_ratios[
                        offset], min_conv_ratios[min_offset]
                    current_kernel_size_val, min_kernel_size_val = kernel_sizes[
                        offset], min_kernel_sizes[min_offset]
                    for idx in range(depths[stage_index]):
                        conv_ratios[offset +
                                    idx] = max(current_conv_ratio_val, min_conv_ratio_val)
                        kernel_sizes[offset +
                                     idx] = max(current_kernel_size_val, min_kernel_size_val)

        return res, tuple(channels), tuple(depths), tuple(conv_ratios), tuple(kernel_sizes), tuple(mlp_ratio), tuple(num_heads), tuple(windows_size), tuple(qk_scale), tuple(v_scale)

    def select_min_arch(self, flops):
        min_archs = self.offline_archs['min']
        limits = max(min_archs.keys())

        if flops > limits:
            return min_archs[limits]

        lower = 0
        arch = []
        for upper in min_archs.keys():
            if lower + 2*self.flops_bound < flops <= upper + 2*self.flops_bound:
                arch = min_archs[upper]
                break
            else:
                lower = upper

        return arch

    def sample_random_subnet_from_range(self, min_flops, max_flops):
        while True:
            if min_flops in self.offline_archs:
                arch = random.choice(self.offline_archs[min_flops])
            else:
                min_arch = self.select_min_arch(flops=min_flops)
                # self.set_arch_slicing(min_arch, True)
                arch = self.arch_sampling_engine(
                    mode='uniform', min_arch=min_arch)

            self.set_arch(*arch)
            sliced_subnet_flops = self.compute_flops(arch=arch)
            if min_flops <= sliced_subnet_flops <= max_flops:
                return arch, sliced_subnet_flops

    def arch_sampling(self, mode='uniform', force_random=False, random_subnet_idx=-1, flops_list=[], current_bank_id=0, use_bank=True):
        if mode == 'min' and len(flops_list) > 0:
            if self.min_arch is not None:
                self.set_arch(*self.min_arch)
                return self.min_arch
            if self.flops_sampling_method == 'adjacent_step':
                target_flops = self.bank_flops_ranges[current_bank_id]
                arch = self.select_min_arch(target_flops)
                # print('set min',target_flops,arch)
                if len(arch) != 0:
                    self.set_arch(*arch)
                    return arch
            elif force_random:
                arch = self.select_min_arch(flops=100)

                if len(arch) != 0:
                    self.set_arch(*arch)
                    return arch
            else:
                avg_flops = int(sum(flops_list)/len(flops_list))
                arch = self.select_min_arch(avg_flops)

                if len(arch) != 0:
                    self.set_arch(*arch)
                    return arch

        if mode == 'uniform' and not force_random:
            bank_nums = len(self.bank_flops_ranges)

            if self.flops_sampling_method == 'adjacent':
                last_bank_id = self.current_bank_id

                if last_bank_id == 0:
                    next_step = [0, 1]
                elif last_bank_id == bank_nums - 1:
                    next_step = [-1, 0]
                else:
                    next_step = [-1, 0, 1]

                bank_id = random.choice(next_step) + last_bank_id
                target_flops = self.bank_flops_ranges[bank_id]
            elif self.flops_sampling_method == 'adjacent_step':
                bank_id = current_bank_id
                target_flops = self.bank_flops_ranges[bank_id]

                if not bank_id in self.history_ids:
                    self.history_ids[bank_id] = 0
                self.history_ids[bank_id] += 1

            elif self.flops_sampling_method == 'cyclic':
                if random_subnet_idx == 0:
                    last_bank_id = self.current_bank_id

                    if last_bank_id == 0:
                        bank_id = len(self.bank_flops_ranges) - 1
                    else:
                        bank_id = last_bank_id - 1
                else:
                    bank_id = self.current_bank_id

                target_flops = self.bank_flops_ranges[bank_id]
            elif self.flops_sampling_method == 'random':
                target_flops = random.choice(self.bank_flops_ranges)
                bank_id = self.bank_flops_ranges.index(target_flops)
            else:
                raise NotImplementedError

            self.current_bank_id = bank_id
            exploitation_prob = random.random()
            banks_size = self.banks_size if target_flops < 500 else self.big_bank_size
            
            if use_bank == False:
                self.banks_prob = 0

            if exploitation_prob < self.banks_prob and len(self.banks[bank_id]) > banks_size//2:
                self.bank_sampling = True
                arch = self._arch_bank_sampling(bank_id=bank_id)
            else:
                self.bank_sampling = False

                arch_set, diffs = [], []
                while len(arch_set) < self.max_importance_comparision_num:
                    arch, _ = self.sample_random_subnet_from_range(
                        target_flops, target_flops + 2 * self.flops_bound)
                    
                    arch_set.append(arch)

                    if len(self.banks[bank_id]) < banks_size//2:
                        diffs.append(-1)
                        break

                    diff = self._importance_comparison(
                        copy.deepcopy(arch), target_flops)
                    diffs.append(diff)

                arch = arch_set[diffs.index(min(diffs))]

            self.set_arch(*arch)
            return arch

        return self.arch_sampling_engine(mode=mode)

    def arch_sampling_engine(self, mode, min_arch=None, mumate_prob=1., res_mutate_prob=1.):
      #  print('arch sampling engine',mode)
        if mode == 'max':
            self.sampled_first_conv_channels = max(self.super_stem_channels)
        elif mode == 'min':
            self.sampled_first_conv_channels = min(self.super_stem_channels)
        else:
            self.sampled_first_conv_channels = mutate_dims(
                self.super_stem_channels, prob=mumate_prob)

        self.first_conv_bn.set_conf(self.sampled_first_conv_channels)
        depths, channels, conv_ratios, kernel_sizes, windows_size, num_heads, qk_scale, v_scale, mlp_ratio = [
        ], [], [], [], [], [], [], [], []

        res_range = self.res_range
        current_min_channels, current_min_depths, current_min_conv_ratio, current_min_kernel_size, current_min_mlp_ratio, current_min_qk_scale, current_min_v_scale = 0, 0, 0, 0, 0, 0, 0
        if min_arch is not None:
            min_res, min_channels, min_depths, min_conv_ratios, min_kernel_sizes, min_mlp_ratio, _, _, min_qk_scale, min_v_scale = min_arch
            res_range = self.res_range[self.res_range.index(min_res):]
            current_min_channels, current_min_depths = min_channels[0], min_depths[0]
            parse_conv_ratio, parse_kernel_size, parse_mlp_ratio, parse_qk_scale, parse_v_scale = self.parse_arch(
                arch=min_arch)

        if mode == 'max':
            res = max(self.res_range)
        elif mode == 'min':
            res = min(self.res_range)
        else:
            res = mutate_dims(res_range, prob=res_mutate_prob)

        res_idx = self.res_range.index(res)
        sampled_stem_depths, sampled_stem_channels, _, sampled_stem_kernel_sizes = self.conv_stem.arch_sampling(
            mode=mode, sampled_res_idx=0, in_channels=self.sampled_first_conv_channels, equal_channels=True, min_channels=current_min_channels, min_depths=current_min_depths, prob=mumate_prob)
        in_channels = sampled_stem_channels
        assert sampled_stem_channels == self.sampled_first_conv_channels

        depths.append(sampled_stem_depths)
        channels.append(sampled_stem_channels)
        kernel_sizes += sampled_stem_kernel_sizes
        conv_ratios += [1] * sampled_stem_depths

        for idx, module in enumerate(self.feature_exactor):
            if min_arch is not None:
                current_min_channels, current_min_depths = min_channels[idx +
                                                                        1], min_depths[idx+1]

            if isinstance(module, SuperCNNLayer):
                if min_arch is not None:
                    current_min_conv_ratio = parse_conv_ratio[idx]
                    current_min_kernel_size = parse_kernel_size[idx]

                sampled_depths, sampled_channels, sampled_conv_ratios, sampled_kernel_sizes = module.arch_sampling(mode=mode, sampled_res_idx=res_idx, in_channels=in_channels,
                                                                                                                   min_channels=current_min_channels, min_depths=current_min_depths, min_conv_ratio=current_min_conv_ratio, min_kr_size=current_min_kernel_size, prob=mumate_prob)

                conv_ratios += sampled_conv_ratios
                kernel_sizes += sampled_kernel_sizes
            elif isinstance(module, SuperTransformerBlock):
                if min_arch is not None:
                    current_min_mlp_ratio = parse_mlp_ratio[idx -
                                                            self.num_conv_stage]
                    current_min_qk_scale = parse_qk_scale[idx -
                                                          self.num_conv_stage]
                    current_min_v_scale = parse_v_scale[idx -
                                                        self.num_conv_stage]

                sampled_depths, sampled_channels, sampled_windows_size, sampled_num_heads, sampled_qk_scale, sampled_v_scale, sampled_mlp_ratio = module.arch_sampling(
                    mode=mode, sampled_res_idx=res_idx, in_channels=in_channels, min_channels=current_min_channels, min_depths=current_min_depths,
                    min_mlp_ratio=current_min_mlp_ratio, min_qk_scale=current_min_qk_scale, min_v_scale=current_min_v_scale, prob=mumate_prob
                )

                windows_size += sampled_windows_size
                num_heads += sampled_num_heads
                qk_scale += sampled_qk_scale
                v_scale += sampled_v_scale
                mlp_ratio += sampled_mlp_ratio
            else:
                raise NotImplementedError

            depths.append(sampled_depths)
            channels.append(sampled_channels)

            in_channels = sampled_channels

        self.last_channels = in_channels
        self.input_res = res

        arch = (res, channels, depths, conv_ratios, kernel_sizes,
                mlp_ratio, num_heads, windows_size, qk_scale, v_scale)
       # print(mode, arch)
        self.set_arch(*arch)
        return res, tuple(channels), tuple(depths), tuple(conv_ratios), tuple(kernel_sizes), tuple(mlp_ratio), tuple(num_heads), tuple(windows_size), tuple(qk_scale), tuple(v_scale)

    def _set_mbv3_head(self):
        self.pre_head_0.set_conf(
            self.last_channels, self.last_channels * self.classifier_mul)
        self.pre_head_norm_0.set_conf(self.last_channels * self.classifier_mul)
        self.pre_head_1.set_conf(
            self.last_channels * self.classifier_mul, self.classifer_head_dim[1])

    def forward(self, x):
        if x.size(-1) != self.input_res:
            x = torch.nn.functional.interpolate(
                x, size=self.input_res, mode='bicubic')

        first_conv_weights = self.first_conv.weight[:
                                                    self.sampled_first_conv_channels, :3]
        x = self.first_conv._conv_forward(x, first_conv_weights, bias=None)
        x = self.first_conv_act(self.first_conv_bn(x))
        x = self.conv_stem(x)

        for idx, module in enumerate(self.feature_exactor):
            if idx > 1:
                if self.stage[idx-1] == 'C' and self.stage[idx] == 'T':
                    x = x.flatten(2).transpose(1, 2)

            x = module(x)

        self._set_mbv3_head()

        x = self.pre_head_act(self.pre_head_norm_0(
            self.pre_head_0(x))).mean(dim=1).squeeze(1)
        x = self.pre_head_act(self.pre_head_1(x))
        if self.head_dropout_prob > 0 and self.training:
            x = torch.nn.functional.dropout(x, p=self.head_dropout_prob)

        return self.classifer(x)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'attention_bias'}
