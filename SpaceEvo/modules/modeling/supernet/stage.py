from __future__ import annotations
import copy
from typing import Iterator, List, OrderedDict, Tuple, Callable

import numpy as np
import torch
from torch import nn, Tensor

from modules.search_space.macro_arch import StageConfig
from modules.modeling.dynamic_blocks import BaseDynamicBlock
from modules.modeling.blocks.base_block import BaseBlockConfig


class Stage(nn.Module):

    def __init__(self, max_cin: int, width_list: List[int], depth_list: List[int], stride: int, building_block: Callable[..., BaseDynamicBlock]) -> None:
        super().__init__()
        self.max_cin = max_cin
        self.width_list = width_list
        self.depth_list = depth_list
        self.stride = stride
        self.building_block = building_block

        blocks = []
        for i in range(self.max_depth):
            max_cin = self.max_cin if i == 0 else self.max_width
            stride = self.stride if i == 0 else 1
            blocks.append(self.building_block(max_cin=max_cin, width_list=self.width_list, stride=stride))
        self.blocks = nn.Sequential(*blocks)

        self.active_depth = self.max_depth

    def forward(self, x: Tensor) -> Tensor:
        for block in self.active_blocks:
            x = block(x)
        return x

    @property
    def max_depth(self) -> int:
        return max(self.depth_list)

    @property
    def min_depth(self) -> int:
        return min(self.depth_list)
        
    @property
    def max_width(self) -> int:
        return max(self.width_list)

    @property
    def active_blocks(self) -> Iterator[BaseDynamicBlock]:
        for block in self.blocks[:self.active_depth]:
            yield block 

    def sample_active_sub_stage(self, align=False, width_list=None):
        # sample depth
        self.active_depth = np.random.choice(self.depth_list)
        kwe_list = []
        for block in self.active_blocks:
            kwe_list.append(block.sample_active_block(width_list=width_list))
        if align:
            for i in range(1, len(kwe_list)):
                kwe_list[i] = kwe_list[0]
        return kwe_list
    
    def sample_min_sub_stage(self, align=False, width_list=None):
        # sample depth
        self.active_depth = self.min_depth
        kwe_list = []
        for block in self.active_blocks:
            kwe_list.append(block.sample_min_block(width_list=width_list))
        if align:
            for i in range(1, len(kwe_list)):
                kwe_list[i] = kwe_list[0]
        return kwe_list
    def set_active_sub_stage(self, sub_stage_config):
        if isinstance(sub_stage_config, Tuple):
            depth, kwe_list = sub_stage_config
        else:
            depth = len(sub_stage_config)
            kwe_list = sub_stage_config
        self.active_depth = depth
        for (k, w, e), block in zip(kwe_list, self.active_blocks):
            block.set_active_block(k, w, e)

    def set_max_sub_stage(self):
        self.active_depth = self.max_depth
        for block in self.active_blocks:
            block.set_max_block()

    def set_min_sub_stage(self):
        self.active_depth = self.min_depth
        for block in self.active_blocks:
            block.set_min_block()

    def get_active_sub_stage(self, cin: int) -> nn.Sequential:
        blocks = []
        for block in self.active_blocks:
            active_block = block.get_active_block(cin)
            blocks.append(active_block)
            cin = active_block.cout
        return nn.Sequential(*blocks)

    def get_active_tf_sub_stage(self, cin: int, name: str):
        from tensorflow import keras
        blocks = []
        for block in self.active_blocks:
            active_block = block.get_active_tf_block(cin)
            blocks.append(active_block)
            cin = active_block.cout
        return keras.Sequential(layers=blocks, name=name)

    def get_active_sub_stage_config(self, cin: int) -> List[BaseBlockConfig]:
        rv = []
        for block in self.active_blocks:
            active_block_config = block.get_active_block_config(cin)
            rv.append(active_block_config)
            cin = active_block_config.cout
        return rv

    @classmethod
    def build_from_config(cls, max_cin: int, config: StageConfig, building_block: Callable[..., BaseDynamicBlock], width_window_choice: int) -> Stage:
        return cls(max_cin=max_cin, width_list=config.get_active_width_list(width_window_choice), depth_list=config.depth_list, stride=config.stride, building_block=building_block)


    # === mutate and crossover during evolutional search ===
    
    @property
    def kernel_list(self):
        return self.blocks[0].kernel_list

    @property
    def expand_list(self):
        return self.blocks[0].expand_list

    def _pad_kwe_list(self, kwe_list):
        padded_kwe_list = copy.deepcopy(kwe_list)
        for _ in range(self.max_depth - len(kwe_list)):
            padded_kwe_list.append((None, None, None))
        return padded_kwe_list

    def mutate(self, kwe_list, prob):
        def mutate_value(choices, original_value=None):
            if not choices:
                return None
            if np.random.rand() < prob or original_value is None:
                return np.random.choice(choices)
            else:
                return original_value

        depth = mutate_value(self.depth_list, len(kwe_list))
        padded_kwe_list = self._pad_kwe_list(kwe_list)

        rv = []
        for k, w, e in padded_kwe_list[:depth]:
            k = mutate_value(self.kernel_list, k)
            w = mutate_value(self.width_list , w)
            e = mutate_value(self.expand_list, e)
            rv.append((k, w, e))
        return rv

    def crossover(self, kwe_list1, kwe_list2):
        def choose_between(kwe1, kwe2):
            if kwe1[0] is None:
                return kwe2 
            if kwe2[0] is None:
                return kwe1 
            k1, w1, e1 = kwe1
            k2, w2, e2 = kwe2 
            return np.random.choice([k1, k2]), np.random.choice([w1, w2]), np.random.choice([e1, e2])

        depth1 = len(kwe_list1)
        depth2 = len(kwe_list2)
        kwe_list1 = self._pad_kwe_list(kwe_list1)
        kwe_list2 = self._pad_kwe_list(kwe_list2)
        depth = np.random.choice([depth1, depth2])
        rv = []
        for kwe1, kwe2 in zip(kwe_list1[:depth], kwe_list2[:depth]):
            rv.append(choose_between(kwe1, kwe2))
        return rv

