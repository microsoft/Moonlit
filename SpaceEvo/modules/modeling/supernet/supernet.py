from __future__ import annotations
from typing import Iterator, List, Optional, OrderedDict, Union

import numpy as np
import torch
from torch import nn, Tensor

from modules.search_space.superspace.superspace import SuperSpace
from modules.modeling.network import QNetwork, QNetworkConfig
from .supernet_config import SupernetConfig
from .subnet_choice_config import SubnetChoiceConfig
from .stage import Stage


class Supernet(nn.Module):

    def __init__(self, superspace: Union[str, SuperSpace], building_block_choices: Union[str, List[int]], width_window_choices: Optional[Union[str, List[int]]]=None) -> None:
        super().__init__()
        self.config = SupernetConfig(superspace, building_block_choices, width_window_choices)

        stages = []
        max_cin = self.macro_arch.cin
        for stage_config, building_block, width_window_choice in zip(self.macro_arch.stage_configs, self.building_blocks, self.width_window_choices):
            stages.append(
                (stage_config.name, 
                Stage.build_from_config(max_cin=max_cin, config=stage_config, building_block=building_block, width_window_choice=width_window_choice))
            )
            max_cin = stage_config.max_width
            
        self.stages = nn.Sequential(OrderedDict(stages))
        self.active_resolution = self.macro_arch.max_hw

        self.align_sample = False

    @property
    def macro_arch(self):
        return self.config.superspace.macro_arch
    
    @property
    def building_blocks(self):
        return self.config.superspace.get_building_blocks(self.config.building_blocks_choices)

    @property
    def width_window_choices(self):
        return self.config.superspace.get_padded_width_window_choices(self.config.width_window_choices)

    def forward(self, x: Tensor) -> Tensor:
        # resize input to active_resolution first
        if x.shape[-1] != self.active_resolution:
            x = torch.nn.functional.interpolate(x, size=self.active_resolution, mode='bicubic')
        return self.stages(x)

    def set_dropout_rate(self, v: float):
        self.stages.logits.blocks[0].dropout.p = v
        
    @classmethod
    def build_from_config(cls, config: SupernetConfig) -> Supernet:
        return cls(superspace=config.superspace, building_block_choices=config.building_blocks_choices, width_window_choices=config.width_window_choices)

    @classmethod
    def build_from_str(cls, supernet_encoding: str) -> Supernet:
        config = SupernetConfig(*supernet_encoding.split('-'))
        return cls.build_from_config(config)

    @property
    def searchable_stages(self) -> Iterator[Stage]:
        for stage, config in zip(self.stages, self.macro_arch.stage_configs):
            if self.macro_arch.need_search(config):
                yield stage

    def set_active_subnet(self, subnet_choice: Union[SubnetChoiceConfig, str]):
        if isinstance(subnet_choice, str):
            subnet_choice = SubnetChoiceConfig.build_from_str(subnet_choice)
        for stage_choice, stage in zip(subnet_choice.stage_choice_list, self.searchable_stages):
            stage.set_active_sub_stage(stage_choice)
        self.active_resolution = subnet_choice.resolution
        
    def sample_active_subnet(self) -> SubnetChoiceConfig:
        substage_choice_list = []
        for stage in self.searchable_stages:
            kwe_list = stage.sample_active_sub_stage(align=self.align_sample)
            substage_choice_list.append(kwe_list)
       # print('resolution',self.macro_arch.supernet_hw_list)
        #self.active_resolution = np.random.choice(self.macro_arch.supernet_hw_list)
        self.active_resolution = np.random.choice([192,208,224])
       # print('resolution',self.active_resolution)
        return SubnetChoiceConfig(substage_choice_list, self.active_resolution)
    
    def sample_min_subnet(self) -> SubnetChoiceConfig:
        substage_choice_list = []
        for stage in self.searchable_stages:
            kwe_list = stage.sample_min_sub_stage(align=self.align_sample)
            substage_choice_list.append(kwe_list)
        self.active_resolution = self.macro_arch.supernet_hw_list[0]
        #self.active_resolution = self.macro_arch.supernet_hw_list[1]
        return SubnetChoiceConfig(substage_choice_list, self.active_resolution)


    def set_max_subnet(self):
        for stage in self.searchable_stages:
            stage.set_max_sub_stage()
        self.active_resolution = self.macro_arch.max_hw
    
    def set_min_subnet(self):
        for stage in self.searchable_stages:
            stage.set_min_sub_stage()
        self.active_resolution = self.macro_arch.min_hw
        #self.active_resolution = 176
            
    def get_active_subnet(self) -> QNetwork:
        stages = []
        cin = self.macro_arch.cin
        for stage_config, stage in zip(self.macro_arch.stage_configs, self.stages):
            sub_stage = stage.get_active_sub_stage(cin)
            stages.append((stage_config.name, sub_stage))
            cin = sub_stage[-1].cout
        return QNetwork(stages, self.active_resolution)

    def get_active_tf_subnet(self):
        from modules.modeling.network.tf_network import TfNetwork
        stages = []
        cin = self.macro_arch.cin
        for stage_config, stage in zip(self.macro_arch.stage_configs, self.stages):
            sub_stage = stage.get_active_tf_sub_stage(cin, stage_config.name)
            stages.append(sub_stage)
            cin = sub_stage.layers[-1].cout
        return TfNetwork(stages=stages, resolution=self.active_resolution)

    def get_active_subnet_config(self) -> QNetworkConfig:
        rv = []
        cin = self.macro_arch.cin
        for stage_config, stage in zip(self.macro_arch.stage_configs, self.stages):
            sub_stage_config = stage.get_active_sub_stage_config(cin)
            rv.append(sub_stage_config)
            cin = sub_stage_config[-1].cout
        return QNetworkConfig(rv, self.active_resolution)

    def zero_last_gamma(self):
        for stage in self.stages:
            for block in stage.blocks:
                block.zero_last_gamma()

    # === mutate and crossover during evolutional search ===

    def mutate(self, subnet_choice: SubnetChoiceConfig, arch_prob: float, resolution_prob=None) -> SubnetChoiceConfig:
        resolution_prob = resolution_prob or arch_prob
        rv = []
        for stage, kwe_list in zip(self.searchable_stages, subnet_choice):
            kwe_list = stage.mutate(kwe_list, arch_prob)
            rv.append(kwe_list)
        if np.random.rand() < resolution_prob:
            #resolution = np.random.choice(self.macro_arch.supernet_hw_list)
            resolution = np.random.choice([192,208,224])
        else:
            resolution = subnet_choice.resolution
        rv = SubnetChoiceConfig(rv, resolution)
        self.set_active_subnet(rv)
        return rv
        
    def crossover(self, sc1: SubnetChoiceConfig, sc2: SubnetChoiceConfig) -> SubnetChoiceConfig:
        resolution = np.random.choice([sc1.resolution, sc2.resolution])
        rv = []
        for stage, kwe_list1, kwe_list2 in zip(self.searchable_stages, sc1, sc2):
            rv.append(stage.crossover(kwe_list1, kwe_list2))
        rv = SubnetChoiceConfig(rv, resolution)
        self.set_active_subnet(rv)
        return rv