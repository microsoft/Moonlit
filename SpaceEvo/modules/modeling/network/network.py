from typing import Dict, List, OrderedDict, Tuple

from torch import Tensor, nn
import torch

from modules.modeling.blocks import BaseBlockConfig, BaseBlock


class QNetworkConfig:

    def __init__(self, stages: List[List[BaseBlockConfig]], resolution=224) -> None:
        self.stages = stages
        self.resolution = resolution

    def print_self(self):
        for stage in self.stages:
            for block in stage:
                print(block.__dict__)

    def as_dict(self):
        rv = {}
        for i, stage in enumerate(self.stages):
            stage_dict = {}
            for j, block in enumerate(stage):
                stage_dict[j] = block.__dict__
            rv[f'stage{i}'] = stage_dict
        rv['resolution'] = self.resolution
        return rv

    def __str__(self) -> str:
        rv = type(self).__name__ + '\n'
        for i, stage in enumerate(self.stages):
            rv += f'stage{i}\n'
            for block in stage:
                rv += str(block) + '\n'
        rv += f'resolution:{self.resolution}'
        return rv

    def as_block_config_str_list(self) -> List[str]:
        rv = []
        for stage in self.stages:
            for block in stage:
                rv.append(str(block))
        return rv 
        
    def as_block_config_list(self) -> List[BaseBlockConfig]:
        rv = []
        for stage in self.stages:
            for block in stage:
                rv.append(block)
        return rv 

class QNetwork(nn.Module):

    def __init__(self, stages: List[Tuple[str, nn.Sequential]], resolution=224) -> None:
        super().__init__()
        self.stages = nn.Sequential(OrderedDict(stages))
        self.resolution = resolution

    def forward(self, x: Tensor) -> Tensor:
        # resize input to active_resolution first
        if x.shape[-1] != self.resolution:
            x = torch.nn.functional.interpolate(x, size=self.resolution, mode='bicubic')
        x = self.stages(x)
        return x


    @property
    def config(self) -> QNetworkConfig:
        stage_configs = []
        for stage in self.stages:
            blocks = []
            for block in stage:
                assert isinstance(block, BaseBlock)
                blocks.append(block.config)
            stage_configs.append(blocks)
        return QNetworkConfig(stage_configs, self.resolution)