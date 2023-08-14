from functools import partial
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from modules.search_space.macro_arch import MacroArch, StageConfig
from modules.modeling.dynamic_blocks import BaseDynamicBlock


class SuperSpace:

    NAME = 'base'

    def __init__(self, macro_arch: MacroArch, building_block_candidates: List[Callable[..., BaseDynamicBlock]],
                other_block_dict: Dict[str, Callable[..., BaseDynamicBlock]]) -> None:
        self.macro_arch = macro_arch
        self.building_block_candidates = building_block_candidates
        self.other_block_dict = other_block_dict

    @property
    def num_searchable_stages(self) -> int:
        return len(self.macro_arch.stage_configs) - len(self.other_block_dict)

    @staticmethod
    def str_to_list(x: Union[str, List[int]]) -> List[int]:
        if isinstance(x, str):
            return [int(v) for v in x]
        return x

    @staticmethod
    def list_to_str(x: Union[str, List[int]]) -> str:
        if isinstance(x, List):
            return ''.join([str(v) for v in x])
        return x 

    def pad_choice(self, choices: List[int], padx) -> List[int]:
        if len(choices) == self.num_searchable_stages:
            padded_choice = np.ones(len(self.macro_arch.stage_configs)) * padx
            c_i = 0
            for i, stage_config in enumerate(self.macro_arch.stage_configs):
                if self.need_choose_block(stage_config):
                    padded_choice[i] = choices[c_i]
                    c_i += 1
            choices = list(padded_choice.astype(np.int32))
        assert len(choices) == len(self.macro_arch.stage_configs)
        return choices

    def get_building_blocks(self, block_choices: Union[str, List[int]]) -> List[Callable[..., BaseDynamicBlock]]:
        block_choices = self.str_to_list(block_choices)
        block_choices = self.pad_choice(block_choices, -1)
        building_blocks = []
        for stage_config, c in zip(self.macro_arch.stage_configs, block_choices):
            if c == -1:
                building_blocks.append(self.other_block_dict[stage_config.name])
            else:
                building_blocks.append(self.building_block_candidates[c])
        return building_blocks

    def get_padded_width_window_choices(self, width_window_choices: Optional[Union[str, List[int]]]=None, padded=True) -> List[int]:
        if width_window_choices is None:
            width_window_choices = [0] * self.num_searchable_stages
        width_window_choices = SuperSpace.str_to_list(width_window_choices)
        if padded:
            width_window_choices = SuperSpace.pad_choice(self, width_window_choices, 0)
        return width_window_choices

    def need_choose_block(self, stage: Union[StageConfig, str]) -> bool:
        if isinstance(stage, StageConfig):
            name = stage.name 
        else:
            name = stage
        return name not in self.other_block_dict
