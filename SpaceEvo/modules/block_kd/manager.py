from typing import List, Tuple
import numpy as np
import torch 
from torch import nn

from modules.modeling.ops import DynamicQConv2d
from modules.modeling.supernet import Stage
from modules.search_space.superspace import SuperSpace
from modules.block_kd.teacher import EfficientNet


class StagePlusProj(nn.Module):

    def __init__(self, stage: Stage, cin: int, cout: int, stage_cin_list, use_in_proj=True, use_out_proj=True) -> None:
        super().__init__()
        self.stage = stage 
        self.cin = cin
        self.cout = cout
        self.stage_cin_list = stage_cin_list

        if use_in_proj:
            self.in_proj = DynamicQConv2d(cin, stage.max_cin, kernel_size_list=[1], stride=1)
        if use_out_proj:
            self.out_proj = DynamicQConv2d(stage.max_width, cout, kernel_size_list=[1], stride=1)

    def forward(self, x) -> torch.Tensor:
        if hasattr(self, 'in_proj'):
            x = self.in_proj(x)
        x = self.stage(x)
        if hasattr(self, 'out_proj'):
            x = self.out_proj(x)
        return x

    def set_max_sub_stage(self):
        self.stage.set_max_sub_stage()
        if hasattr(self, 'in_proj'):
            self.in_proj.active_out_channels = max(self.stage_cin_list)

    def set_min_sub_stage(self):
        self.stage.set_min_sub_stage()
        if hasattr(self, 'in_proj'):
            self.in_proj.active_out_channels = min(self.stage_cin_list)

    def sample_active_sub_stage(self, width_list=None):
        rv = self.stage.sample_active_sub_stage(width_list=width_list)
        if hasattr(self, 'in_proj'):
            self.in_proj.active_out_channels = np.random.choice(self.stage_cin_list)
        return rv

    def set_active_sub_stage(self, stage_cin, kwe_list):
        if hasattr(self, 'in_proj'):
            self.in_proj.active_out_channels = stage_cin 
        self.stage.set_active_sub_stage(kwe_list)


class BlockKDManager:

    def __init__(self, superspace: SuperSpace, teacher: EfficientNet) -> None:
        self.superspace = superspace
        self.teacher = teacher

    def get_stages(self, stage_name: str) -> List[Tuple[str, StagePlusProj]]:
        stage_idx = self.stage_name_to_idx(stage_name)
        # teacher
        teacher_cin = self.teacher.get_cin(stage_name)
        teacher_cout = self.teacher.get_cout(stage_name)
        # student
        max_cin = self.stage_cin_list[stage_idx]
        building_blocks = self.get_building_blocks(stage_name)
        
        rv = []
        for i, building_block in enumerate(building_blocks):
            # set width_window_choice = -1 to return the whole width list
            stage = Stage.build_from_config(max_cin, self.stage_configs[stage_idx], building_block, width_window_choice=-1)
            use_out_proj = stage_name != 'logits'
            use_in_proj = stage_name != 'first_conv'
            stage_cin_list = [3] if stage_idx == 0 else self.stage_configs[stage_idx - 1].width_list
            student_stage = StagePlusProj(stage=stage, cin=teacher_cin, cout=teacher_cout, stage_cin_list=stage_cin_list,
                                        use_in_proj=use_in_proj, use_out_proj=use_out_proj)
            rv.append((f'{stage_name}_{i}', student_stage))
        return rv

    # ========== helper functions ==========

    def get_building_blocks(self, stage_name: str):
        if stage_name in self.superspace.other_block_dict:
            return [self.superspace.other_block_dict[stage_name]]
        else:
            return self.superspace.building_block_candidates

    def stage_name_to_idx(self, stage_name: str) -> int:
        assert stage_name in self.stage_name_list, f'{stage_name} not in {self.stage_name_list}'
        name2idx_dict = {}
        for i, name in enumerate(self.stage_name_list):
            name2idx_dict[name] = i
        return name2idx_dict[stage_name]

    @property
    def macro_arch(self):
        return self.superspace.macro_arch

    @property
    def stage_configs(self):
        return self.macro_arch.stage_configs

    @property
    def stage_name_list(self):
        rv = []
        for config in self.stage_configs:
            rv.append(config.name)
        return rv 

    @property
    def stage_cin_list(self):
        cin = self.macro_arch.cin
        rv = [cin]
        for config in self.stage_configs[:-1]:
            rv.append(config.max_width)
        return rv

    def get_hw(self, stage_name, hw=224) -> int:
        for name, stage_configs in zip(self.stage_name_list, self.stage_configs):
            if name == stage_name: 
                break
            hw //= stage_configs.stride
        return hw

    def get_num_width_window_choices(self, stage_name) -> int:
        idx = self.stage_name_to_idx(stage_name)
        return self.stage_configs[idx].num_width_window_choices
    
    def get_active_width_list(self, stage_name, c) -> List[int]:
        idx = self.stage_name_to_idx(stage_name)
        return self.stage_configs[idx].get_active_width_list(c)