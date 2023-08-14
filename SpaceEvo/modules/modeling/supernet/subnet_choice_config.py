from __future__ import annotations

import numpy as np


class SubnetChoiceConfig:

    def __init__(self, stage_choice_list, resolution):
        self.stage_choice_list = stage_choice_list # stage_choice: list of kwe, e.g. [(k1, w1, e1), (k2, w2, e2)]
        self.resolution = resolution

    def __iter__(self):
        for v in self.stage_choice_list:
            yield v
            
    def __str__(self) -> str:
        depth_list = []
        kernel_list = []
        width_list = []
        expand_list = []

        for stage_choice in self.stage_choice_list:
            depth_list.append(len(stage_choice))
            for k, w, e in stage_choice:
                kernel_list.append(k)
                width_list.append(w)
                expand_list.append(float(e) if e else 0)
            
        array2str_func = lambda x, t: t + '#'.join([str(v) for v in x])
        depth_str = array2str_func(depth_list, 'd')
        kernel_str =array2str_func(kernel_list, 'k')
        width_str = array2str_func(width_list, 'w')
        expand_str = array2str_func(expand_list, 'e')
        resolution_str = f'r{self.resolution}'
        return '_'.join([depth_str, kernel_str, width_str, expand_str, resolution_str])

    @classmethod
    def build_from_str(cls, coding: str) -> SubnetChoiceConfig:
        depth_str, kernel_str, width_str, expand_str, resolution_str = coding.split('_')
        str2array_func = lambda x, dtype: [dtype(v) for v in x[1:].split('#')]
        depth_list = str2array_func(depth_str, int)
        kernel_list = str2array_func(kernel_str, int)
        width_list = str2array_func(width_str, int)
        expand_list = str2array_func(expand_str, float)

        assert np.sum(depth_list) == len(kernel_list)
        
        stage_choice_list = []
        block_idx = 0
        for depth in depth_list:
            stage_choice = []
            for _ in range(depth):
                k = kernel_list[block_idx]
                w = width_list[block_idx]
                e = expand_list[block_idx]
                if e == 0: e = None
                stage_choice.append((k, w, e))
                block_idx += 1
            stage_choice_list.append(stage_choice)
        return SubnetChoiceConfig(stage_choice_list, int(resolution_str[1:]))

    def __eq__(self, other) -> bool:
        return str(self) == str(other)