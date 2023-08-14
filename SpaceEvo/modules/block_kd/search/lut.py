from collections import defaultdict
import math
import os
from typing import Dict, List, Optional

import numpy as np

from ...latency_predictor import LatencyPredictor
from ...search_space.superspace import SuperSpace
from ...modeling.supernet import Supernet 


class LUT:

    def __init__(self, lut_dir: str, platform: str, superspace: SuperSpace):
        self.lut_dir = lut_dir
        self.lut = self._build_lut()
        self.latency_predictor = LatencyPredictor(platform)
        self.superspace = superspace 
        self.other_stage_latency = self._get_other_stage_latency()

    @property
    def resolution_list(self):
        return [224, 192, 160]

    @property
    def hw_to_resolution_map(self):
        try:
            return self._hw_to_resolution_map 
        except:
            self._hw_to_resolution_map = {}
            for div in [1, 2, 4, 8, 16, 32]:
                for resolution in self.resolution_list:
                    assert resolution % div == 0
                    self._hw_to_resolution_map[resolution // div] = resolution
            return self._hw_to_resolution_map

    def _convert_resolution(self, hw):
        hw = int(hw)
        return self.hw_to_resolution_map[hw]

    @property
    def avail_stage_list(self):
        return list(self.lut[self.resolution_list[0]].keys())

    def _sample_block(self, stage_name, n=1, resolution=224):
        total_samples = len(self.lut[resolution][stage_name]['latency'])
        idx_list = np.random.choice(total_samples, n)
        loss_rv = self.lut[resolution][stage_name]['loss'][idx_list]
        latency_rv = self.lut[resolution][stage_name]['latency'][idx_list]
        return loss_rv, latency_rv

    def sample_indiv(self, block_choices: List[int], width_window_choices: List[int], n=1):
        if width_window_choices is None: # to be backward compatible with v1
            width_window_choices = [None] * len(block_choices)
        stage_loss_array = np.zeros([n, len(block_choices)])
        stage_latency_array = np.zeros([n, len(block_choices)])
        resolution = np.random.choice(self.resolution_list)
        for i, (bc, wwc) in enumerate(zip(block_choices, width_window_choices), start=1):
            if wwc is None:
                stage_choice = f'stage{i}_{bc}'
            else:
                stage_choice = f'stage{i}_{bc}_{wwc}'
            stage_loss_array[:, i - 1], stage_latency_array[:, i - 1] = self._sample_block(stage_choice, n=n, resolution=resolution)

        reduced_loss_array = np.sum(stage_loss_array, axis=1)      
        reduced_latency_array = np.sum(stage_latency_array, axis=1) + self.other_stage_latency[resolution]           
        return reduced_loss_array, reduced_latency_array

    def get_min_latency(self, block_choices: List[int], width_window_choices: Optional[List[int]]=None):
        if width_window_choices is None: # to be backward compatible with v1
            width_window_choices = [None] * len(block_choices)
        resolution = min(self.resolution_list)
        latency = self.other_stage_latency[resolution]
        for i, (bc, wwc) in enumerate(zip(block_choices, width_window_choices), start=1):
            if wwc is None:
                stage_choice = f'stage{i}_{bc}'
            else:
                stage_choice = f'stage{i}_{bc}_{wwc}'
            latency += min(self.lut[resolution][stage_choice]['latency'])
        return latency
        
    def _build_lut(self) -> Dict:
        rv = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
        for file_name in os.listdir(self.lut_dir):
            if file_name.endswith('.csv'):
                stage_name = file_name.replace('.csv', '')
                with open(os.path.join(self.lut_dir, file_name), 'r') as f:
                    for line in f.readlines():
                        sample_str, input_shape, loss, flops, params, latency = line.split(',')
                        hw = int(input_shape.split('x')[-1])
                        resolution = self._convert_resolution(hw)
                        rv[resolution][stage_name]['loss'].append(float(loss))
                        rv[resolution][stage_name]['latency'].append(float(latency))
                for resolution in self.resolution_list:
                    rv[resolution][stage_name]['loss'] = np.array(rv[resolution][stage_name]['loss'])
                    rv[resolution][stage_name]['latency'] = np.array(rv[resolution][stage_name]['latency'])
        return rv

    def _get_other_stage_latency(self) -> Dict[int, float]:
        # calculate the latency of [first_conv, stage0, final_expand, feature_mix, logits] stages
        # which are not taken into account when doing block_kd
        # in min net under resolution [160, 192, 224]
        rv = {}
        supernet = Supernet(superspace=self.superspace, building_block_choices=[0]*self.superspace.num_searchable_stages, width_window_choices=None)
        supernet.set_min_subnet()
        for resolution in self.resolution_list:
            supernet.active_resolution = resolution 
            subnet_config = supernet.get_active_subnet_config()
            hw = subnet_config.resolution
            block_args_list = []
            for stage_config, stage in zip(self.superspace.macro_arch.stage_configs, subnet_config.stages):
                    for block in stage:
                        if stage_config.name in self.superspace.other_block_dict:
                            block_args_list.append(self.latency_predictor._get_block_args(block, hw))
                        hw = math.ceil(hw / (block.stride or 1))
            latency = self.latency_predictor.block_predictor.get_latency(block_args_list)
            rv[resolution] = latency
        return rv
