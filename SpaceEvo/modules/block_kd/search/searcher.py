import copy
from json.encoder import INFINITY
from logging import warning
from typing import List, Tuple

import numpy as np

from ...search_space.superspace import get_superspace
from .lut import LUT
from ...modeling.supernet import Supernet
from ...latency_predictor import LatencyPredictor


class BlockKDSearcher:

    def __init__(self, lut_dir: str, superspace, platform, num_samples_per_indiv=100000, top=50, 
                    latency_constraint=25, lat_loss_t=0.1, lat_loss_a=0.01) -> None:
        if isinstance(superspace, str):
            self.superspace = get_superspace(superspace)
        else:
            self.superspace = superspace
        self.lut = LUT(lut_dir=lut_dir, platform=platform, superspace=self.superspace)
        self.num_samples_per_indiv = num_samples_per_indiv
        self.top = top
        if isinstance(latency_constraint, List):
            self.latency_constraint_list = latency_constraint
        else:
            self.latency_constraint_list = [latency_constraint]
        self.lat_loss_t = lat_loss_t
        self.lat_loss_a = lat_loss_a

    @property
    def latency_predictor(self) -> LatencyPredictor:
        return self.lut.latency_predictor

    @property
    def num_stages(self) -> int:
        return self.superspace.num_searchable_stages

    @property
    def num_block_choices(self) -> int:
        return len(self.superspace.building_block_candidates)

    @property
    def num_width_window_choices_list(self) -> List[int]:
        try:
            return self._num_width_window_choices_list
        except:
            stage_idxs = []
            for i in range(len(self.superspace.macro_arch.stage_configs)):
                if self.superspace.need_choose_block(self.superspace.macro_arch.stage_configs[i]):
                    stage_idxs.append(i)
            self._num_width_window_choices_list = [self.superspace.macro_arch.stage_configs[i].num_width_window_choices for i in stage_idxs]
            return self.num_width_window_choices_list

    # sample static stages from one indv (dynamic stage)
    def sample_individual(self, block_choices: List[int], width_window_choices: List[int], n=int(1e6)):
        loss_list, latency_list = [], []
        for _ in range(n // 500):
            loss_samples, latency_samples = self.lut.sample_indiv(block_choices, width_window_choices, n=500)
            loss_list.extend(loss_samples)
            latency_list.extend(latency_samples)
        return loss_list, latency_list

    def eval_individual(self, block_choices: List[int], width_window_choices: List[int]):
        score, latency_loss, acc_loss = (0, 0, 0)
        for latency_constraint in self.latency_constraint_list:
            _score, _latency_loss, _acc_loss = self._eval_individual_one_latency(block_choices, width_window_choices, latency_constraint)
            score += _score 
            acc_loss += _acc_loss
            latency_loss += _latency_loss
        return score, latency_loss, acc_loss


    def _find_latency_end(self, indiv_sample_list, latency_constraint) -> int:
        start = 0
        end = len(indiv_sample_list)
        if indiv_sample_list[end - 1]['latency'] <= latency_constraint:
            return end 
        if indiv_sample_list[start]['latency'] > latency_constraint:
            return 0
        while start < end - 1:
            mid = (start + end) >> 1
            if indiv_sample_list[mid]['latency'] <= latency_constraint:
                start = mid
            else:
                end = mid 
        return end

    def _eval_individual_one_latency(self, block_choices: List[int], width_window_choices: List[int], latency_constraint):
        assert len(block_choices) == self.num_stages, (block_choices, self.num_stages)
        
        loss_list, latency_list = [], []
        for _ in range(self.num_samples_per_indiv // 500):
            loss_sample, latency_sample = self.lut.sample_indiv(block_choices, width_window_choices, n=500)
            loss_list.extend(loss_sample)
            latency_list.extend(latency_sample)
        indiv_sample_list = [{'loss': loss, 'latency': latency} for (loss, latency) in zip(loss_list, latency_list)]
        # eval score
        # latency
        indiv_sample_list = sorted(indiv_sample_list, key=lambda x: x['latency'])
        end = self._find_latency_end(indiv_sample_list, latency_constraint)
        p_latency_smaller_than_constraint = end / self.num_samples_per_indiv + 1e-5
        # loss
        avg_top_loss = 0
        if end:
            indiv_sample_list = indiv_sample_list[:end]
            indiv_sample_list = sorted(indiv_sample_list, key=lambda x: x['loss'])
            for sample in indiv_sample_list[:self.top]:
                avg_top_loss += sample['loss']
            avg_top_loss /= len(indiv_sample_list[:self.top]) + 1e-5
        else:
            arch = ''.join([str(v) for v in block_choices]) + '_' + ''.join([str(v) for v in width_window_choices])
            warning(f'Sample 0 indv that satisfy constraint. arch: {arch}. min_lat: {min(latency_list):.2f}')
            avg_top_loss = 1e5

        # calculate score
        latency_loss = np.clip((self.lat_loss_t / p_latency_smaller_than_constraint) ** self.lat_loss_a, a_min=1, a_max=INFINITY)
        score = 1 / (latency_loss * avg_top_loss)
        return score, latency_loss, avg_top_loss

    def pred_min_max_latency(self, block_choices: List[int], width_window_choices: List[int]) -> Tuple[float, float]:
        supernet = Supernet(self.superspace, building_block_choices=block_choices, width_window_choices=width_window_choices)
        supernet.set_max_subnet()
        max_ms = self.latency_predictor.predict_subnet(supernet.get_active_subnet_config())
        supernet.set_min_subnet()
        min_ms = self.latency_predictor.predict_subnet(supernet.get_active_subnet_config())
        return max_ms, min_ms 
        
    # ===== functions for evolution =====

    def random_sample(self):
        block_choices = np.random.choice(self.num_block_choices, size=[self.num_stages])
        width_window_choices = []
        #print(self.num_width_window_choices_list)
        for num_choices in self.num_width_window_choices_list:
            width_window_choices.append(np.random.choice(num_choices))
        #print(block_choices,width_window_choices)
        #width_window_choices=[0,0,0,0,0,0]
        block_choices=[1,2,1,2,2,2]
        return block_choices, width_window_choices

    def _mutate(self, block_choices: List[int], width_window_choices, mutate_block: bool, mutate_width_window: bool) -> Tuple[List[int], List[int]]:
        block_choices = copy.deepcopy(block_choices)
        width_window_choices = copy.deepcopy(width_window_choices)
        
        # mutate block type
        if mutate_block:
            stage_idx = np.random.choice(self.num_stages)
            c = np.random.choice(self.num_block_choices)
            while c == block_choices[stage_idx]:
                c = np.random.choice(self.num_block_choices)
            block_choices[stage_idx] = c

        # mutate width window
        if mutate_width_window:
            stage_idx = np.random.choice(self.num_stages)
            c = np.random.choice(self.num_width_window_choices_list[stage_idx])
            while c == width_window_choices[stage_idx]:
                c = np.random.choice(self.num_width_window_choices_list[stage_idx])
            width_window_choices[stage_idx] = c

        return block_choices, width_window_choices

    def mutate_block(self, block_choices: List[int], width_window_choices) -> Tuple[List[int], List[int]]:
        return self._mutate(block_choices, width_window_choices, mutate_block=True, mutate_width_window=False)

    def mutate_width_window(self, block_choices: List[int], width_window_choices) -> Tuple[List[int], List[int]]:
        return self._mutate(block_choices, width_window_choices, mutate_block=False, mutate_width_window=True)