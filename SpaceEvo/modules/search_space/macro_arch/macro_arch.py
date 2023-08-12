from typing import List


class StageConfig:

    def __init__(self, name: str, width_list: List[int], depth_list: List[int], stride: int, width_window_size=None) -> None:
        self.name = name
        self.width_list = width_list
        self.depth_list = depth_list
        self.stride = stride
        self.width_window_size = width_window_size or len(width_list)

    def get_active_width_list(self, width_window_choice=0):
        if width_window_choice == -1: # return the whole width_list
            return self.width_list
        assert width_window_choice < self.num_width_window_choices
        return self.width_list[width_window_choice: width_window_choice + self.width_window_size]
    
    @property
    def num_width_window_choices(self):
        return len(self.width_list) - self.width_window_size + 1

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
    def min_width(self) -> int:
        return min(self.width_list)

    def __str__(self) -> str:
        return f'name:{self.name}_width:{self.width_list}_depth:{self.depth_list}_stride:{self.stride}_wwsize:{self.width_window_size}'


class MacroArch:

    def __init__(self, stage_configs: List[StageConfig], block_kd_hw_list=[224], supernet_hw_list=[224], cin=3) -> None:
        self.stage_configs = stage_configs
        self.block_kd_hw_list = block_kd_hw_list
        self.supernet_hw_list = supernet_hw_list
        self.cin = cin
        self._assert_no_idential_stage_name()

    @property
    def max_hw(self):
        return max(self.supernet_hw_list)
    
    @property
    def min_hw(self):
        return min(self.supernet_hw_list)

    @staticmethod
    def need_search(stage: StageConfig) -> bool:
        raise NotImplementedError()

    def _assert_no_idential_stage_name(self):
        name_list = []
        for config in self.stage_configs:
            name_list.append(config.name)
        name_set = set(name_list)
        assert len(name_set) == len(name_list), 'Stage names must be mutually different.'

    def __str__(self) -> str:
        rv = type(self).__name__ + '\n'
        rv += '\tstages:\n'
        rv += ''.join(['\t\t' + str(stage) + '\n' for stage in self.stage_configs])
        rv += f'\thw:{self.supernet_hw_list}\n'
        rv += f'\tcin:{self.cin}\n'
        return rv
