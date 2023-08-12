from typing import List, Optional, Tuple, Union
from torch import Tensor, nn

import numpy as np

from modules.modeling.blocks import BaseBlock, BaseBlockConfig

from modules.modeling.common.utils import get_activation


class BaseDynamicBlock(nn.Module):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], stride: int, 
                expand_list: Optional[List[float]]=None, activation='relu') -> None:
        super().__init__()
        self.max_cin = max_cin
        self.width_list = width_list
        self.stride = stride
        self.activation_name, self.activation_layer = get_activation(activation)
        
        self.kernel_list = kernel_list
        self.expand_list = expand_list or []

        # We store the active_expand_ratio but not the active_kernel_size and active_width
        # because the latter two values can be directly drived from the ops in the block
        # but expand_ratio is the ratio of two widths, and one of them is maken_divisible by 8: 
        #               width2 = make_divisible(width1 * expand_ratio).
        self._active_expand_ratio = self.max_expand_ratio
            
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    # ====  max config ====
    @property
    def max_kernel_size(self) -> int:
        return max(self.kernel_list)

    @property
    def max_width(self) -> int:
        return max(self.width_list)

    @property
    def max_expand_ratio(self) -> Optional[float]: 
        if self.expand_list:
            return max(self.expand_list)
        return None

    # ====  active config setter and getter ====
    @property
    def active_kernel_size(self) -> int:
        raise NotImplementedError()

    @active_kernel_size.setter
    def active_kernel_size(self, kernel_size: int):
        raise NotImplementedError()

    @property
    def active_width(self) -> int:
        raise NotImplementedError()

    @active_width.setter
    def active_width(self, width: int):
        raise NotImplementedError()

    @property
    def active_expand_ratio(self) -> Optional[float]:
        raise NotImplementedError()

    @active_expand_ratio.setter
    def active_expand_ratio(self, expand_ratio: float):
        raise NotImplementedError()

    # ==== set, sample and get active block ==== 
    def sample_active_block(self, width_list=None) -> Tuple[int, int, Optional[float]]:
        self.active_kernel_size = np.random.choice(self.kernel_list)
        self.active_width = np.random.choice(width_list or self.width_list)
        if self.expand_list:
            self.active_expand_ratio = np.random.choice(self.expand_list)
        return self.active_kernel_size, self.active_width, self.active_expand_ratio
    def sample_min_block(self, width_list=None) -> Tuple[int, int, Optional[float]]:
        self.active_kernel_size = self.kernel_list[0]
        if width_list:
            self.active_width=width_list[0]
        if self.width_list:
            self.active_width=self.width_list[0]
        if self.expand_list:
            self.active_expand_ratio = self.expand_list[0]
        return self.active_kernel_size, self.active_width, self.active_expand_ratio

    def set_active_block(self, kernel_size: Optional[int]=None, width: Optional[int]=None, expand_ratio: Optional[float]=None):
        self.active_kernel_size = kernel_size or self.active_kernel_size
        self.active_width = width or self.active_width
        if expand_ratio:
            self.active_expand_ratio = expand_ratio

    def set_active_block_from_config(self, config: BaseBlockConfig):
        self.set_active_block(kernel_size=config.kernel_size, width=config.cout, expand_ratio=config.expand_ratio)
        
    def set_max_block(self):
        max_expand_ratio = max(self.expand_list) if self.expand_list else None
        self.set_active_block(max(self.kernel_list), max(self.width_list), max_expand_ratio)

    def set_min_block(self):
        min_expand_ratio = min(self.expand_list) if self.expand_list else None
        self.set_active_block(min(self.kernel_list), min(self.width_list), min_expand_ratio)

    def get_active_block_config(self, cin: int) -> BaseBlockConfig:
        return BaseBlockConfig(name=type(self).__name__, cin=cin, cout=self.active_width, kernel_size=self.active_kernel_size,
                                expand_ratio=self.active_expand_ratio, stride=self.stride, activation=self.activation_name)
        
    def get_active_block(self, cin: int, retain_weights=True) -> BaseBlock:
        raise NotImplementedError()

    def get_active_tf_block(self, cin: int):
        raise NotImplementedError()
    
    # === useful for building look up table ===
    def list_all_sub_block_configs(self, cin_list: Union[List, int]) -> List[BaseBlockConfig]:
        if isinstance(cin_list, int):
            cin_list = [cin_list]
        rv = []
        for cin in cin_list:
            for width in self.width_list:
                for kernel_size in self.kernel_list:
                    expand_list =  self.expand_list or [None]
                    for expand in expand_list:
                        rv.append(
                            BaseBlockConfig(
                                name=type(self).__name__, 
                                cin=cin, 
                                cout=width, 
                                kernel_size=kernel_size,
                                expand_ratio=expand,
                                stride=self.stride,
                                activation=self.activation_name
                        ))

    # === help training ===
    def zero_last_gamma(self):
        pass