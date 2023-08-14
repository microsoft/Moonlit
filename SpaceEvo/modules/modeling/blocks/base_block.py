from __future__ import annotations
from typing import Dict, Optional, Union
from torch import Tensor, nn

from modules.modeling.common.utils import get_activation


class BaseBlockConfig:

    def __init__(self, name: str, cin: int, cout: int, kernel_size: int, expand_ratio: Optional[float], 
                stride: int, activation: str='relu') -> None:
        self.name = name
        self.cin = cin
        self.cout = cout
        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.activation = activation

    def __str__(self) -> str:
        return '_'.join([f"{k}:{v}" for k, v in self.__dict__.items()])


class BaseBlock(nn.Module):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: Optional[float], stride: int, activation='relu') -> None:
        super().__init__()
        self.cin = cin
        self.cout = cout 
        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.activation_name, self.activation_layer = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    @staticmethod
    def build_config(name: str, cin: int, cout: int, kernel_size: int, expand_ratio: Optional[float], 
                    stride: int, activation: str='relu') -> BaseBlockConfig:
        return BaseBlockConfig(name=name, cin=cin, cout=cout, kernel_size=kernel_size,
                                expand_ratio=expand_ratio, stride=stride, activation=activation)

    @property
    def config(self) -> BaseBlockConfig:
        return self.build_config(name=type(self).__name__, cin=self.cin, cout=self.cout, 
                            kernel_size=self.kernel_size, expand_ratio=self.expand_ratio, 
                            stride=self.stride, activation=self.activation_name)

    @classmethod
    def build_from_config(cls, config: BaseBlockConfig) -> BaseBlock:
        raise NotImplementedError()
