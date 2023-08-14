from __future__ import annotations
from typing import Dict, Optional, Union, Tuple
import tensorflow as tf
from tensorflow.keras import layers
from ....modeling.ops.tf_ops.base_ops import HSigmoid, Relu6, HSwish, Swish
from ..base_block import BaseBlockConfig


def get_activation(activation):
    name_module_pairs = [
        ('relu', layers.ReLU), ('relu6', Relu6), ('hswish', HSwish), ('hsigmoid', HSigmoid), ('swish', Swish), ('none', None)
    ]
    if not isinstance(activation, str):
        for name, module in name_module_pairs:
            if activation == module:
                return name, module
        return 'unknown', module
    else:
        for name, module in name_module_pairs:
            if activation == name:
                return name, module
        raise ValueError(f'unrecognized activation {activation}')


class BaseBlock(tf.keras.Model):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: Optional[float], stride: int, activation='relu') -> None:
        super().__init__()
        self.cin = cin
        self.cout = cout 
        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio
        self.stride = stride
        # import pdb; pdb.set_trace()
        self.activation_name, self.activation_layer = get_activation(activation)

    def call(self, x):
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
