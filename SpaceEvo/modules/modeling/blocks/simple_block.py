from __future__ import annotations

from torch import nn, Tensor

from modules.modeling.ops import QConvNormActivation, QLinear
from .base_block import BaseBlock, BaseBlockConfig


class _ConvBnActBlock(BaseBlock):

    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int,
                activation='relu', inplace=True) -> None:
        super().__init__(cin, cout, kernel_size, None, stride, activation)

        self.conv = QConvNormActivation(cin, cout, kernel_size, stride, activation_layer=self.activation_layer, inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x


    @classmethod
    def build_from_config(cls, config: BaseBlockConfig) -> _ConvBnActBlock:
        return cls(config.cin, config.cout, config.kernel_size, config.stride, config.activation)


class FirstConvBlock(_ConvBnActBlock):

    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int, activation='relu', inplace=True) -> None:
        super().__init__(cin, cout, kernel_size, stride, activation, inplace)


class FinalExpandBlock(_ConvBnActBlock):

    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int, activation='hswish', inplace=True) -> None:
        super().__init__(cin, cout, kernel_size, stride, activation, inplace)
        

class FeatureMixBlock(_ConvBnActBlock):

    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int, activation='hswish', inplace=True) -> None:
        super().__init__(cin, cout, kernel_size, stride, activation, inplace)
    
    def forward(self, x: Tensor):
        x = x.mean([2, 3], keepdim=True)
        x = self.conv(x)
        return x


class LogitsBlock(BaseBlock):

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(cin=in_features, cout=out_features, kernel_size=0, expand_ratio=None, stride=0)
        self.linear = QLinear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = x.squeeze()
        x = self.linear(x)
        return x

    @classmethod
    def build_from_config(cls, config: BaseBlockConfig) -> LogitsBlock:
        return LogitsBlock(config.cin, config.cout)
