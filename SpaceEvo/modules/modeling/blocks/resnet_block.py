from __future__ import annotations

from torch import Tensor, nn

from .base_block import BaseBlock, BaseBlockConfig
from modules.modeling.ops import QConvNormActivation, QSE4Resnet
from modules.modeling.common.utils import make_divisible
from modules.modeling.ops.lsq_plus import DEFAULT_BITS


class _ResNetBaseBlock(BaseBlock):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: float, stride: int, activation='relu', use_se=False) -> None:
        super().__init__(cin, cout, kernel_size, expand_ratio, stride, activation)
        hidden_size = make_divisible(self.cout * expand_ratio)

        self.conv1 = QConvNormActivation(cin, hidden_size, kernel_size=1, stride=1, activation_layer=self.activation_layer)
        self.conv2 = QConvNormActivation(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, activation_layer=self.activation_layer)
        self.conv3 = QConvNormActivation(hidden_size, cout, kernel_size=1, stride=1, activation_layer=None)

        if stride > 1 or cin != cout:
            self.downsample = QConvNormActivation(cin, cout, kernel_size=1, stride=stride, activation_layer=None)
        else:
            self.downsample = None

        self.activation = self.activation_layer()

        if use_se:
            self.se = QSE4Resnet(hidden_size, QSE4Resnet.get_mid_channels(cin))

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample:
            identity = self.downsample(x)
        else:
            identity = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        if hasattr(self, 'se'):
            x = self.se(x)
        x = self.conv3(x)

        x = x + identity
        x = self.activation(x)
        return x

    @classmethod
    def build_from_config(cls, config: BaseBlockConfig) -> _ResNetBaseBlock:
        return cls(cin=config.cin, cout=config.cout, kernel_size=config.kernel_size,
                    expand_ratio=config.expand_ratio, stride=config.stride, activation=config.activation)


class ResNetBlock(_ResNetBaseBlock):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: float, stride: int, activation='relu') -> None:
        super().__init__(cin, cout, kernel_size, expand_ratio, stride, activation, use_se=False)


class ResNetSEBlock(_ResNetBaseBlock):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: float, stride: int, activation='relu') -> None:
        super().__init__(cin, cout, kernel_size, expand_ratio, stride, activation, use_se=True)


class ResNetBugBlock(_ResNetBaseBlock):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: float, stride: int, activation='relu') -> None:
        super().__init__(cin, cout, kernel_size, expand_ratio, stride, activation, use_se=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample:
            identity = self.downsample(x)
        else:
            identity = x
        
        x = self.activation(identity)
        return x
