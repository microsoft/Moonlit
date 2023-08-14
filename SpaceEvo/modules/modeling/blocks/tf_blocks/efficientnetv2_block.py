from __future__ import annotations

from tensorflow.keras import layers

from .base_block import BaseBlock, BaseBlockConfig, get_activation
from modules.modeling.ops.tf_ops import QConvNormActivation, QSE4Resnet
from modules.modeling.common.utils import make_divisible


class _FusedMBConvBlock(BaseBlock):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: float, stride: int, 
                    use_se: bool=False, activation_layer=layers.ReLU,
                    norm_layer=layers.BatchNormalization, inplace=True, force_residual=False):
        self.use_res_connect = stride == 1 and cin == cout
        self.use_se = use_se

        feature_size = make_divisible(cin * expand_ratio)
        activation_name, activation_layer = get_activation(activation_layer)

        super().__init__(cin, cout, kernel_size, expand_ratio, stride, activation_name)

        self.fused_conv = QConvNormActivation(cin, feature_size, kernel_size=kernel_size, stride=stride,
        norm_layer=norm_layer, activation_layer=activation_layer, inplace=inplace)

        if use_se:
            self.se = QSE4Resnet(in_channels=feature_size, mid_channels=make_divisible(cin // QSE4Resnet.REDUCTION))

        self.point_conv = QConvNormActivation(feature_size, cout, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=None)

        # residual connection
        self.force_residual = force_residual
        if force_residual and not self.use_res_connect:
            self.downsample = QConvNormActivation(cin, cout, kernel_size=1, stride=stride, norm_layer=norm_layer, activation_layer=None)

    def call(self, x):
        if self.use_res_connect:
            x0 = x
        elif self.force_residual:
            x0 = self.downsample(x)

        x = self.fused_conv(x)

        if hasattr(self, 'se'):
            x = self.se(x)
        x = self.point_conv(x)

        if self.use_res_connect or self.force_residual:
            x = x0 + x

        return x


class FusedMBConvResBlock(_FusedMBConvBlock):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: float, stride: int, 
                    activation_layer=layers.ReLU,
                    norm_layer=layers.BatchNormalization, inplace=True):
        super().__init__(cin, cout, kernel_size, expand_ratio, stride, use_se=False, activation_layer=activation_layer, norm_layer=norm_layer, inplace=inplace, force_residual=True)

    @classmethod
    def build_from_config(cls, config: BaseBlockConfig) -> FusedMBConvResBlock:
        return cls(cin=config.cin, cout=config.cout, kernel_size=config.kernel_size, expand_ratio=config.expand_ratio, stride=config.stride, activation_layer=config.activation)


class FusedMBConvSEResBlock(_FusedMBConvBlock):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: float, stride: int, 
                    activation_layer=layers.ReLU,
                    norm_layer=layers.BatchNormalization, inplace=True):
        super().__init__(cin, cout, kernel_size, expand_ratio, stride, use_se=True, activation_layer=activation_layer, norm_layer=norm_layer, inplace=inplace, force_residual=True)

    @classmethod
    def build_from_config(cls, config: BaseBlockConfig) -> FusedMBConvSEResBlock:
        return cls(cin=config.cin, cout=config.cout, kernel_size=config.kernel_size, expand_ratio=config.expand_ratio, stride=config.stride, activation_layer=config.activation)


