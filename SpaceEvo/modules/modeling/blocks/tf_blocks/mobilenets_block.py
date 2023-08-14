from __future__ import annotations

from tensorflow.keras import layers

from .base_block import BaseBlock, BaseBlockConfig, get_activation
from modules.modeling.ops.tf_ops import QConvNormActivation, QDWConvNormActivation, QDWConvK3NormActivation, QSE
from modules.modeling.common.utils import make_divisible


class _QInvertedResidual(BaseBlock):
    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: float, stride: int, 
                    use_se: bool=False, activation_layer=layers.ReLU,
                    norm_layer=layers.BatchNormalization, inplace=True, force_residual=False, DWConvNA=QDWConvNormActivation):
    
        self.use_res_connect = stride == 1 and cin == cout
        self.use_se = use_se

        feature_size = make_divisible(cin * expand_ratio)
        activation_name, activation_layer = get_activation(activation_layer)

        super().__init__(cin, cout, kernel_size, expand_ratio, stride, activation_name)

        # expand
        if cin != feature_size:
            self.point_conv1 = QConvNormActivation(cin, feature_size, kernel_size=1, norm_layer=norm_layer, 
                                                    activation_layer=activation_layer, inplace=inplace)
                                                    

        # depthwise
        self.depth_conv = DWConvNA(feature_size, kernel_size, stride, 
                                                norm_layer=norm_layer, activation_layer=activation_layer)
        if use_se:
            self.se = QSE(feature_size)

        # project
        self.point_conv2 = QConvNormActivation(feature_size, cout, kernel_size=1, norm_layer=norm_layer, 
                                                    activation_layer=None, inplace=inplace)

        # residual connection
        self.force_residual = force_residual
        if force_residual:
            self.downsample = QConvNormActivation(cin, cout, kernel_size=1, stride=stride, norm_layer=norm_layer, activation_layer=None)

    def call(self, x):
        if self.use_res_connect:
            x0 = x
        elif self.force_residual:
            x0 = self.downsample(x)

        if hasattr(self, 'point_conv1'):
            x = self.point_conv1(x)
        x = self.depth_conv(x)
        if hasattr(self, 'se'):
            x = self.se(x)
        x = self.point_conv2(x)
        if self.use_res_connect or self.force_residual:
            x = x0 + x
        return x


class MobileNetV1Block(BaseBlock):
    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int) -> None:
        super().__init__(cin, cout, kernel_size, 0, stride)
        self.depth_conv = QDWConvNormActivation(channels=cin, kernel_size=kernel_size, stride=stride)
        self.point_conv = QConvNormActivation(in_channels=cin, out_channels=cout, kernel_size=1)

    def call(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

    @classmethod
    def build_from_config(cls, config: BaseBlockConfig) -> MobileNetV1Block:
        return cls(cin=config.cin, cout=config.cout, kernel_size=config.kernel_size, stride=config.stride)


class MobileNetV1DualBlock(BaseBlock):
    
    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int) -> None:
        super().__init__(cin, cout, kernel_size, 0, stride)
        self.depth_conv1 = QDWConvNormActivation(channels=cin, kernel_size=kernel_size, stride=stride)
        self.point_conv1 = QConvNormActivation(in_channels=cin, out_channels=cout, kernel_size=1)
        self.depth_conv2 = QDWConvNormActivation(channels=cout, kernel_size=kernel_size, stride=1)
        self.point_conv2 = QConvNormActivation(in_channels=cout, out_channels=cout, kernel_size=1)
        if cin != cout or stride == 2:
            self.downsample = QConvNormActivation(cin, cout, kernel_size=1, stride=stride, activation_layer=None)
        else:
            self.downsample = None

    def call(self, x):
        if self.downsample:
            x0 = self.downsample(x)
        else:
            x0 = x
        x = self.depth_conv1(x)
        x = self.point_conv1(x)
        x = self.depth_conv2(x)
        x = self.point_conv2(x)
        x = x0 + x
        return x

    @classmethod
    def build_from_config(cls, config: BaseBlockConfig) -> MobileNetV1DualBlock:
        return cls(cin=config.cin, cout=config.cout, kernel_size=config.kernel_size, stride=config.stride)


class MobileNetV2Block(_QInvertedResidual):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: float, stride: int,
                    activation_layer=layers.ReLU, norm_layer=layers.BatchNormalization, inplace=True):
        super().__init__(cin, cout, kernel_size, expand_ratio, stride, False, activation_layer, norm_layer, inplace)

    @classmethod
    def build_from_config(cls, config: BaseBlockConfig) -> MobileNetV2Block:
        return cls(cin=config.cin, cout=config.cout, kernel_size=config.kernel_size,
                    expand_ratio=config.expand_ratio, stride=config.stride, activation_layer=config.activation)


class MobileNetV3Block(_QInvertedResidual):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: float, stride: int,
                    activation_layer=layers.ReLU, norm_layer=layers.BatchNormalization, inplace=True):
        super().__init__(cin, cout, kernel_size, expand_ratio, stride, True, activation_layer, norm_layer, inplace)

    @classmethod
    def build_from_config(cls, config: BaseBlockConfig) -> MobileNetV3Block:
        return cls(cin=config.cin, cout=config.cout, kernel_size=config.kernel_size,
                    expand_ratio=config.expand_ratio, stride=config.stride, activation_layer=config.activation)


class MobileNetV2ResBlock(_QInvertedResidual):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: float, stride: int,
                    activation_layer=layers.ReLU, norm_layer=layers.BatchNormalization, inplace=True):
        super().__init__(cin, cout, kernel_size, expand_ratio, stride, False, activation_layer, norm_layer, inplace, force_residual=True)

    @classmethod
    def build_from_config(cls, config: BaseBlockConfig) -> MobileNetV2ResBlock:
        return cls(cin=config.cin, cout=config.cout, kernel_size=config.kernel_size,
                    expand_ratio=config.expand_ratio, stride=config.stride, activation_layer=config.activation)


class MobileNetV3ResBlock(_QInvertedResidual):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: float, stride: int,
                    activation_layer=layers.ReLU, norm_layer=layers.BatchNormalization, inplace=True):
        super().__init__(cin, cout, kernel_size, expand_ratio, stride, True, activation_layer, norm_layer, inplace, force_residual=True)

    @classmethod
    def build_from_config(cls, config: BaseBlockConfig) -> MobileNetV3ResBlock:
        return cls(cin=config.cin, cout=config.cout, kernel_size=config.kernel_size,
                    expand_ratio=config.expand_ratio, stride=config.stride, activation_layer=config.activation)


class MobileNetV2K3ResBlock(_QInvertedResidual):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: float, stride: int,
                    activation_layer=layers.ReLU, norm_layer=layers.BatchNormalization, inplace=True):
        super().__init__(cin, cout, kernel_size, expand_ratio, stride, False, activation_layer, norm_layer, inplace, force_residual=True, DWConvNA=QDWConvK3NormActivation)

    @classmethod
    def build_from_config(cls, config: BaseBlockConfig) -> MobileNetV2K3ResBlock:
        return cls(cin=config.cin, cout=config.cout, kernel_size=config.kernel_size,
                    expand_ratio=config.expand_ratio, stride=config.stride, activation_layer=config.activation)


class MobileNetV3K3ResBlock(_QInvertedResidual):

    def __init__(self, cin: int, cout: int, kernel_size: int, expand_ratio: float, stride: int,
                    activation_layer=layers.ReLU, norm_layer=layers.BatchNormalization, inplace=True):
        super().__init__(cin, cout, kernel_size, expand_ratio, stride, True, activation_layer, norm_layer, inplace, force_residual=True, DWConvNA=QDWConvK3NormActivation)

    @classmethod
    def build_from_config(cls, config: BaseBlockConfig) -> MobileNetV3K3ResBlock:
        return cls(cin=config.cin, cout=config.cout, kernel_size=config.kernel_size,
                    expand_ratio=config.expand_ratio, stride=config.stride, activation_layer=config.activation)