from typing import Dict, List

from torch import Tensor, nn
import torch

from modules.modeling.dynamic_blocks.base_dynamic_block import BaseDynamicBlock
from modules.modeling.ops import (
    DynamicBatchNorm2d, DynamicQConv2d, DynamicQDWConv2d, 
    DynamicQConvNormActivation, DynamicQDWConvNormActivation, 
    DynamicQLinear, DynamicQSE, DynamicQDWConvK3NormActivation
)
from modules.modeling.blocks import (
    MobileNetV1Block, MobileNetV2Block, MobileNetV3Block, MobileNetV2ResBlock, MobileNetV3ResBlock, _QInvertedResidual, MobileNetV1DualBlock, MobileNetV2K3ResBlock, MobileNetV3K3ResBlock
)
from modules.modeling.common.utils import make_divisible, get_activation


class _DynamicQInvertedResidual(BaseDynamicBlock):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], expand_list: List[float], stride: int, 
                    use_se: bool, activation=nn.ReLU, inplace=True, force_residual=False, DWConvNA=DynamicQDWConvNormActivation) -> None:
        super().__init__(max_cin, width_list, kernel_list, stride, expand_list, activation)
        self.use_se = use_se
        self.inplace = inplace

        self.point_conv1 = DynamicQConvNormActivation(
            max_in_channels=self.max_cin, max_out_channels=self.max_hidden_size,
            kernel_size_list=[1], stride=1, activation_layer=self.activation_layer, inplace=inplace
        )
        self.depth_conv = DWConvNA(
            max_channels=self.max_hidden_size, kernel_size_list=self.kernel_list, stride=stride, 
            activation_layer=self.activation_layer, inplace=inplace
        )

        if use_se:
            self.se = DynamicQSE(self.max_hidden_size)

        self.point_conv2 = DynamicQConvNormActivation(
            max_in_channels=self.max_hidden_size, max_out_channels=self.max_width,
            kernel_size_list=[1], stride=1, activation_layer=None
        )

        # residual connection
        self.force_residual = force_residual
        if self.force_residual:
            self.downsample = DynamicQConvNormActivation(
                max_in_channels=self.max_cin, max_out_channels=self.max_width,
                kernel_size_list=[1], stride=stride, activation_layer=None
            )
        
    def use_res_connect(self, cin):
        return cin == self.active_width and self.stride == 1

    def forward(self, x: Tensor) -> Tensor:
        cin = x.shape[1]
        if self.use_res_connect(cin):
            x0 = x
        elif self.force_residual:
            x0 = self.downsample(x)

        active_hidden_size = self.get_active_hidden_size(cin)

        if active_hidden_size != cin:
            x = self.point_conv1.forward(x, out_channels=active_hidden_size)
        x = self.depth_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.point_conv2(x)
        if self.use_res_connect(cin) or self.force_residual:
            x = x0 + x
        return x

    @property
    def max_hidden_size(self):
        return make_divisible(self.max_expand_ratio * self.max_cin)

    def get_active_hidden_size(self, cin: int) -> int:
        return make_divisible(self.active_expand_ratio * cin)

    @property
    def active_kernel_size(self) -> int:
        return self.depth_conv.active_kernel_size

    @active_kernel_size.setter
    def active_kernel_size(self, kernel_size: int):
        self.depth_conv.active_kernel_size = kernel_size

    @property
    def active_width(self) -> int:
        return self.point_conv2.active_out_channels

    @active_width.setter
    def active_width(self, width: int):
        self.point_conv2.active_out_channels = width
        if self.force_residual:
            self.downsample.active_out_channels = width

    @property
    def active_expand_ratio(self) -> float:
        return self._active_expand_ratio

    @active_expand_ratio.setter
    def active_expand_ratio(self, expand_ratio: float):
        self._active_expand_ratio = expand_ratio

    def zero_last_gamma(self):
        if self.force_residual:
            self.point_conv2.norm.weight.data.zero_()


class DynamicMobileNetV1Block(BaseDynamicBlock):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], stride: int) -> None:
        super().__init__(max_cin, width_list, kernel_list, stride, expand_list=None)
        self.depth_conv = DynamicQDWConvNormActivation(
            max_channels=self.max_cin, kernel_size_list=self.kernel_list, stride=stride
        )
        self.point_conv = DynamicQConvNormActivation(
            max_in_channels=self.max_cin, max_out_channels=self.max_width,
            kernel_size_list=[1], stride=1
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

    @property
    def active_kernel_size(self) -> int:
        return self.depth_conv.active_kernel_size

    @active_kernel_size.setter
    def active_kernel_size(self, kernel_size: int):
        self.depth_conv.active_kernel_size = kernel_size

    @property
    def active_width(self) -> int:
        return self.point_conv.active_out_channels

    @active_width.setter
    def active_width(self, width: int):
        self.point_conv.active_out_channels = width

    @property
    def active_expand_ratio(self) -> None:
        return None

    def get_active_block(self, cin: int, retain_weights=True) -> MobileNetV1Block:
        block = MobileNetV1Block.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            block.depth_conv.load_state_dict(self.depth_conv.active_state_dict(cin))
            block.point_conv.load_state_dict(self.point_conv.active_state_dict(cin))
        return block
    
    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import MobileNetV1Block as MobileNetV1Block_TF
        return MobileNetV1Block_TF.build_from_config(self.get_active_block_config(cin))


class DynamicMobileNetV1DualBlock(BaseDynamicBlock):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], stride: int) -> None:
        super().__init__(max_cin, width_list, kernel_list, stride, expand_list=None)
        self.depth_conv1 = DynamicQDWConvNormActivation(
            max_channels=self.max_cin, kernel_size_list=self.kernel_list, stride=stride
        )
        self.point_conv1 = DynamicQConvNormActivation(
            max_in_channels=self.max_cin, max_out_channels=self.max_width,
            kernel_size_list=[1], stride=1
        )
        self.depth_conv2 = DynamicQDWConvNormActivation(
            max_channels=self.max_width, kernel_size_list=self.kernel_list, stride=1
        )
        self.point_conv2 = DynamicQConvNormActivation(
            max_in_channels=self.max_width, max_out_channels=self.max_width,
            kernel_size_list=[1], stride=1
        )
        self.downsample = DynamicQConvNormActivation(
            max_in_channels=self.max_cin, max_out_channels=self.max_width,
            kernel_size_list=[1], stride=stride, activation_layer=None
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.active_width or self.stride > 1:
            x0 = self.downsample(x)
        else:
            x0 = x
        x = self.depth_conv1(x)
        x = self.point_conv1(x)
        x = self.depth_conv2(x)
        x = self.point_conv2(x)
        x = x0 + x
        return x

    @property
    def active_kernel_size(self) -> int:
        return self.depth_conv1.active_kernel_size

    @active_kernel_size.setter
    def active_kernel_size(self, kernel_size: int):
        self.depth_conv1.active_kernel_size = kernel_size
        self.depth_conv2.active_kernel_size = kernel_size

    @property
    def active_width(self) -> int:
        return self.point_conv1.active_out_channels

    @active_width.setter
    def active_width(self, width: int):
        self.point_conv1.active_out_channels = width
        self.point_conv2.active_out_channels = width
        self.downsample.active_out_channels = width

    @property
    def active_expand_ratio(self) -> None:
        return None

    def get_active_block(self, cin: int, retain_weights=True) -> MobileNetV1DualBlock:
        block = MobileNetV1DualBlock.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            block.depth_conv1.load_state_dict(self.depth_conv1.active_state_dict(cin))
            block.point_conv1.load_state_dict(self.point_conv1.active_state_dict(cin))
            block.depth_conv2.load_state_dict(self.depth_conv2.active_state_dict(self.active_width))
            block.point_conv2.load_state_dict(self.point_conv2.active_state_dict(self.active_width))
            if block.downsample:
                block.downsample.load_state_dict(self.downsample.active_state_dict(cin))
        return block

    def zero_last_gamma(self):
        self.point_conv2.norm.weight.data.zero_()

    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import MobileNetV1DualBlock as MobileNetV1DualBlock_TF
        return MobileNetV1DualBlock_TF.build_from_config(self.get_active_block_config(cin))


class DynamicMobileNetV2Block(_DynamicQInvertedResidual):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], expand_list: List[float],
                    stride: int, activation_layer=nn.ReLU, inplace=True) -> None:
        super().__init__(max_cin, width_list, kernel_list, expand_list, stride, False, activation_layer, inplace)

    def get_active_block(self, cin: int, retain_weights=True) -> MobileNetV2Block:
        block = MobileNetV2Block.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            # hidden_size is determined by cin, so it must be calculated on the run. point_conv1.active_width has no use.
            hidden_size = self.get_active_hidden_size(cin)
            if hasattr(block, 'point_conv1'):
                block.point_conv1.load_state_dict(self.point_conv1.active_state_dict(cin, cout=hidden_size))
            block.depth_conv.load_state_dict(self.depth_conv.active_state_dict(hidden_size))
            block.point_conv2.load_state_dict(self.point_conv2.active_state_dict(hidden_size))
        return block

    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import MobileNetV2Block as MobileNetV2Block_TF
        return MobileNetV2Block_TF.build_from_config(self.get_active_block_config(cin))


class DynamicMobileNetV3Block(_DynamicQInvertedResidual):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], expand_list: List[float], 
                    stride: int, activation_layer=nn.ReLU, inplace=True) -> None:
        super().__init__(max_cin, width_list, kernel_list, expand_list, stride, True, activation_layer, inplace)

    def get_active_block(self, cin: int, retain_weights=True) -> MobileNetV3Block:
        block = MobileNetV3Block.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            hidden_size = self.get_active_hidden_size(cin)
            if hasattr(block, 'point_conv1'):
                block.point_conv1.load_state_dict(self.point_conv1.active_state_dict(cin, cout=hidden_size))
            block.depth_conv.load_state_dict(self.depth_conv.active_state_dict(hidden_size))
            block.point_conv2.load_state_dict(self.point_conv2.active_state_dict(hidden_size))
            block.se.load_state_dict(self.se.active_state_dict(hidden_size))
        return block

    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import MobileNetV3Block as MobileNetV3Block_TF
        return MobileNetV3Block_TF.build_from_config(self.get_active_block_config(cin))


class DynamicMobileNetV2ResBlock(_DynamicQInvertedResidual):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], expand_list: List[float],
                    stride: int, activation_layer=nn.ReLU, inplace=True) -> None:
        super().__init__(max_cin, width_list, kernel_list, expand_list, stride, False, activation_layer, inplace, force_residual=True)

    def get_active_block(self, cin: int, retain_weights=True) -> MobileNetV2ResBlock:
        block = MobileNetV2ResBlock.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            # hidden_size is determined by cin, so it must be calculated on the run. point_conv1.active_width has no use.
            hidden_size = self.get_active_hidden_size(cin)
            if hasattr(block, 'point_conv1'):
                block.point_conv1.load_state_dict(self.point_conv1.active_state_dict(cin, cout=hidden_size))
            block.depth_conv.load_state_dict(self.depth_conv.active_state_dict(hidden_size))
            block.point_conv2.load_state_dict(self.point_conv2.active_state_dict(hidden_size))
            if hasattr(block, 'downsample'):
                block.downsample.load_state_dict(self.downsample.active_state_dict(cin))
        return block

    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import MobileNetV2ResBlock as MobileNetV2ResBlock_TF
        return MobileNetV2ResBlock_TF.build_from_config(self.get_active_block_config(cin))


class DynamicMobileNetV3ResBlock(_DynamicQInvertedResidual):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], expand_list: List[float], 
                    stride: int, activation_layer=nn.ReLU, inplace=True) -> None:
        super().__init__(max_cin, width_list, kernel_list, expand_list, stride, True, activation_layer, inplace, force_residual=True)

    def get_active_block(self, cin: int, retain_weights=True) -> MobileNetV3ResBlock:
        block = MobileNetV3ResBlock.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            hidden_size = self.get_active_hidden_size(cin)
            if hasattr(block, 'point_conv1'):
                block.point_conv1.load_state_dict(self.point_conv1.active_state_dict(cin, cout=hidden_size))
            block.depth_conv.load_state_dict(self.depth_conv.active_state_dict(hidden_size))
            block.point_conv2.load_state_dict(self.point_conv2.active_state_dict(hidden_size))
            block.se.load_state_dict(self.se.active_state_dict(hidden_size))
            if hasattr(block, 'downsample'):
                block.downsample.load_state_dict(self.downsample.active_state_dict(cin))
        return block
    
    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import MobileNetV3ResBlock as MobileNetV3ResBlock_TF
        return MobileNetV3ResBlock_TF.build_from_config(self.get_active_block_config(cin))


class DynamicMobileNetV2K3ResBlock(_DynamicQInvertedResidual):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], expand_list: List[float],
                    stride: int, activation_layer=nn.ReLU, inplace=True) -> None:
        super().__init__(max_cin, width_list, kernel_list, expand_list, stride, False, activation_layer, inplace, force_residual=True, DWConvNA=DynamicQDWConvK3NormActivation)

    def get_active_block(self, cin: int, retain_weights=True) -> MobileNetV2K3ResBlock:
        block = MobileNetV2K3ResBlock.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            # hidden_size is determined by cin, so it must be calculated on the run. point_conv1.active_width has no use.
            hidden_size = self.get_active_hidden_size(cin)
            if hasattr(block, 'point_conv1'):
                block.point_conv1.load_state_dict(self.point_conv1.active_state_dict(cin, cout=hidden_size))
            block.depth_conv.load_state_dict(self.depth_conv.active_state_dict(hidden_size))
            block.point_conv2.load_state_dict(self.point_conv2.active_state_dict(hidden_size))
            if hasattr(block, 'downsample'):
                block.downsample.load_state_dict(self.downsample.active_state_dict(cin))
        return block

    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import MobileNetV2K3ResBlock as MobileNetV2K3ResBlock_TF
        return MobileNetV2K3ResBlock_TF.build_from_config(self.get_active_block_config(cin))


class DynamicMobileNetV3K3ResBlock(_DynamicQInvertedResidual):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], expand_list: List[float], 
                    stride: int, activation_layer=nn.ReLU, inplace=True) -> None:
        super().__init__(max_cin, width_list, kernel_list, expand_list, stride, True, activation_layer, inplace, force_residual=True, DWConvNA=DynamicQDWConvK3NormActivation)

    def get_active_block(self, cin: int, retain_weights=True) -> MobileNetV3K3ResBlock:
        block = MobileNetV3K3ResBlock.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            hidden_size = self.get_active_hidden_size(cin)
            if hasattr(block, 'point_conv1'):
                block.point_conv1.load_state_dict(self.point_conv1.active_state_dict(cin, cout=hidden_size))
            block.depth_conv.load_state_dict(self.depth_conv.active_state_dict(hidden_size))
            block.point_conv2.load_state_dict(self.point_conv2.active_state_dict(hidden_size))
            block.se.load_state_dict(self.se.active_state_dict(hidden_size))
            if hasattr(block, 'downsample'):
                block.downsample.load_state_dict(self.downsample.active_state_dict(cin))
        return block
    
    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import MobileNetV3K3ResBlock as MobileNetV3K3ResBlock_TF
        return MobileNetV3K3ResBlock_TF.build_from_config(self.get_active_block_config(cin))
