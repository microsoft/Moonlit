from typing import Dict, List

from torch import Tensor, nn
import torch

from modules.modeling.dynamic_blocks.base_dynamic_block import BaseDynamicBlock
from modules.modeling.ops import (
    DynamicQConvNormActivation, DynamicQSE4ResNet
)
from modules.modeling.blocks import (
    FusedMBConvResBlock, FusedMBConvSEResBlock
)
from modules.modeling.common.utils import make_divisible, get_activation


class _DynamicFusedMBConvBlock(BaseDynamicBlock):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], stride: int, expand_list: List[float], use_se: bool, activation=nn.ReLU, inplace=True, force_residual=False) -> None:
        super().__init__(max_cin, width_list, kernel_list, stride, expand_list, activation)
        self.use_se = use_se
        self.force_residual = force_residual

        self.fused_conv = DynamicQConvNormActivation(max_in_channels=self.max_cin, max_out_channels=self.max_hidden_size, kernel_size_list=kernel_list, stride=stride, activation_layer=self.activation_layer, inplace=inplace)

        if self.use_se:
            self.se = DynamicQSE4ResNet(max_cin=self.max_hidden_size, max_mid=DynamicQSE4ResNet.get_mid_channels(self.max_cin))

        self.point_conv = DynamicQConvNormActivation(max_in_channels=self.max_hidden_size, max_out_channels=self.max_width, kernel_size_list=[1], stride=1, activation_layer=None)

        self.force_residual = force_residual
        if self.force_residual:
            self.downsample = DynamicQConvNormActivation(
                max_in_channels=self.max_cin, max_out_channels=self.max_width,
                kernel_size_list=[1], stride=stride, activation_layer=None
            )

    def forward(self, x: Tensor) -> Tensor:
        cin =  x.shape[1]
        if self.use_res_connect(cin):
            x0 = x 
        elif self.force_residual:
            x0 = self.downsample(x)
        
        active_hidden_size = self.get_active_hidden_size(cin)

        x = self.fused_conv.forward(x, out_channels=active_hidden_size)

        if self.use_se:
            self.se.active_mid_channels = DynamicQSE4ResNet.get_mid_channels(cin)
            x = self.se(x)
        x = self.point_conv(x)

        if self.use_res_connect(cin) or self.force_residual:
            x = x0 + x 
        return x

    def use_res_connect(self, cin):
        return cin == self.active_width and self.stride == 1

    @property
    def max_hidden_size(self):
        return make_divisible(self.max_expand_ratio * self.max_cin)

    def get_active_hidden_size(self, cin: int) -> int:
        return make_divisible(self.active_expand_ratio * cin)

    @property
    def active_kernel_size(self) -> int:
        return self.fused_conv.active_kernel_size

    @active_kernel_size.setter 
    def active_kernel_size(self, kernel_size: int):
        self.fused_conv.active_kernel_size = kernel_size

    @property
    def active_width(self) -> int:
        return self.point_conv.active_out_channels

    @active_width.setter
    def active_width(self, width: int):
        self.point_conv.active_out_channels = width 
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
            self.point_conv.norm.weight.data.zero_()
        


class DynamicFusedMBConvResBlock(_DynamicFusedMBConvBlock):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], stride: int, expand_list: List[float], activation=nn.ReLU, inplace=True) -> None:
        super().__init__(max_cin, width_list, kernel_list, stride, expand_list, use_se=False, activation=activation, inplace=inplace, force_residual=True)

    def get_active_block(self, cin: int, retain_weights=True) -> FusedMBConvResBlock:
        block = FusedMBConvResBlock.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            hidden_size = self.get_active_hidden_size(cin)
            block.fused_conv.load_state_dict(self.fused_conv.active_state_dict(cin, cout=hidden_size))
            block.point_conv.load_state_dict(self.point_conv.active_state_dict(hidden_size))
            if hasattr(block, 'downsample'):
                block.downsample.load_state_dict(self.downsample.active_state_dict(cin))
        return block

    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import FusedMBConvResBlock as FusedMBConvResBlock_TF
        return FusedMBConvResBlock_TF.build_from_config(self.get_active_block_config(cin))


class DynamicFusedMBConvSEResBlock(_DynamicFusedMBConvBlock):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], stride: int, expand_list: List[float], activation=nn.ReLU, inplace=True) -> None:
        super().__init__(max_cin, width_list, kernel_list, stride, expand_list, use_se=True, activation=activation, inplace=inplace, force_residual=True)

    def get_active_block(self, cin: int, retain_weights=True) -> FusedMBConvSEResBlock:
        block = FusedMBConvSEResBlock.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            hidden_size = self.get_active_hidden_size(cin)
            block.fused_conv.load_state_dict(self.fused_conv.active_state_dict(cin, cout=hidden_size))
            self.se.active_mid_channels = self.se.get_mid_channels(cin)
            block.se.load_state_dict(self.se.active_state_dict(hidden_size))
            block.point_conv.load_state_dict(self.point_conv.active_state_dict(hidden_size))
            if hasattr(block, 'downsample'):
                block.downsample.load_state_dict(self.downsample.active_state_dict(cin))
        return block

    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import FusedMBConvSEResBlock as FusedMBConvSEResBlock_TF
        return FusedMBConvSEResBlock_TF.build_from_config(self.get_active_block_config(cin))

