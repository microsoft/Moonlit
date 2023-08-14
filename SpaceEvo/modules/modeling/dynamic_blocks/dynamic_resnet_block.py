from typing import Dict, List

import torch
from torch import Tensor, nn

from modules.modeling.dynamic_blocks.base_dynamic_block import BaseDynamicBlock
from modules.modeling.ops import DynamicQConvNormActivation, DynamicQSE4ResNet
from modules.modeling.blocks import ResNetBlock, ResNetSEBlock, ResNetBugBlock
from modules.modeling.common.utils import make_divisible


class _DynamicResNetBaseBlock(BaseDynamicBlock):

    def __init__(self, max_cin: int, width_list: List[int], stride: int, kernel_list: List[int],
                 expand_list: List[float],
                 activation=nn.ReLU, inplace: bool = True, use_se=False) -> None:
        super().__init__(max_cin, width_list, kernel_list, stride, expand_list, activation)

        self.conv1 = DynamicQConvNormActivation(
            max_in_channels=self.max_cin, max_out_channels=self.max_hidden_size, kernel_size_list=[1],
            stride=1, activation_layer=self.activation_layer, inplace=inplace
        )
        self.conv2 = DynamicQConvNormActivation(
            max_in_channels=self.max_hidden_size, max_out_channels=self.max_hidden_size,
            kernel_size_list=self.kernel_list, stride=self.stride, activation_layer=self.activation_layer,
            inplace=inplace
        )

        if use_se:
            self.se = DynamicQSE4ResNet(self.max_hidden_size, DynamicQSE4ResNet.get_mid_channels(max_cin))

        self.conv3 = DynamicQConvNormActivation(
            max_in_channels=self.max_hidden_size, max_out_channels=self.max_width,
            kernel_size_list=[1], stride=1, activation_layer=None
        )
        self.downsample = DynamicQConvNormActivation(
            max_in_channels=self.max_cin, max_out_channels=self.max_width,
            kernel_size_list=[1], stride=stride, activation_layer=None
        )

        self.activation = self.activation_layer(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        cin = x.shape[1]
        if self.stride > 1 or cin != self.active_width:
            identity = self.downsample(x)
        else:
            identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        if hasattr(self, 'se'):
            self.se.active_mid_channels = self.se.get_mid_channels(cin)
            x = self.se(x)
        x = self.conv3(x)

        x = identity + x
        x = self.activation(x)
        return x

    @property
    def max_hidden_size(self):
        return make_divisible(self.max_expand_ratio * self.max_width)

    @property
    def active_hidden_size(self) -> int:
        return make_divisible(self.active_expand_ratio * self.active_width)

    @property
    def active_kernel_size(self) -> int:
        return self.conv2.active_kernel_size

    @active_kernel_size.setter
    def active_kernel_size(self, kernel_size: int):
        self.conv2.active_kernel_size = kernel_size

    @property
    def active_width(self) -> int:
        return self.conv3.active_out_channels

    @active_width.setter
    def active_width(self, width: int):
        self.conv3.active_out_channels = width
        self.conv1.active_out_channels = self.active_hidden_size
        self.conv2.active_out_channels = self.active_hidden_size
        self.downsample.active_out_channels = width

    @property
    def active_expand_ratio(self) -> float:
        return self._active_expand_ratio

    @active_expand_ratio.setter
    def active_expand_ratio(self, expand_ratio: float):
        self._active_expand_ratio = expand_ratio
        self.conv1.active_out_channels = self.active_hidden_size
        self.conv2.active_out_channels = self.active_hidden_size

    def get_active_block(self, cin: int, retain_weights=True):
        raise NotImplementedError()

    def zero_last_gamma(self):
        self.conv3.norm.weight.data.zero_()


class DynamicResNetBlock(_DynamicResNetBaseBlock):

    def __init__(self, max_cin: int, width_list: List[int], stride: int, kernel_list: List[int],
                 expand_list: List[float], activation=nn.ReLU,
                 inplace: bool = True) -> None:
        super().__init__(max_cin, width_list, stride, kernel_list, expand_list, activation, inplace, use_se=False)

    def get_active_block(self, cin: int, retain_weights=True) -> ResNetBlock:
        block = ResNetBlock.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            block.conv1.load_state_dict(self.conv1.active_state_dict(cin))
            block.conv2.load_state_dict(self.conv2.active_state_dict(self.active_hidden_size))
            block.conv3.load_state_dict(self.conv3.active_state_dict(self.active_hidden_size))
            if block.downsample:
                block.downsample.load_state_dict(self.downsample.active_state_dict(cin))
        return block

    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import ResNetBlock as ResNetBlock_TF
        return ResNetBlock_TF.build_from_config(self.get_active_block_config(cin))


class DynamicResNetSEBlock(_DynamicResNetBaseBlock):

    def __init__(self, max_cin: int, width_list: List[int], stride: int, kernel_list: List[int],
                 expand_list: List[float], activation=nn.ReLU,
                 inplace: bool = True) -> None:
        super().__init__(max_cin, width_list, stride, kernel_list, expand_list, activation, inplace, use_se=True)

    def get_active_block(self, cin: int, retain_weights=True) -> ResNetSEBlock:
        block = ResNetSEBlock.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            block.conv1.load_state_dict(self.conv1.active_state_dict(cin))
            block.conv2.load_state_dict(self.conv2.active_state_dict(self.active_hidden_size))
            block.conv3.load_state_dict(self.conv3.active_state_dict(self.active_hidden_size))
            self.se.active_mid_channels = self.se.get_mid_channels(cin)
            block.se.load_state_dict(self.se.active_state_dict(self.active_hidden_size))
            if block.downsample:
                block.downsample.load_state_dict(self.downsample.active_state_dict(cin))
        return block

    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import ResNetSEBlock as ResNetSEBlock_TF
        return ResNetSEBlock_TF.build_from_config(self.get_active_block_config(cin))


class DynamicResNetBugBlock(_DynamicResNetBaseBlock):

    def __init__(self, max_cin: int, width_list: List[int], stride: int, kernel_list: List[int],
                 expand_list: List[float], activation=nn.ReLU,
                 inplace: bool = True) -> None:
        super().__init__(max_cin, width_list, stride, kernel_list, expand_list, activation, inplace, use_se=False)

    def get_active_block(self, cin: int, retain_weights=True) -> ResNetBugBlock:
        block = ResNetBugBlock.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            if block.downsample:
                block.downsample.load_state_dict(self.downsample.active_state_dict(cin))
        return block

    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import ResNetBugBlock as ResNetBugBlock_TF
        return ResNetBugBlock_TF.build_from_config(self.get_active_block_config(cin))
