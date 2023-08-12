from typing import Dict, List

from torch import Tensor, nn

from modules.modeling.dynamic_blocks.base_dynamic_block import BaseDynamicBlock
from modules.modeling.ops import (
    DynamicBatchNorm2d, DynamicQConv2d, DynamicQDWConv2d, 
    DynamicQConvNormActivation, DynamicQDWConvNormActivation, 
    DynamicQLinear, DynamicQSE
)
from modules.modeling.blocks import FirstConvBlock, FinalExpandBlock, FeatureMixBlock, LogitsBlock
from modules.modeling.common.utils import make_divisible, get_activation


class _DynamicConvNormActBlock(BaseDynamicBlock):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], stride: int, activation='relu') -> None:
        super().__init__(max_cin, width_list, kernel_list, stride, None, activation)

        self.conv = DynamicQConvNormActivation(
            max_in_channels=max_cin, max_out_channels=self.max_width, kernel_size_list=self.kernel_list, stride=self.stride, activation_layer=self.activation_layer
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x 

    @property
    def active_kernel_size(self) -> int:
        return self.conv.active_kernel_size

    @active_kernel_size.setter
    def active_kernel_size(self, kernel_size: int):
        self.conv.active_kernel_size = kernel_size

    @property
    def active_width(self) -> int:
        return self.conv.active_out_channels

    @active_width.setter
    def active_width(self, width: int):
        self.conv.active_out_channels = width

    @property
    def active_expand_ratio(self) -> float:
        return self._active_expand_ratio

    def get_active_block(self, cin: int, retain_weights=True):
        raise NotImplementedError()


class DynamicFirstConvBlock(_DynamicConvNormActBlock):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], stride: int, activation='relu') -> None:
        super().__init__(max_cin, width_list, kernel_list, stride, activation)

    def get_active_block(self, cin: int, retain_weights=True) -> FirstConvBlock:
        block = FirstConvBlock.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            block.conv.load_state_dict(self.conv.active_state_dict(cin))
        return block

    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import FirstConvBlock as FirstConvBlock_TF
        return FirstConvBlock_TF.build_from_config(self.get_active_block_config(cin))


class DynamicFinalExpandBlock(_DynamicConvNormActBlock):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], stride: int, activation='relu') -> None:
        super().__init__(max_cin, width_list, kernel_list, stride, activation)
    
    def get_active_block(self, cin: int, retain_weights=True) -> FinalExpandBlock:
        block = FinalExpandBlock.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            block.conv.load_state_dict(self.conv.active_state_dict(cin))
        return block
    
    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import FinalExpandBlock as FinalExpandBlock_TF
        return FinalExpandBlock_TF.build_from_config(self.get_active_block_config(cin))


class DynamicFeatureMixBlock(_DynamicConvNormActBlock):

    def __init__(self, max_cin: int, width_list: List[int], kernel_list: List[int], stride: int, activation='relu') -> None:
        super().__init__(max_cin, width_list, kernel_list, stride, activation)

    def forward(self, x: Tensor) -> Tensor:
        x = x.mean([2, 3], keepdim=True)
        x = self.conv(x)
        return x 

    def get_active_block(self, cin: int, retain_weights=True) -> FeatureMixBlock:
        block = FeatureMixBlock.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            block.conv.load_state_dict(self.conv.active_state_dict(cin))
        return block

    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import FeatureMixBlock as FeatureMixBlock_TF
        return FeatureMixBlock_TF.build_from_config(self.get_active_block_config(cin))


class DynamicLogitsBlock(BaseDynamicBlock):

    def __init__(self, max_cin: int, width_list: List[int], stride: int) -> None:
        assert len(width_list) == 1, 'the width of logis block must be equal to num_classes'
        super().__init__(max_cin=max_cin, width_list=width_list, kernel_list=[0], stride=stride, expand_list=None, activation='none')
        self.linear = DynamicQLinear(max_in_features=max_cin, max_out_features=self.max_width)
        self.dropout = nn.Dropout(inplace=True) # set dropout rate in supernet

    def forward(self, x: Tensor) -> Tensor:
        x = x.squeeze()
        x = self.dropout(x)
        x = self.linear(x)
        return x 

    @property
    def active_kernel_size(self) -> int:
        return 0

    @property
    def active_width(self) -> int:
        return self.max_width

    @property
    def active_expand_ratio(self) -> float:
        return self._active_expand_ratio

    def get_active_block(self, cin: int, retain_weights=True) -> LogitsBlock:
        block = LogitsBlock.build_from_config(self.get_active_block_config(cin))
        if retain_weights:
            state_dict = {}
            self.linear.set_active_state_dict('', state_dict, cin, self.active_width)
            block.linear.load_state_dict(state_dict)
        return block
    
    def get_active_tf_block(self, cin: int):
        from modules.modeling.blocks.tf_blocks import LogitsBlock as LogitsBlock_TF
        return LogitsBlock_TF.build_from_config(self.get_active_block_config(cin))

