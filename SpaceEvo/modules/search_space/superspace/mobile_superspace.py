from functools import partial

from torch import nn

from modules.search_space.macro_arch import MobileMacroArchOriginal, MobileMacroArchV1, MobileMacroArchW
from modules.modeling.dynamic_blocks import (
    DynamicMobileNetV1Block, DynamicMobileNetV2Block, DynamicMobileNetV3Block, DynamicMobileNetV2ResBlock, DynamicMobileNetV3ResBlock, DynamicMobileNetV1DualBlock, DynamicMobileNetV2K3ResBlock, DynamicMobileNetV3K3ResBlock,
    DynamicResNetBlock, DynamicResNetSEBlock, DynamicResNetBugBlock,
    DynamicFirstConvBlock, DynamicFinalExpandBlock, DynamicFeatureMixBlock, DynamicLogitsBlock,
    DynamicFusedMBConvResBlock, DynamicFusedMBConvSEResBlock
)
from .superspace import SuperSpace


class MobileSuperSpaceOriginal(SuperSpace):

    NAME = 'mobile_original'

    def __init__(self):
        macro_arch = MobileMacroArchOriginal()
        building_block_candidates = [
            partial(DynamicMobileNetV1Block, kernel_list=[3, 5, 7]), 
            partial(DynamicMobileNetV2Block, kernel_list=[3, 5, 7], expand_list=[3, 6, 8], activation_layer='relu6'), 
            partial(DynamicMobileNetV3Block, kernel_list=[3, 5, 7], expand_list=[3, 6, 8], activation_layer='hswish'), 
            partial(DynamicResNetBlock, kernel_list=[3, 5], expand_list=[0.5, 1.0, 1.5]),
            partial(DynamicResNetSEBlock, kernel_list=[3, 5], expand_list=[0.5, 1.0, 1.5]),
        ]
        other_block_dict = {
            'first_conv': partial(DynamicFirstConvBlock, kernel_list=[3]),
            'stage0': partial(DynamicMobileNetV2Block, kernel_list=[3], expand_list=[1], activation_layer='relu'),
            'final_expand': partial(DynamicFinalExpandBlock, kernel_list=[1], activation='hswish'),
            'feature_mix': partial(DynamicFeatureMixBlock, kernel_list=[1], activation='hswish'),
            'logits': DynamicLogitsBlock
        }
        super().__init__(macro_arch=macro_arch, building_block_candidates=building_block_candidates, other_block_dict=other_block_dict)


class MobileSuperSpaceV1(MobileSuperSpaceOriginal): # finatune depth

    NAME = 'mobile_v1'

    def __init__(self):
        super().__init__()
        self.macro_arch = MobileMacroArchV1()


class MobileSuperSpaceV1Res(MobileSuperSpaceV1): # force mobilenetv2 and mobilenetv3 block to use residual connection as resnet block

    NAME = 'mobile_v1_res'
    
    def __init__(self):
        super().__init__()
        self.building_block_candidates = [
            partial(DynamicMobileNetV1Block, kernel_list=[3, 5, 7]), 
            partial(DynamicMobileNetV2ResBlock, kernel_list=[3, 5, 7], expand_list=[3, 6, 8], activation_layer='relu6'), 
            partial(DynamicMobileNetV3ResBlock, kernel_list=[3, 5, 7], expand_list=[3, 6, 8], activation_layer='hswish'), 
            partial(DynamicResNetBlock, kernel_list=[3, 5], expand_list=[0.5, 1.0, 1.5]),
            partial(DynamicResNetSEBlock, kernel_list=[3, 5], expand_list=[0.5, 1.0, 1.5]),
            partial(DynamicMobileNetV1DualBlock, kernel_list=[3, 5, 7]), 
            partial(DynamicResNetBugBlock, kernel_list=[3, 5], expand_list=[0.5, 1.0, 1.5]),
        ]


class MobileSuperSpaceW(SuperSpace):

    NAME = 'mobilew'

    def __init__(self):
        macro_arch = MobileMacroArchW()
        building_block_candidates = [
            partial(DynamicMobileNetV1DualBlock, kernel_list=[3, 5, 7]), 
            partial(DynamicMobileNetV2ResBlock, kernel_list=[3, 5, 7], expand_list=[3, 6, 8], activation_layer='relu6'), 
            partial(DynamicMobileNetV3ResBlock, kernel_list=[3, 5, 7], expand_list=[3, 6, 8], activation_layer='swish'), 
            partial(DynamicResNetBlock, kernel_list=[3, 5], expand_list=[0.5, 1.0, 1.5]),
            partial(DynamicFusedMBConvSEResBlock, kernel_list=[3, 5, 7], expand_list=[1, 2, 3, 4], activation='swish'),
            partial(DynamicMobileNetV3K3ResBlock, kernel_list=[3, 5, 7], expand_list=[3, 6, 8], activation_layer='swish'),
            partial(DynamicMobileNetV2K3ResBlock, kernel_list=[3, 5, 7], expand_list=[3, 6, 8], activation_layer='relu6'),
        ]
        other_block_dict = {
            'first_conv': partial(DynamicFirstConvBlock, kernel_list=[3]),
            'stage0': partial(DynamicMobileNetV2Block, kernel_list=[3], expand_list=[1], activation_layer='relu'),
            'final_expand': partial(DynamicFinalExpandBlock, kernel_list=[1], activation='swish'),
            'feature_mix': partial(DynamicFeatureMixBlock, kernel_list=[1], activation='swish'),
            'logits': DynamicLogitsBlock
        }
        super().__init__(macro_arch=macro_arch, building_block_candidates=building_block_candidates, other_block_dict=other_block_dict)
