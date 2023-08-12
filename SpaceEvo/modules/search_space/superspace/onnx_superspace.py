from functools import partial

from torch import nn

from modules.search_space.macro_arch import OnnxMacroArch, OnnxMacroArchW
from modules.modeling.dynamic_blocks import (
    DynamicMobileNetV1DualBlock, DynamicMobileNetV2ResBlock, DynamicMobileNetV3ResBlock,
    DynamicResNetBlock, DynamicResNetSEBlock, 
    DynamicFirstConvBlock, DynamicFinalExpandBlock, DynamicFeatureMixBlock, DynamicLogitsBlock,
    DynamicFusedMBConvResBlock
)
from .superspace import SuperSpace


class OnnxSuperSpace(SuperSpace):

    NAME = 'onnx'

    def __init__(self):
        macro_arch = OnnxMacroArch()
        building_block_candidates = [
            partial(DynamicMobileNetV1DualBlock, kernel_list=[3, 5]),
            partial(DynamicMobileNetV2ResBlock, kernel_list=[3, 5], expand_list=[4, 6, 8], activation_layer='relu6'),
            partial(DynamicMobileNetV3ResBlock, kernel_list=[3, 5], expand_list=[4, 6, 8], activation_layer='hswish'),
            partial(DynamicResNetBlock, kernel_list=[3, 5, 7], expand_list=[0.5, 1.0, 1.5]),
            partial(DynamicResNetSEBlock, kernel_list=[3, 5, 7], expand_list=[0.5, 1.0, 1.5]),
        ]
        other_block_dict = {
            'first_conv': partial(DynamicFirstConvBlock, kernel_list=[3]),
            'stage0': partial(DynamicResNetBlock, kernel_list=[3], expand_list=[0.5], activation='relu'),
            'final_expand': partial(DynamicFinalExpandBlock, kernel_list=[1], activation='hswish'),
            'feature_mix': partial(DynamicFeatureMixBlock, kernel_list=[1], activation='hswish'),
            'logits': DynamicLogitsBlock
        }
        super().__init__(macro_arch=macro_arch, building_block_candidates=building_block_candidates, other_block_dict=other_block_dict)


class OnnxSuperSpaceV1(SuperSpace):

    NAME = 'onnx_v1'

    def __init__(self):
        macro_arch = OnnxMacroArch()
        building_block_candidates = [
            partial(DynamicMobileNetV1DualBlock, kernel_list=[3, 5]),
            partial(DynamicMobileNetV2ResBlock, kernel_list=[3, 5], expand_list=[4, 6, 8], activation_layer='relu6'),
            partial(DynamicMobileNetV3ResBlock, kernel_list=[3, 5], expand_list=[4, 6, 8], activation_layer='hswish'),
            partial(DynamicResNetBlock, kernel_list=[3, 5], expand_list=[0.5, 1.0, 1.5]),
            partial(DynamicResNetSEBlock, kernel_list=[3, 5], expand_list=[0.5, 1.0, 1.5]),
        ]
        other_block_dict = {
            'first_conv': partial(DynamicFirstConvBlock, kernel_list=[3]),
            'stage0': partial(DynamicResNetBlock, kernel_list=[3], expand_list=[0.5], activation='relu'),
            'final_expand': partial(DynamicFinalExpandBlock, kernel_list=[1], activation='hswish'),
            'feature_mix': partial(DynamicFeatureMixBlock, kernel_list=[1], activation='hswish'),
            'logits': DynamicLogitsBlock
        }
        super().__init__(macro_arch=macro_arch, building_block_candidates=building_block_candidates, other_block_dict=other_block_dict)


class OnnxSuperSpaceW(SuperSpace):
    
    NAME = 'onnxw'
    
    def __init__(self):
        macro_arch = OnnxMacroArchW()
        building_block_candidates = [
            partial(DynamicMobileNetV1DualBlock, kernel_list=[3, 5]),
            partial(DynamicMobileNetV2ResBlock, kernel_list=[3, 5], expand_list=[4, 6, 8], activation_layer='relu6'),
            partial(DynamicMobileNetV3ResBlock, kernel_list=[3, 5], expand_list=[4, 6, 8], activation_layer='hswish'),
            partial(DynamicResNetBlock, kernel_list=[3, 5, 7], expand_list=[0.5, 1.0, 1.5]),
            partial(DynamicResNetSEBlock, kernel_list=[3, 5, 7], expand_list=[0.5, 1.0, 1.5]),
            partial(DynamicFusedMBConvResBlock, kernel_list=[3, 5, 7], expand_list=[1, 2, 3, 4], activation='swish'),
        ]
        other_block_dict = {
            'first_conv': partial(DynamicFirstConvBlock, kernel_list=[3]),
            'stage0': partial(DynamicResNetBlock, kernel_list=[3], expand_list=[0.5], activation='relu'),
            'final_expand': partial(DynamicFinalExpandBlock, kernel_list=[1], activation='hswish'),
            'feature_mix': partial(DynamicFeatureMixBlock, kernel_list=[1], activation='hswish'),
            'logits': DynamicLogitsBlock
        }
        super().__init__(macro_arch=macro_arch, building_block_candidates=building_block_candidates, other_block_dict=other_block_dict)
