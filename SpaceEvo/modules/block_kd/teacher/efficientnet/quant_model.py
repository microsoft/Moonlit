import copy
import math
from re import M

import torch
from torch import nn
import torch.nn.functional as F

from modules.modeling.ops.lsq_plus import QBase, set_quant_mode
from modules.modeling.ops.op import QConv2d, QDWConv2d, QLinear
from .model import EfficientNet
from .utils import Conv2dStaticSamePadding, load_pretrained_weights


class QConv2dStaticSamePadding(QConv2d):

    def __init__(self, static_padding, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1, groups: int = 1, bias: bool = True) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.static_padding = static_padding

    def forward(self, x):
        x = self.static_padding(x)
        x = super().forward(x)
        return x


class QEfficientNet(EfficientNet):

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)
        replace_quant_op(self)
        set_quant_mode(self)

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        model = EfficientNet.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path,
                                load_fc=(num_classes == 1000), advprop=advprop)
        model._change_in_channels(in_channels)
        
        replace_quant_op(model)
        set_quant_mode(model)
        return model


def replace_quant_op(model):
    def replace_op(m: nn.Module):
        for name, module in m.named_children():
            if len(list(module.children())) > 0:
                replace_op(module)

            if isinstance(module, Conv2dStaticSamePadding):
                new_module = QConv2dStaticSamePadding(
                    copy.deepcopy(module.static_padding), 
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None
                )
                new_module.load_state_dict(module.state_dict(), strict=False)
                setattr(m, name, new_module)
                
            if isinstance(module, nn.Linear):
                new_module = QLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None
                )
                new_module.load_state_dict(module.state_dict(), strict=False)
                setattr(m, name, new_module)
    
    replace_op(model)
