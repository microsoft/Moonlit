'''
Implementation of LSQ+ OPs
'''

from collections import OrderedDict
from typing import Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from modules.modeling.common.utils import make_divisible
from .lsq_plus import QBase


class QConv2d(QBase, nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, 
                padding=0, dilation=1, groups: int = 1, bias: bool = True) -> None:
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        QBase.__init__(self)

    def forward(self, x: Tensor) -> Tensor:
        xq, wq = self.get_quant_x_w(x, self.weight)
        y = F.conv2d(xq, wq, self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)
        return y


class QDWConv2d(QConv2d):

    def __init__(self, channels: int, kernel_size: int, stride=1, padding=0, dilation=1, bias: bool = True) -> None:
        super().__init__(channels, channels, kernel_size, stride, padding, dilation, channels, bias)


class QDWConv2dK3(nn.Sequential):

    def __init__(self, channels: int, kernel_size: int, stride=1, padding=0, dilation=1, bias: bool=True) -> None:
        layers = [QDWConv2d(channels=channels, kernel_size=3, stride=stride, padding=1, dilation=dilation, bias=bias)]
        stacks = {3: 1, 5: 2, 7: 3}[kernel_size]
        for _ in range(stacks - 1):
            layers.append(QDWConv2d(channels=channels, kernel_size=3, stride=1, padding=1, dilation=dilation, bias=bias))
        super().__init__(*layers)


class QLinear(QBase, nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        nn.Linear.__init__(self, in_features, out_features, bias)
        QBase.__init__(self)

    def forward(self, x: Tensor) -> Tensor:
        xq, wq = self.get_quant_x_w(x, self.weight)
        return F.linear(xq, wq, self.bias)


class QConvNormActivation(torch.nn.Sequential):
  
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [('conv', QConv2d(in_channels, out_channels, kernel_size, stride, padding,
                                  dilation=dilation, groups=groups, bias=norm_layer is None))]
        if norm_layer is not None:
            layers.append(('norm', norm_layer(out_channels)))
        if activation_layer is not None:
            layers.append(('act', activation_layer(inplace=inplace)))
       
        super().__init__(OrderedDict(layers))
        self.out_channels = out_channels


class _QDWConvNormActivation(torch.nn.Sequential):
  
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
        DWConv=QDWConv2d
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [('dwconv', DWConv(channels, kernel_size, stride, padding,
                                  dilation=dilation, bias=norm_layer is None))]
        if norm_layer is not None:
            layers.append(('norm', norm_layer(channels)))
        if activation_layer is not None:
            layers.append(('act', activation_layer(inplace=inplace)))
       
        super().__init__(OrderedDict(layers))
        self.out_channels = channels


class QDWConvNormActivation(_QDWConvNormActivation):

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, padding: Optional[int] = None, norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d, activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU, dilation: int = 1, inplace: bool = True) -> None:
        super().__init__(channels, kernel_size, stride, padding, norm_layer, activation_layer, dilation, inplace, DWConv=QDWConv2d)


class QDWConvK3NormActivation(_QDWConvNormActivation):

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, padding: Optional[int] = None, norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d, activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU, dilation: int = 1, inplace: bool = True) -> None:
        super().__init__(channels, kernel_size, stride, padding, norm_layer, activation_layer, dilation, inplace, DWConv=QDWConv2dK3)


class QSE(nn.Module):

    REDUCTION = 4

    def __init__(self, channel, reduction=None):
        super().__init__()

        self.channel = channel
        self.reduction = QSE.REDUCTION if reduction is None else reduction

        num_mid = make_divisible(self.channel // self.reduction)

        self.fc = nn.Sequential(OrderedDict([
            ('reduce', QConv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('expand', QConv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
            ('hsigmoid', nn.Hardsigmoid(inplace=True)),
        ]))

    def _scale(self, x: Tensor) -> Tensor:
        y = x.mean([2, 3], keepdim=True)
        y = self.fc(y)
        return y

    def forward(self, x: Tensor) -> Tensor:
        scale = self._scale(x)
        return x * scale


class QSE4Resnet(nn.Module):

    '''se ratio in regnet is defined with respect to the input width of this block instead of hidden width'''
    REDUCTION = 4

    def __init__(self, in_channels, mid_channels) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels

        self.fc = nn.Sequential(OrderedDict([
            ('reduce', QConv2d(in_channels, mid_channels, 1, 1, 0, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('expand', QConv2d(mid_channels, in_channels, 1, 1, 0, bias=True)),
            ('hsigmoid', nn.Hardsigmoid(inplace=True)),
        ]))

    def _scale(self, x: Tensor) -> Tensor:
        y = x.mean([2, 3], keepdim=True)
        y = self.fc(y)
        return y

    def forward(self, x: Tensor) -> Tensor:
        scale = self._scale(x)
        return x * scale

    @staticmethod
    def get_mid_channels(cin: int) -> int:
        return make_divisible(cin // QSE4Resnet.REDUCTION)