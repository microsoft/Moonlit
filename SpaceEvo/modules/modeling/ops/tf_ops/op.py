'''
Implementation of LSQ+ OPs
'''

from collections import OrderedDict
from typing import Callable, Optional

import tensorflow as tf
from tensorflow.keras import layers

from modules.modeling.common.utils import make_divisible
from .base_ops import HSigmoid


class QConvNormActivation(tf.keras.Model):
  
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., layers.Layer]] = layers.BatchNormalization,
        activation_layer: Optional[Callable[..., layers.Layer]] = layers.ReLU,
        dilation: int = 1,
        inplace: bool = True,
    ) -> None:
        super().__init__()
        self.conv = layers.Conv2D(filters=out_channels, kernel_size=int(kernel_size), strides=stride, padding='same', use_bias=False)
        if norm_layer is not None:
            self.bn = norm_layer()
        else:
            self.bn = None
        if activation_layer is not None:
            self.activation = activation_layer()
        else:
            self.activation = None
        self.out_channels = out_channels
    
    def call(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DWConvK3(tf.keras.Sequential):

    def __init__(self, kernel_size, strides, padding='same', use_bias=True):
        _layers = []
        stacks = {3: 1, 5: 2, 7: 3}[kernel_size]
        for i in range(stacks):
            s = strides if i == 0 else 1
            _layers.append(layers.DepthwiseConv2D(kernel_size=3, strides=s, padding=padding, use_bias=use_bias))
        super().__init__(layers=_layers)


class _QDWConvNormActivation(tf.keras.Model):
  
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        norm_layer: Optional[Callable[..., layers.Layer]] = layers.BatchNormalization,
        activation_layer: Optional[Callable[..., layers.Layer]] = layers.ReLU,
        dilation: int = 1,
        inplace: bool = True,
        DWConv = layers.DepthwiseConv2D
    ) -> None:
        super().__init__()
        self.dwconv = DWConv(kernel_size=int(kernel_size), strides=stride, padding='same', use_bias=False)
        self.bn = norm_layer()
        self.activation = activation_layer()
        self.out_channels = channels
    
    def call(self, x):
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class QDWConvNormActivation(_QDWConvNormActivation):

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, padding: Optional[int] = None, norm_layer: Optional[Callable[..., layers.Layer]] = layers.BatchNormalization, activation_layer: Optional[Callable[..., layers.Layer]] = layers.ReLU, dilation: int = 1, inplace: bool = True) -> None:
        super().__init__(channels, kernel_size, stride, padding, norm_layer, activation_layer, dilation, inplace, DWConv=layers.DepthwiseConv2D)


class QDWConvK3NormActivation(_QDWConvNormActivation):

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, padding: Optional[int] = None, norm_layer: Optional[Callable[..., layers.Layer]] = layers.BatchNormalization, activation_layer: Optional[Callable[..., layers.Layer]] = layers.ReLU, dilation: int = 1, inplace: bool = True) -> None:
        super().__init__(channels, kernel_size, stride, padding, norm_layer, activation_layer, dilation, inplace, DWConv=DWConvK3)


class HSwish(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.relu6 = layers.ReLU(6)

    def call(self, x):
        return x * self.relu6(x + 3.) * (1. / 6.)

class Relu6(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.relu6 = layers.ReLU(6)

    def call(self, x):
        return self.relu6(x)


class QSE(tf.keras.Model):

    REDUCTION = 4

    def __init__(self, channel, reduction=None):
        super().__init__()
        self.channel = channel
        self.reduction = QSE.REDUCTION if reduction is None else reduction
        
        self.pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.squeeze = layers.Conv2D(filters=make_divisible(self.channel // self.reduction), kernel_size=1, padding='same')
        self.relu = layers.ReLU()
        self.excite = layers.Conv2D(filters=self.channel, kernel_size=1, padding='same')
        self.hsigmoid = HSigmoid()

    def call(self, x):
        x0 = x
        x = self.pool(x)
        x = self.squeeze(x)
        x = self.relu(x)
        x = self.excite(x)
        x = self.hsigmoid(x)
        return x * x0


class QSE4Resnet(tf.keras.Model):

    '''se ratio in regnet is defined with respect to the input width of this block instead of hidden width'''
    REDUCTION = 4

    def __init__(self, in_channels, mid_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        self.pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.squeeze = layers.Conv2D(filters=self.mid_channels, kernel_size=1, padding='same')
        self.relu = layers.ReLU()
        self.excite = layers.Conv2D(filters=self.in_channels, kernel_size=1, padding='same')
        self.hsigmoid = HSigmoid()

    def call(self, x):
        x0 = x
        x = self.pool(x)
        x = self.squeeze(x)
        x = self.relu(x)
        x = self.excite(x)
        x = self.hsigmoid(x)
        return x * x0

    @staticmethod
    def get_mid_channels(cin: int) -> int:
        return make_divisible(cin // QSE4Resnet.REDUCTION)