
from collections import OrderedDict

from typing import Callable, Dict, List, Optional
import torch
from torch.nn import functional as F
from torch import nn, Tensor

from modules.modeling.common.utils import get_same_padding, make_divisible, sub_filter_start_end
from modules.modeling.ops.lsq_plus import QBase
from modules.modeling.ops import QSE


class DynamicQConv2d(QBase, nn.Conv2d):

    def __init__(self, max_in_channels: int, max_out_channels: int, kernel_size_list: List[int],
                stride=1, dilation=1, groups: int = 1, bias: bool = True) -> None:
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size_list = kernel_size_list
        
        nn.Conv2d.__init__(self, max_in_channels, max_out_channels, self.max_kernel_size, stride, 0, dilation, groups, bias)
        QBase.__init__(self)

        self.active_out_channels = self.max_out_channels
        self.active_kernel_size = self.max_kernel_size

    @property
    def max_kernel_size(self):
        return max(self.kernel_size_list)

    def get_active_weight_and_bias(self, out_channels, in_channels, kernel_size):
        start, end = sub_filter_start_end(self.max_kernel_size, kernel_size)
        weight = self.weight[:out_channels, :in_channels, start:end, start:end]
        bias = self.bias[:out_channels] if self.bias is not None else None
        return weight, bias 

    def forward(self, x: Tensor, out_channels=None, kernel_size=None) -> Tensor:
        out_channels = out_channels or self.active_out_channels
        in_channels = x.shape[1]
        kernel_size = kernel_size or self.active_kernel_size

        weight, bias = self.get_active_weight_and_bias(out_channels, in_channels, kernel_size)
        weight = weight.contiguous()
        padding = get_same_padding(kernel_size)

        xq, wq = self.get_quant_x_w(x, weight)
        y = F.conv2d(xq, wq, bias, self.stride, padding, self.dilation, self.groups)
        return y

    def set_active_state_dict(self, prefix: str, rv: dict, cin: int, cout=None, kernel_size=None):
        cout = cout or self.active_out_channels
        kernel_size = kernel_size or self.active_kernel_size
        weight, bias = self.get_active_weight_and_bias(cout, cin, kernel_size)
        rv[prefix + 'weight'] = weight.data
        if bias is not None:
            rv[prefix + 'bias'] = bias.data
        self.set_lsq_state_dict(prefix, rv)


class DynamicQDWConv2d(DynamicQConv2d):

    def __init__(self, max_channels: int, kernel_size_list: List[int], stride=1, dilation=1, bias: bool = True) -> None:
        super().__init__(max_channels, max_channels, kernel_size_list, stride=stride, 
                            dilation=dilation, groups=max_channels, bias=bias)
        self.max_channels = max_channels

    def get_active_weight_and_bias(self, channels, kernel_size):
        return super().get_active_weight_and_bias(channels, channels, kernel_size)
    
    def forward(self, x: Tensor, kernel_size=None) -> Tensor:
        channels = x.shape[1]
        kernel_size = kernel_size or self.active_kernel_size

        weight, bias = self.get_active_weight_and_bias(channels, kernel_size)
        weight = weight.contiguous()
        padding = get_same_padding(kernel_size)

        xq, wq = self.get_quant_x_w(x, weight)
        y = F.conv2d(xq, wq, bias, self.stride, padding, self.dilation, channels)
        return y

    def set_active_state_dict(self, prefix: str, rv: dict, cin: int, kernel_size=None):
        kernel_size = kernel_size or self.active_kernel_size
        weight, bias = self.get_active_weight_and_bias(cin, kernel_size)
        rv[prefix + 'weight'] = weight
        if bias is not None:
            rv[prefix + 'bias'] = bias 
        self.set_lsq_state_dict(prefix, rv)


class DynamicQDWConv2dK3(nn.Module):

    def __init__(self, max_channels: int, kernel_size_list: List[int], stride=1, dilation=1, bias: bool = True) -> None:
        super().__init__()
        self.max_channels = max_channels
        self.kernel_size_list = kernel_size_list
        for i in range(self.get_active_num_stacks(self.max_kernel_size)):
            s = stride if i == 0 else 1
            dwconv = DynamicQDWConv2d(max_channels, kernel_size_list=[3], stride=s, dilation=dilation, bias=bias)
            setattr(self, f'dwconv{i}', dwconv)

        self.active_kernel_size = max(self.kernel_size_list)

    @property
    def max_kernel_size(self):
        return max(self.kernel_size_list)

    def get_active_num_stacks(self, kernel_size: int):
        return {3: 1, 5: 2, 7: 3}[kernel_size]

    def forward(self, x: Tensor, kernel_size=None) -> Tensor:
        kernel_size = kernel_size or self.active_kernel_size
        for i in range(self.get_active_num_stacks(self.active_kernel_size)):
            dwconv = getattr(self, f'dwconv{i}')
            x = dwconv(x)
        return x 

    def set_active_state_dict(self, prefix: str, rv: Dict, cin: int, kernel_size=None):
        kernel_size = kernel_size or self.active_kernel_size
        for i in range(self.get_active_num_stacks(self.active_kernel_size)):
            getattr(self, f'dwconv{i}').set_active_state_dict(prefix=prefix+f'{i}.', rv=rv, cin=cin)
        return rv 


class DynamicQLinear(QBase, nn.Linear):

    def __init__(self, max_in_features, max_out_features, bias=True):
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features

        nn.Linear.__init__(self, max_in_features, max_out_features, bias)
        QBase.__init__(self)

        self.active_out_features = self.max_out_features

    def get_active_weight(self, out_features, in_features):
        return self.weight[:out_features, :in_features]

    def get_active_bias(self, out_features):
        return None if self.bias is None else self.bias[:out_features]

    def forward(self, x: Tensor, out_features=None) -> Tensor:
        out_features = out_features or self.active_out_features
        in_features = x.shape[1]

        weight = self.get_active_weight(out_features, in_features).contiguous()
        bias = self.get_active_bias(out_features)
        
        xq, wq = self.get_quant_x_w(x, weight)
        y = F.linear(xq, wq, bias)
        return y

    def set_active_state_dict(self, prefix: str, rv: dict, in_features: int, out_features=None):
        out_features = out_features or self.active_out_features
        rv[prefix + 'weight'] = self.get_active_weight(out_features, in_features)
        if self.bias is not None:
            rv[prefix + 'bias'] = self.get_active_bias(out_features) 
        self.set_lsq_state_dict(prefix, rv)


class DynamicBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, max_num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__(max_num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self.max_num_features = max_num_features

    def forward(self, x: Tensor) -> Tensor:
        num_features = x.shape[1]

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean[:num_features]
            if not self.training or self.track_running_stats
            else None,
            self.running_var[:num_features] if not self.training or self.track_running_stats else None,
            self.weight[:num_features],
            self.bias[:num_features],
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def set_active_state_dict(self, prefix: str, rv: dict, channels: int):
        rv[prefix + 'running_mean'] = self.running_mean[:channels].data 
        rv[prefix + 'running_var'] = self.running_var[:channels].data
        rv[prefix + 'num_batches_tracked'] = self.num_batches_tracked.data
        rv[prefix + 'weight'] = self.weight[:channels].data
        rv[prefix + 'bias'] = self.bias[:channels].data 


class DynamicQSE(QSE):

    def __init__(self, channel, reduction=None):
        super().__init__(channel, reduction)

    def get_active_reduce_weight(self, num_mid, in_channel):
        return self.fc.reduce.weight[:num_mid, :in_channel, :, :]

    def get_active_reduce_bias(self, num_mid):
        return (
            self.fc.reduce.bias[:num_mid] if self.fc.reduce.bias is not None else None
        )

    def get_active_expand_weight(self, num_mid, in_channel):
        return self.fc.expand.weight[:in_channel, :num_mid, :, :]

    def get_active_expand_bias(self, in_channel):
        return (
            self.fc.expand.bias[:in_channel]
            if self.fc.expand.bias is not None
            else None
        )

    def _scale(self, x: Tensor) -> Tensor:
        in_channel = x.shape[1]
        num_mid = make_divisible(
            in_channel // self.reduction
        )

        y = x.mean([2, 3], keepdim=True)
        # reduce
        reduce_filter = self.get_active_reduce_weight(num_mid, in_channel).contiguous()
        reduce_bias = self.get_active_reduce_bias(num_mid)
        yq, wq = self.fc.reduce.get_quant_x_w(y, reduce_filter)
        y = F.conv2d(yq, wq, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.relu(y)
        # expand
        expand_filter = self.get_active_expand_weight(num_mid, in_channel).contiguous()
        expand_bias = self.get_active_expand_bias(in_channel)
        yq, wq = self.fc.expand.get_quant_x_w(y, expand_filter)
        y = F.conv2d(yq, wq, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.hsigmoid(y)
        return y

    def forward(self, x: Tensor) -> Tensor:
        scale = self._scale(x)
        return x * scale

    def active_state_dict(self, cin: int):
        mid_c = make_divisible(cin // self.reduction)
        rv = {}
        reduce_weight = self.get_active_reduce_weight(mid_c, cin).contiguous()
        reduce_bias = self.get_active_reduce_bias(mid_c)
        rv['fc.reduce.weight'] = reduce_weight.data
        if reduce_bias is not None:
            rv['fc.reduce.bias'] = reduce_bias.data 
        
        expand_weight = self.get_active_expand_weight(mid_c, cin).contiguous()
        expand_bias = self.get_active_expand_bias(cin)
        rv['fc.expand.weight'] = expand_weight.data 
        if expand_bias is not None:
            rv['fc.expand.bias'] = expand_bias.data

        self.fc.reduce.set_lsq_state_dict('fc.reduce.', rv)
        self.fc.expand.set_lsq_state_dict('fc.expand.', rv)
        return rv
        

class DynamicQSE4ResNet(nn.Module):

    REDUCTION = 4

    def __init__(self, max_cin, max_mid):
        super().__init__()
        self.max_cin = max_cin 
        self.max_mid = max_mid 

        self.fc = nn.Sequential(OrderedDict([
            ('reduce', DynamicQConv2d(self.max_cin, self.max_mid, [1], 1, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('expand', DynamicQConv2d(self.max_mid, self.max_cin, [1], 1, bias=True)),
            ('hsigmoid', nn.Hardsigmoid(inplace=True)),
        ]))

    @property
    def active_mid_channels(self):
        return self.fc.reduce.active_out_channels

    @active_mid_channels.setter
    def active_mid_channels(self, v):
        self.fc.reduce.active_out_channels = v 

    def _scale(self, x: Tensor) -> Tensor:
        cin = x.shape[1]
        self.fc.expand.active_out_channels = cin
        x = x.mean([2, 3], keepdim=True) 
        return self.fc(x)

    def forward(self, x: Tensor) -> Tensor:
        return self._scale(x) * x

    def active_state_dict(self, cin: int):
        rv = {}
        self.fc.reduce.set_active_state_dict('fc.reduce.', rv, cin=cin)
        self.fc.expand.set_active_state_dict('fc.expand.', rv, cin=self.active_mid_channels, cout=cin)
        return rv

    @staticmethod
    def get_mid_channels(cin: int) -> int:
        return make_divisible(cin // DynamicQSE4ResNet.REDUCTION)
        
        
class DynamicQConvNormActivation(nn.Sequential):
    
    def __init__(
        self,
        max_in_channels: int,
        max_out_channels: int,
        kernel_size_list: List[int],
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = DynamicBatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
    ) -> None:
        layers = [('conv', DynamicQConv2d(max_in_channels, max_out_channels, kernel_size_list, stride,
                                  dilation=dilation, groups=groups, bias=norm_layer is None))]
        if norm_layer is not None:
            layers.append(('norm', norm_layer(max_out_channels)))
        if activation_layer is not None:
            layers.append(('act', activation_layer(inplace=inplace)))

        super().__init__(OrderedDict(layers))

    def forward(self, x: Tensor, out_channels=None, kernel_size=None) -> Tensor:
        x = self.conv(x, out_channels, kernel_size)
        x = self.norm(x)
        if hasattr(self, 'act'):
            x = self.act(x)
        return x

    @property
    def active_out_channels(self):
        return self.conv.active_out_channels

    @active_out_channels.setter
    def active_out_channels(self, cout: int):
        self.conv.active_out_channels = cout

    @property
    def active_kernel_size(self):
        return self.conv.active_kernel_size

    @active_kernel_size.setter
    def active_kernel_size(self, kernel_size: int):
        self.conv.active_kernel_size = kernel_size

    def active_state_dict(self, cin: int, cout=None, kernel_size=None):
        rv = {}
        self.conv.set_active_state_dict('conv.', rv, cin, cout, kernel_size)
        self.norm.set_active_state_dict('norm.', rv, cout or self.active_out_channels)
        return rv


class _DynamicQDWConvNormActivation(nn.Sequential):
    
    def __init__(
        self,
        max_channels: int,
        kernel_size_list: List[int],
        stride: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = DynamicBatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
        DWConv=DynamicQDWConv2d
    ) -> None:
        layers = [('dwconv', DWConv(max_channels, kernel_size_list, stride,
                                  dilation=dilation, bias=norm_layer is None))]
        if norm_layer is not None:
            layers.append(('norm', norm_layer(max_channels)))
        if activation_layer is not None:
            layers.append(('act', activation_layer(inplace=inplace)))

        super().__init__(OrderedDict(layers))

    def forward(self, x: Tensor, kernel_size=None) -> Tensor:
        x = self.dwconv(x, kernel_size)
        x = self.norm(x)
        x = self.act(x)
        return x

    @property
    def active_kernel_size(self):
        return self.dwconv.active_kernel_size

    @active_kernel_size.setter
    def active_kernel_size(self, kernel_size: int):
        self.dwconv.active_kernel_size = kernel_size

    def active_state_dict(self, cin: int, kernel_size=None):
        rv = {}
        self.dwconv.set_active_state_dict('dwconv.', rv, cin, kernel_size)
        self.norm.set_active_state_dict('norm.', rv, cin)
        return rv


class DynamicQDWConvNormActivation(_DynamicQDWConvNormActivation):

    def __init__(self, max_channels: int, kernel_size_list: List[int], stride: int = 1, norm_layer: Optional[Callable[..., torch.nn.Module]] = DynamicBatchNorm2d, activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU, dilation: int = 1, inplace: bool = True) -> None:
        super().__init__(max_channels, kernel_size_list, stride, norm_layer, activation_layer, dilation, inplace, DWConv=DynamicQDWConv2d)


class DynamicQDWConvK3NormActivation(_DynamicQDWConvNormActivation):

    def __init__(self, max_channels: int, kernel_size_list: List[int], stride: int = 1, norm_layer: Optional[Callable[..., torch.nn.Module]] = DynamicBatchNorm2d, activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU, dilation: int = 1, inplace: bool = True) -> None:
        super().__init__(max_channels, kernel_size_list, stride, norm_layer, activation_layer, dilation, inplace, DynamicQDWConv2dK3)