# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .common_ops import BNSuper2d, drop_connection
from .sample_utils import mutate_dims

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1), w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class DynamicConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _bn_forward(self, x, bn, channels, type=None):
        bn.set_conf(channels)
        return bn(x)

    def _conv_flops(self, inp, oup, groups, kr_size, oup_size):
        return (inp*oup*(kr_size**2)*(oup_size)**2)//groups
    
    def _slice_conv_op(self, conv, ):
        pass

    def set_conf(self, sampled_channels, sampled_ratio, sampled_kernel_size, sampled_channels_2):
        pass

class MobileBlock(DynamicConvBlock):
    def __init__(self, max_channels, min_channels, in_channels, ratio, stride, layer_idx, act=nn.ReLU, inverse_sampling=False, drop_connect_prob=0., se=False, kr_size=[5, 3]):
        super().__init__()
        self.max_channels = max_channels if stride == 1 else in_channels
        self.min_channels = min_channels
        self.max_out_channels = max_channels
        self.ratio = ratio
        self.stride = stride
        self.inverse_sampling = inverse_sampling
        self.se_module = se
        self.layer_idx = layer_idx
        self.kr_size = kr_size

        max_ratio = max(ratio)
        self.max_inner_channels = max_channels * max_ratio

        self.sampled_res = -1

        # regulation settings
        self.drop_connect_prob = drop_connect_prob

        # pw conv
        if max(ratio) > 1:
            self.pw_conv = nn.Conv2d(
                max_channels, self.max_inner_channels, 1, 1, bias=False, padding=0)
            # self.pw_bn = nn.BatchNorm2d(self.max_inner_channels)
            self.pw_bn = BNSuper2d(self.max_inner_channels)
            self.pw_act = act()

        # dw conv
        self.dw_conv = nn.Conv2d(self.max_inner_channels, self.max_inner_channels,
                                 max(self.kr_size), stride, 1, groups=self.max_inner_channels, bias=False)
        # self.dw_bn = nn.BatchNorm2d(self.max_inner_channels)
        self.dw_bn = BNSuper2d(self.max_inner_channels)
        self.dw_act = act()

        if self.se_module:
            self.se_adaptivepooling = nn.AdaptiveAvgPool2d(1)
            self.se_conv1 = nn.Conv2d(max_channels*max(self.ratio), max_channels*max(self.ratio)//4, 1, 1, bias=True, padding=0)
            self.se_act1 = nn.ReLU()
            self.se_conv2 = nn.Conv2d(max_channels*max(self.ratio)//4, max_channels*max(self.ratio), 1, 1, bias=True, padding=0)
            self.se_act2 = nn.Hardsigmoid()

        # pw-linear conv
        self.pwl_conv = nn.Conv2d(
            self.max_inner_channels, self.max_out_channels, 1, 1, bias=False, padding=0)
        # self.pwl_bn = nn.BatchNorm2d(self.max_out_channels)
        self.pwl_bn = BNSuper2d(
            self.max_out_channels, init=0 if self.stride == 1 else 1)

        
        self.shortcut_conv = nn.Conv2d(max_channels, self.max_out_channels, 1, 1, bias=False, padding=0)
            # self.shortcut_bn = nn.BatchNorm2d(self.max_out_channels)
            # self.shortcut_bn = BNSuper2d(self.max_out_channels)

        self.sampled_channels = -1
        self.sampled_ratio = -1
        self.sampled_kernel_size = -1
        self.in_channels = -1

    def compute_flops(self):
        if max(self.ratio) > 1:
            inner_channel = self.sampled_channels * self.sampled_ratio
            pw_conv_flops = self._conv_flops(
                self.sampled_channels, inner_channel, 1, 1, self.sampled_res*2 if self.stride > 1 else self.sampled_res)
        else:
            inner_channel = self.sampled_channels
            pw_conv_flops = 0

        dw_conv_flops = self._conv_flops(
            inner_channel, inner_channel, inner_channel, self.sampled_kernel_size, self.sampled_res)
        pwl_conv_flops = self._conv_flops(
            inner_channel, self.sampled_channels_2, 1, 1, self.sampled_res)

        total = pw_conv_flops + dw_conv_flops + pwl_conv_flops

        if self.se_module:
            se_conv_flops = self._conv_flops(
                inner_channel, inner_channel//4, 1, 1, 1) * 2
            total += se_conv_flops

        if self.stride == 2:
            shortcut_flops = self._conv_flops(
                self.sampled_channels, self.sampled_channels_2, 1, 1, self.sampled_res)
            total += shortcut_flops

        return total

    def forward(self, x):
        res = x
        in_channels = x.size(1)
        assert in_channels == self.sampled_channels

        inner_channel = in_channels * self.sampled_ratio

        if max(self.ratio) > 1:
            if not self.inverse_sampling:
                activated_conv_weight = self.pw_conv.weight[:inner_channel, :in_channels].contiguous()
            else:
                activated_conv_weight = self.pw_conv.weight[self.max_inner_channels-inner_channel:self.max_inner_channels,
                                                            self.max_channels-self.sampled_channels:self.max_channels]
            out = self.pw_conv._conv_forward(x, activated_conv_weight, bias=None)
            out = self._bn_forward(out, self.pw_bn, inner_channel, type='pw')
            out = self.pw_act(out)
        else:
            out = x

        if not self.inverse_sampling:
            activated_conv_weight = self.dw_conv.weight[:inner_channel, ].contiguous()
        else:
            activated_conv_weight = self.dw_conv.weight[self.max_inner_channels -
                                                        inner_channel:self.max_inner_channels, ]
        self.dw_conv.groups = inner_channel
        assert self.sampled_kernel_size in self.kr_size
        self.dw_conv.padding = 1 if self.sampled_kernel_size == 3 else 2

        activated_conv_weight = activated_conv_weight
        if max(self.kr_size) != min(self.kr_size) and self.sampled_kernel_size == 3:
            activated_conv_weight = activated_conv_weight[:, :, 1:4, 1:4]
            
        out = self.dw_conv._conv_forward(
            out, activated_conv_weight, bias=None)
        out = self._bn_forward(out, self.dw_bn, inner_channel, type='dw')
        out = self.dw_act(out)
        # print(self.sampled_kernel_size,out.shape)

        if self.se_module:
            se_out = self.se_adaptivepooling(out)
            activated_conv_weight = self.se_conv1.weight[: inner_channel//4, :inner_channel]
            activated_conv_bias = self.se_conv1.bias[:inner_channel//4]
            se_out = F.conv2d(se_out, activated_conv_weight, activated_conv_bias, 1, 0, 1, 1)
            se_out = self.se_act1(se_out)
            
            activated_conv_weight = self.se_conv2.weight[:inner_channel, :inner_channel//4]
            activated_conv_bias = self.se_conv2.bias[:inner_channel]
            se_out = F.conv2d(se_out, activated_conv_weight, activated_conv_bias, 1, 0, 1, 1)
            se_out = self.se_act2(se_out)

            out = se_out * out

        if not self.inverse_sampling:
            activated_conv_weight = self.pwl_conv.weight[:
                                                         self.sampled_channels_2, :inner_channel]
        else:
            activated_conv_weight = self.pwl_conv.weight[self.max_out_channels-self.sampled_channels_2: self.max_out_channels,
                                                         self.max_inner_channels-inner_channel:self.max_inner_channels]
        out = self.pwl_conv._conv_forward(
            out, activated_conv_weight, bias=None)
        out = self._bn_forward(
            out, self.pwl_bn, self.sampled_channels_2, type='pwl')
        
        if in_channels == self.sampled_channels_2 and self.stride == 1:
            return out + res
        
        if self.stride > 1:
            padding = 0 if res.size(-1) % 2 == 0 else 1
            res = F.avg_pool2d(res, self.stride, padding=padding)
        
        if in_channels != self.sampled_channels_2:
            activated_conv_weight = self.shortcut_conv.weight[:self.sampled_channels_2, :self.sampled_channels]
            res = self.shortcut_conv._conv_forward(res, activated_conv_weight, bias=None)
        
        return out + res

    def set_conf(self, channels, last_stage_sampled_channels, sampled_res, sampled_kernel_size, sampled_ratio, sampled_out_channels):
        self.sampled_res = sampled_res
        self.sampled_ratio = sampled_ratio
        self.sampled_kernel_size = sampled_kernel_size

        self.sampled_channels_2 = sampled_out_channels
        self.sampled_channels = channels if self.layer_idx != 0 else last_stage_sampled_channels

    def arch_sampling(self, mode, sampled_res, channels, last_stage_sampled_channels=0, min_conv_ratio=0, min_kr_size=0, prob=1.):
        self.sampled_res = sampled_res

        if mode == 'max':
            sampled_ratio = max(self.ratio)
            sampled_kernel_size = max(self.kr_size)

        elif mode == 'min':
            sampled_ratio = min(self.ratio)
            sampled_kernel_size = min(self.kr_size)
        else:
            ratio, kr_size = [], []
            for r in self.ratio:
                if r >= min_conv_ratio:
                    ratio.append(r)
            
            for k in self.kr_size:
                if k >= min_kr_size:
                    kr_size.append(k)
            
            sampled_ratio = mutate_dims(ratio, prob=prob)
            sampled_kernel_size = mutate_dims(kr_size, prob=prob)

        return sampled_ratio, sampled_kernel_size


class SuperCNNLayer(nn.Module):
    def __init__(self, depth, min_depth, min_channels, in_channels: int, out_channels: int, ratio, act=nn.ReLU, downsampling=True, res=-1, se=False):
        super().__init__()

        self.cnn_type = None
        self.cnn_choices = ['mbnet', 'shufflenet']
        self.max_depth = depth
        self.min_depth = min_depth
        self.ratio = ratio
        self.last_stage_max_channels = in_channels
        self.max_channels = out_channels
        self.min_channels = min_channels
        self.channels_list = [c for c in range(self.min_channels, self.max_channels+1, 8)]
        self.res = [ele for ele in res]
        self.sampled_res = -1

        self.sampled_layers = -1
        self.sampled_channels = -1
        self.layer_wise_sampling = False # True for layer-wise, False for block-wise
        
        layers = []
        for idx in range(self.max_depth):
            stride = 2 if idx == 0 and downsampling else 1

            layers.append(nn.ModuleList([
                MobileBlock(max_channels=out_channels, min_channels=min_channels, in_channels=in_channels,
                            ratio=ratio, stride=stride, act=act, layer_idx=idx, se=se), 
            ]))
        self.layers = nn.ModuleList(layers)

    def set_conf(self, cnn_type, sampled_layers, sampled_channels):
        self.sampled_channels = sampled_channels
        self.sampled_layers = sampled_layers
        self.cnn_type = cnn_type

    def get_conf(self):
        return (self.cnn_type, self.sampled_layers, self.sampled_channels)

    def compute_flops(self):
        total = 0
        
        for idx in range(len(self.layers)):
            
            if idx >= self.sampled_layers:
                break
            
            total += self.layers[idx][self.cnn_type[idx]].compute_flops()

        return total

    def arch_sampling(self, mode='max', sampled_res_idx=-1, in_channels=-1, equal_channels=False, 
        min_depths=0, min_channels=0, min_conv_ratio=0, min_kr_size=0, prob=1):
        assert mode in ['min', 'uniform', 'r_uniform', 'max']
        assert sampled_res_idx != -1
        
        self.sampled_res = self.res[sampled_res_idx]

        if mode == 'max':
            sampled_layers = self.max_depth
            sampled_channels = self.max_channels
        elif mode == 'uniform':
            min_depths = max(self.min_depth, min_depths)
            depths_choices = [d for d in range(min_depths, self.max_depth+1)]
            sampled_layers = mutate_dims(depths_choices, prob=prob)

            start_channels_id = 0 if min_channels <= 0 else self.channels_list.index(min_channels)
            channels_list = self.channels_list[start_channels_id:]

            while True:
                sampled_channels = mutate_dims(channels_list, prob=prob)

                assert sampled_channels % 8 == 0

                if not equal_channels:
                    if sampled_channels >= in_channels:
                        break
                else:
                    if sampled_channels == in_channels:
                        break
        elif mode == 'min':
            sampled_layers = self.min_depth
            sampled_channels = self.min_channels

        cnn_type = [0 for _ in range(sampled_layers)] # just for MobileNet blocks (id=0)

        self.set_conf(cnn_type, sampled_layers, sampled_channels)

        if not self.layer_wise_sampling:
            first_conv_layer = self.layers[0][self.cnn_type[0]]
            assert isinstance(first_conv_layer, MobileBlock)
            block_wise_conv_conf = first_conv_layer.arch_sampling(mode=mode, sampled_res=self.sampled_res, channels=sampled_channels, 
                min_conv_ratio=min_conv_ratio, min_kr_size=min_conv_ratio, prob=prob)

        conv_ratios = []
        kernel_sizes = []
        for idx in range(len(self.layers)):
            if idx >= self.sampled_layers:
                break

            last_stage_sampled_channels = in_channels if idx == 0 else -1
            layer = self.layers[idx][self.cnn_type[idx]]

            if mode == 'max':
                out_channels = layer.max_out_channels
            elif mode == 'min':
                out_channels = layer.min_channels
            else:
                out_channels = sampled_channels

            if self.layer_wise_sampling:
                sampled_conv_ratio, sampled_kernel_size = layer.arch_sampling(
                    mode, sampled_res=self.sampled_res, channels=sampled_channels, last_stage_sampled_channels=last_stage_sampled_channels)
            else:
                sampled_conv_ratio, sampled_kernel_size = block_wise_conv_conf
            
            layer.set_conf(sampled_channels, last_stage_sampled_channels, self.sampled_res, sampled_kernel_size, sampled_conv_ratio, out_channels)

            conv_ratios.append(sampled_conv_ratio)
            kernel_sizes.append(sampled_kernel_size)

        return sampled_layers, sampled_channels, conv_ratios, kernel_sizes

    def forward(self, x):
        for idx in range(len(self.layers)):
            
            if idx >= self.sampled_layers:
                break
            x = self.layers[idx][self.cnn_type[idx]](x)

        return x


def conv_stem(n, act=nn.ReLU):
    return torch.nn.Sequential(
        Conv2d_BN(3, n // 8, 3, 2, 1),
        act(),
        Conv2d_BN(n//8, n//8, ks=3, stride=1, pad=1, groups=n // 8),
        act(),
        nn.Conv2d(n//8, n//8, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(n//8)
    )