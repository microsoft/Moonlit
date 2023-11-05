# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.from webbrowser import Elinks
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from timm.models.layers import trunc_normal_
from .common_ops import LNSuper, BNSuper1d, BNSuper2d, drop_connection
from .cnn import MobileBlock
from .sample_utils import mutate_dims

class ELinear(nn.Linear):
    def __init__(self, max_in_dim: int, max_out_dim: int or tuple, bias = True, scale=1, dim_variable=False, dim_variable_factor=1., inverse_sampling=False, **kwargs) -> None:
        super().__init__(max_in_dim, max_out_dim if isinstance(max_out_dim, int) else sum(max_out_dim), bias)
        self.max_in_dim = max_in_dim

        if isinstance(max_out_dim, int):
            self.max_out_dim = max_out_dim
        else:
            self.max_qk_dim = max_out_dim[0]
            self.v_qk_dim = max_out_dim[1]
        
        self.scale = scale
        self.activated_in_dim = -1
        self.activated_out_dim = -1

        self.dim_variable = dim_variable
        self.dim_variable_factor = dim_variable_factor

        self.inverse_sampling = inverse_sampling

    def set_conf(self, activated_in_dim, activated_out_dims):
        self.activated_in_dim = activated_in_dim
        self.activated_out_dim = activated_out_dims

    def get_conf(self):
        return (self.activated_in_dim, self.activated_out_dim)

    def _get_weight_bias(self):
        if not isinstance(self.activated_out_dim, (tuple, list)):

            return (self.weight[: self.activated_out_dim, :self.activated_in_dim], self.bias[: self.activated_out_dim] if self.bias is not None else None)

        else:
            q_dim_start = 0
            q_dim_end = self.activated_out_dim[0]
            q_weights = self.weight[q_dim_start: q_dim_end, : self.activated_in_dim]

            k_dim_start = self.max_qk_dim//2
            k_dim_end = self.max_qk_dim // 2 + q_dim_end
            k_weights = self.weight[k_dim_start: k_dim_end, : self.activated_in_dim]

            v_dim_start = self.max_qk_dim
            v_dim_end = self.max_qk_dim + self.activated_out_dim[1]
            v_weights = self.weight[v_dim_start: v_dim_end, : self.activated_in_dim]

            if self.bias is not None:
                q_bias = self.bias[: q_dim_end]
                k_bias = self.bias[k_dim_start: k_dim_end]
                v_bias = self.bias[v_dim_start: v_dim_end]
                bias = torch.cat([q_bias, k_bias, v_bias])
            else:
                bias = None

            return (torch.cat([q_weights, k_weights, v_weights], dim=0), bias)

    def forward(self, x):
        w, b = self._get_weight_bias()
        return F.linear(x, w, b) * self.scale

class SuperFFN(nn.Module):
    def __init__(self, super_dim, ratio, pre_norm=True, act=nn.Hardswish, norm=LNSuper, drop_connect_prob=0.):
        super().__init__()
        self.super_dim = super_dim
        self.ratio = ratio
        self.sampled_ratio = -1
        self.pre_norm = pre_norm

        super_inner_dim = int(ratio[0] * super_dim)
        if self.pre_norm:
            self.norm1 = norm(super_dim)
        else:
            self.norm1 = norm(super_inner_dim)
            self.norm2 = norm(super_dim, init=0)
        self.fc1 = ELinear(super_dim, super_inner_dim)
        self.fc2 = ELinear(super_inner_dim, super_dim)
        self.act = act()

        # regulation settings
        self.drop_connect_prob = drop_connect_prob

    def set_conf(self, activated_dim, ratio):
        self.sampled_ratio = ratio
        self.fc1.set_conf(activated_in_dim=activated_dim,
                          activated_out_dims=int(ratio*activated_dim))
        self.fc2.set_conf(activated_in_dim=int(ratio*activated_dim),
                          activated_out_dims=activated_dim)
        
        if self.pre_norm:
            self.norm1.set_conf(activated_dim)
        else:
            self.norm1.set_conf(int(ratio*activated_dim))
            self.norm2.set_conf(activated_dim)

    def get_conf(self):
        return (self.sampled_ratio)
    
    def compute_flops(self):
        return self.fc1.activated_in_dim * self.fc1.activated_out_dim + self.fc2.activated_in_dim * self.fc2.activated_out_dim

    def arch_sampling(self, mode='max', sampled_channels=-1, min_mlp_ratio=0, prob=1.):
        if mode == 'max':
            sampled_ratio = max(self.ratio)
        elif mode == 'min':
            sampled_ratio = min(self.ratio)
        else:
            ratio = []
            for r in self.ratio:
                if r >= min_mlp_ratio:
                    ratio.append(r)
            sampled_ratio = mutate_dims(ratio, prob=1)
        
        self.set_conf(sampled_channels, sampled_ratio)
        return sampled_ratio

    def forward(self, x):
        res = x
        
        if self.pre_norm:
            out = self.fc2(self.act(self.fc1(self.norm1(x))))
        else:
            out = self.act(self.norm1(self.fc1(x)))
            out = self.norm2(self.fc2(out))
        
        if self.drop_connect_prob > 0:
            out = drop_connection(out, self.drop_connect_prob, is_training = self.training, transformer = True)
        
        return res + out

class SuperAttention(nn.Module):
    def __init__(self, stride, super_dim, windows_size, num_heads, last_stage_max_channels, res, 
                    qk_scale, v_scale, qkv_bias=False, pre_norm=True, attn_ratio=2, norm=LNSuper, inverse_sampling=False, act=nn.Hardswish, conv_downsampling=True, 
                    layer_idx=-1, drop_connect_prob=0., talking_head=False, dw_downsampling=False, head_dims=16, use_res_specific_RPN=False):
        super().__init__()
        self.super_dim = super_dim

        self.head_dim = head_dims
        self.fix_head_dim = True

        self.num_heads = num_heads
        self.pre_norm = pre_norm

        self.shift_size = 0
        self.windows_size = windows_size

        self.stride = stride
        self.attn_ratio = attn_ratio
        self.qk_scale = qk_scale
        assert max(self.qk_scale) <= 1, 'qk scale must <= 1!'
        self.v_scale = v_scale
        self.act = act()

        self.layer_idx = layer_idx

        self.qk_super_dim = super_dim*2
        self.v_super_dim = int(super_dim*max(self.v_scale))
        self.qkv = ELinear(max_in_dim=super_dim, max_out_dim=(self.qk_super_dim, self.v_super_dim), bias=qkv_bias)
        self.proj = ELinear(max_in_dim=self.v_super_dim, max_out_dim=super_dim)

        max_heads = self.super_dim // self.head_dim if self.fix_head_dim else max(num_heads)

        self.linear_attn = talking_head
        self.dw_v = True
        norm_v = not pre_norm
        
        if self.dw_v:
            self.dw_v_conv = nn.Conv2d(self.v_super_dim, self.v_super_dim, groups=self.v_super_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.dw_v_norm = None
            if norm_v:
                self.dw_v_norm = norm(self.v_super_dim)

        self.conv_downsampling = conv_downsampling
        
        # regularization settings
        self.drop_connect_prob = drop_connect_prob

        self.sampled_res = -1
        
        if not self.pre_norm:
            self.norm1 = norm(super_channels=(self.qk_super_dim, self.v_super_dim))
            self.norm2 = norm(super_dim, init=0)
        else:
            self.norm1 = norm(super_channels=super_dim)
        
        self.attn_scale = 1
        self.res = res

        self.downsampling = False
        self.use_dw_downsampling = dw_downsampling

        if layer_idx == 0:
            self.downsampling = True if stride == 2 else False
            self.dw_ratio = 6
            self.reduction = MobileBlock(super_dim, super_dim, last_stage_max_channels, ratio=[6], stride=stride, layer_idx=0, act=act, kr_size=[3], se=True)

        self.sampled_channels = -1
        self.sampled_windows_size = -1
        self.sampled_qkdim = -1
        self.sampled_vdim = -1
        self.sampled_num_heads = -1
        self.sampled_qk_scale = -1
        self.sampled_v_scale = -1
        self.inverse_sampling = inverse_sampling

        def attn_pos_idx(input_res):
            coords_h = torch.arange(input_res)
            coords_w = torch.arange(input_res)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - \
                coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += input_res - 1  # shift to start from 0
            relative_coords[:, :, 1] += input_res - 1
            relative_coords[:, :, 0] *= 2 * input_res - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

            return relative_position_index
        
        self.use_res_specific_attn_bias = use_res_specific_RPN

        max_res = max(self.res)
        self.attention_bias = nn.Parameter(torch.zeros(((2 * max_res - 1) * (2 * max_res - 1), max_heads)))
        max_pos_index = attn_pos_idx(max_res)
        self.register_buffer("relative_position_index", max_pos_index)
        trunc_normal_(self.attention_bias, std=.02)

    def _do_dw_v_conv(self, dim, x):
        assert len(x.shape) == 3 # _B, N, _L
        assert hasattr(self, 'dw_v_conv')

        _B, N, _L = x.shape
        H = W = int(N**0.5)

        x = x.transpose(1, 2).reshape((-1, _L, H, W))
        weights = self.dw_v_conv.weight[:dim, ]
        self.dw_v_conv.groups = dim # set groups
        y = self.dw_v_conv._conv_forward(x, weight=weights, bias=None)
        y = y.flatten(2).transpose(1, 2)

        if self.dw_v_norm is not None:
            self.dw_v_norm.set_conf(dim)
            return self.dw_v_norm(y)
        else:
            return y

    def _window_attention(self, x: torch.Tensor):
        # public code does not include swin self-attention
        raise NotImplementedError
    
    def _attention(self, x: torch.Tensor):
        B, HW, C = x.shape
        
        qkv = self.qkv(x)

        if not self.pre_norm:
            qkv = self.norm1(qkv)

        q, k, v = qkv.view(B, HW, self.sampled_num_heads, -1).split(
            [self.sampled_qkdim, self.sampled_qkdim, self.sampled_vdim], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        if self.dw_v:
            L = self.sampled_num_heads * self.sampled_vdim
            v = v.reshape(B, HW, L)
            v = self._do_dw_v_conv(L, v).view(B, HW, self.sampled_num_heads, -1)
        
        v = v.permute(0, 2, 1, 3)

        attn = (q@k.transpose(-2, -1)) * (self.sampled_qkdim ** -0.5)

        _H, _W = int(attn.shape[3]**.5), int(attn.shape[3]**.5)
        
        relative_position_bias = self.attention_bias[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(max(self.res), max(self.res), max(self.res), max(self.res), -1)
        relative_position_bias = relative_position_bias[:_H, :_W, :_H, :_W, :].contiguous().reshape(_H*_W, _H*_W, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)[:, :attn.shape[1]]  # nH, Wh*Ww, Wh*Ww
        
        attn += relative_position_bias

        attn = attn.softmax(dim=-1)
        
        out = (attn@v).transpose(1, 2)
        
        out = torch.reshape(out, (B, HW, -1))
        out = self.act(out)

        out = self.proj(out)
        if not self.pre_norm:
            out = self.norm2(out)
        
        return out
    
    def compute_flops(self):
        total = 0 

        if self.layer_idx == 0:
            kr_size = 3
            sampled_res = self.sampled_res*2 if self.stride == 2 else self.sampled_res
            
            inner_channel = self.last_stage_sampled_channels*6
            output_size = self.sampled_res*self.sampled_res
            downsampling_flops = self.last_stage_sampled_channels*inner_channel*sampled_res*sampled_res + \
                                    inner_channel*kr_size*kr_size*output_size + \
                                    inner_channel*self.sampled_channels*output_size

            total += downsampling_flops
        
        if self.sampled_windows_size == 1:
            r1 = r2 = self.sampled_res
        else:
            r1 = r2 = self.sampled_windows_size
        
        # q @ k; softmax; (qk.T)v
        qk_flops = self.sampled_num_heads * ((r1*r2)**2) * self.sampled_qkdim
        softmax_flops = self.sampled_num_heads * (r1*r2)**2
        attention_flops = self.sampled_num_heads * self.sampled_vdim * (r1*r2)**2

        total += (qk_flops + attention_flops + softmax_flops)

        # q,k,v = qkv(x) & proj
        qkv_flops = self.sampled_channels * self.sampled_res * self.sampled_res * (self.sampled_qkdim * self.sampled_num_heads * 2 + self.sampled_vdim * self.sampled_num_heads)
        proj_flops = self.sampled_res * self.sampled_res * self.sampled_vdim * self.sampled_num_heads * self.sampled_channels
        total += (qkv_flops + proj_flops)

        # extra flops
        if self.linear_attn:
            linear_attn_flops = 2 * (self.sampled_res * self.sampled_res * self.sampled_vdim * self.sampled_num_heads * self.sampled_num_heads)
            total += linear_attn_flops
        
        if self.dw_v:
            dw_v_flops = self.sampled_res * self.sampled_res * (self.sampled_vdim * self.sampled_num_heads)
            total += dw_v_flops

        return total
    
    def set_conf(self, sampled_windows_size, sampled_num_heads, sampled_qk_scale, sampled_v_scale, sampled_channels, last_stage_sampled_channels, sampled_res):
        self.sampled_num_heads = sampled_num_heads

        sampled_qkdim = int(sampled_channels//sampled_num_heads * sampled_qk_scale)
        sampled_vdim = int(sampled_v_scale * sampled_channels//sampled_num_heads)
        
        self.sampled_channels = sampled_channels
        self.last_stage_sampled_channels = last_stage_sampled_channels
        self.sampled_res = sampled_res
        
        if self.pre_norm:
            self.norm1.set_conf(sampled_channels)
        else:
            self.norm1.set_conf((sampled_qkdim * self.sampled_num_heads, sampled_vdim*self.sampled_num_heads))
            self.norm2.set_conf(sampled_channels)
        self.qkv.set_conf(sampled_channels, (sampled_qkdim *
                          self.sampled_num_heads, sampled_vdim*self.sampled_num_heads))
        self.proj.set_conf(
            sampled_vdim*self.sampled_num_heads, sampled_channels)
        
        self.sampled_qkdim = sampled_qkdim
        self.sampled_vdim = sampled_vdim
        self.sampled_windows_size = sampled_windows_size
        self.sampled_v_scale = sampled_v_scale
        self.sampled_num_heads = sampled_num_heads

    def get_conf(self):
        return (self.sampled_qkdim, self.sampled_vdim, self.sampled_windows_size, self.sampled_num_heads)
    
    def _windows_size_sampling(self, mode, res):
        _windows_size = []
        res_idx = self.res.index(res)

        for w in self.windows_size[res_idx]:
            if w <= res and res % w == 0:
                _windows_size.append(w)
        
        if len(_windows_size) == 0 or (len(_windows_size) == 1 and _windows_size[0] == 1):
            return 1
        
        if 1 in _windows_size:
            _windows_size.pop(_windows_size.index(1))
        
        if mode == 'max':
            sampled_windows_size = min(_windows_size)
        elif mode == 'min':
            sampled_windows_size = max(_windows_size)
        else:
            sampled_windows_size = random.choice(_windows_size)
        
        return sampled_windows_size

    def arch_sampling(self, mode='max', sampled_res=-1, sampled_channels=-1, last_stage_sampled_channels=-1, min_qk_scale=0, min_v_scale=0, prob=1.):
        assert sampled_channels > 0

        if mode == 'max':
            sampled_windows_size = self._windows_size_sampling(mode, sampled_res)
            sampled_num_heads = max(self.num_heads)
            sampled_qk_scale = max(self.qk_scale)
            sampled_v_scale = max(self.v_scale)
            self.dw_ratio = 6
        elif mode == 'min':
            sampled_windows_size = self._windows_size_sampling(mode, sampled_res)
            sampled_num_heads = min(self.num_heads)
            sampled_qk_scale = min(self.qk_scale)
            sampled_v_scale = min(self.v_scale)
            self.dw_ratio = 6
        else:
            sampled_windows_size = self._windows_size_sampling(mode, sampled_res)
            v_scale, qk_scale = [], []
            for qk in self.qk_scale:
                if qk >= min_qk_scale:
                    qk_scale.append(qk)
            
            for v in self.v_scale:
                if v >= min_v_scale:
                    v_scale.append(v)
            sampled_qk_scale = random.choice(qk_scale)
            sampled_v_scale = mutate_dims(v_scale, prob=prob)
            self.dw_ratio = 6
        
        sampled_num_heads = sampled_channels // self.head_dim

        return sampled_windows_size, sampled_num_heads, sampled_qk_scale, sampled_v_scale

    def forward(self, x):
        if self.layer_idx == 0:
            _, N, C = x.shape
            H = int(N**0.5)
            W = int(N**0.5)
            
            if self.conv_downsampling:
                x = x.transpose(1, 2).reshape((-1, C, H, W))

                self.reduction.set_conf(channels=C, last_stage_sampled_channels=C, sampled_res=self.sampled_res, sampled_kernel_size=3, sampled_ratio=self.dw_ratio, sampled_out_channels=self.sampled_channels)
                x = self.reduction(x)
            
                x = x.flatten(2).transpose(1, 2)
            else:
                raise NotImplementedError
                
        res = x

        if self.pre_norm:
            x = self.norm1(x)
        
        out = self._attention(x)
        
        if self.drop_connect_prob > 0:
            out = drop_connection(out, self.drop_connect_prob, is_training = self.training, transformer = True)

        return res + out

class SuperTransformerBlock(nn.Module):
    def __init__(self, depth, min_depth, in_channels, min_channels, max_channels, num_heads, res, windows_size,
                 pre_norm, mlp_ratio, qk_scale, v_scale, norm_layer, downsampling, act=nn.Hardswish, talking_head=False, dw_downsampling=False, head_dims=16, use_res_specific_RPN=False):
        super().__init__()
        self.max_depth = depth
        self.min_depth = min_depth
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.channels_list = [c for c in range(self.min_channels, self.max_channels+1, head_dims)]
        self.num_heads = num_heads
        self.windows_size = windows_size
        self.res = [ele for ele in res]
        self.head_dims = head_dims

        self.sampled_res = -1
        # self.max_inner_dim = inner_dim
        
        self.max_mlp_ratio = mlp_ratio
        self.sampled_channels = -1

        self.layer_wise_sampling = False # True for layer-wise, False for block-wise

        layers = []
        for i in range(self.max_depth):
            stride = 2 if i == 0 and downsampling else 1
            layers.append(
                SuperAttention(stride=stride, super_dim=max_channels,
                          windows_size=windows_size, num_heads=num_heads, res=self.res, 
                          last_stage_max_channels=in_channels, qk_scale=qk_scale, v_scale=v_scale, pre_norm=pre_norm, norm=norm_layer, act=act, layer_idx=i, 
                          talking_head=talking_head, dw_downsampling=dw_downsampling, head_dims=head_dims, use_res_specific_RPN=use_res_specific_RPN)
                          )
            layers.append(SuperFFN(super_dim=self.max_channels, ratio=mlp_ratio, pre_norm=pre_norm, norm=norm_layer, act=act))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i >= self.sampled_layers*2:
                break
            x = layer(x)
        return x
    
    def compute_flops(self):
        total = 0

        for i, layer in enumerate(self.layers):
            if i >= self.sampled_layers*2:
                break
            layer_flops = layer.compute_flops()

            if isinstance(layer, SuperFFN):
                layer_flops *= (self.sampled_res * self.sampled_res)
            
            total += layer_flops
        
        return total
    
    def arch_sampling(self, mode='max', sampled_res_idx = -1, in_channels=-1, min_channels=0, min_depths=0, min_mlp_ratio=0, min_qk_scale=0, min_v_scale=0, prob=1):
        assert sampled_res_idx != -1
        self.sampled_res = self.res[sampled_res_idx]

        if mode == 'max':
            sampled_channels = self.max_channels
            sampled_layers = self.max_depth
        elif mode == 'min':
            sampled_channels = self.min_channels
            sampled_layers = self.min_depth
        else:
            min_depths = max(self.min_depth, min_depths)
            depths_choices = [d for d in range(min_depths, self.max_depth+1)]
            sampled_layers = mutate_dims(depths_choices, prob=prob)
            start_channels_id = 0 if min_channels <= 0 else self.channels_list.index(min_channels)
            channels_list = self.channels_list[start_channels_id:]

            while True:
                sampled_channels = mutate_dims(channels_list, prob=prob)
                
                assert sampled_channels%self.head_dims == 0
                if sampled_channels >= in_channels:
                    break
        
        windows_size, num_heads, qk_scale, v_scale, mlp_ratio = [], [], [], [], []
                
        if not self.layer_wise_sampling:
            first_attn_layer = self.layers[0]
            first_ffn_layer = self.layers[1]
            assert isinstance(first_attn_layer, SuperAttention) and isinstance(first_ffn_layer, SuperFFN)
            
            block_wise_attn_conf = first_attn_layer.arch_sampling(mode = mode, sampled_res=self.sampled_res, sampled_channels=sampled_channels,
                min_qk_scale=min_qk_scale, min_v_scale=min_v_scale, prob=prob
                )
            block_wise_ffn_conf = first_ffn_layer.arch_sampling(mode=mode, prob=prob)

        for i, layer in enumerate(self.layers):
            if i >= sampled_layers*2:
                break
            if isinstance(layer, SuperAttention):
                last_stage_sampled_channels=in_channels if i == 0 else -1

                sampled_windows_size, sampled_num_heads, sampled_qk_scale, sampled_v_scale = block_wise_attn_conf
                
                layer.set_conf(sampled_windows_size, sampled_num_heads, sampled_qk_scale, sampled_v_scale, sampled_channels, last_stage_sampled_channels, self.sampled_res)
                
                windows_size.append(sampled_windows_size)
                num_heads.append(sampled_num_heads)
                qk_scale.append(sampled_qk_scale)
                v_scale.append(sampled_v_scale)

            elif isinstance(layer, SuperFFN):
                sampled_mlp_ratio = block_wise_ffn_conf

                layer.set_conf(sampled_channels, sampled_mlp_ratio)

                mlp_ratio.append(sampled_mlp_ratio)
        
        self.set_conf(sampled_layers=sampled_layers, sampled_channels=sampled_channels)

        return sampled_layers, sampled_channels, windows_size, num_heads, qk_scale, v_scale, mlp_ratio

    def set_conf(self, sampled_layers, sampled_channels):
        self.sampled_channels = sampled_channels
        self.sampled_layers = sampled_layers

        return self.get_conf()

    def get_conf(self):
        return (self.sampled_layers, self.sampled_channels)
 