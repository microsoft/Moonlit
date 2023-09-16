# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
BN layers for Transformer/Conv
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_connection(inputs, p, is_training, transformer=False):
    assert 0 <= p <= 1, 'p must be in range of [0,1]'
    if not is_training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1.0 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob

    random_tensor += torch.rand([batch_size, 1, 1, 1] if not transformer else [
                                batch_size, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


class LNSuper(nn.LayerNorm):
    def __init__(self, super_channels, inverse_sampling=False, init=1) -> None:
        super().__init__(super_channels)

        self.super_channels = super_channels

        self.activated_channels = -1
        self.inverse_sampling = inverse_sampling
        nn.init.constant_(self.weight, init)
        nn.init.constant_(self.bias, 0)

    def set_conf(self, activated_channels):
        pass  # use dynamic channel here
        # self.activated_channels = activated_channels

    def forward(self, x):
        return F.layer_norm(x, x.shape[1:], self.weight[:x.shape[-1]].expand(x.shape[1:]), self.bias[:x.shape[-1]].expand(x.shape[1:]))


class BNSuper1d(nn.BatchNorm1d):
    def __init__(self, super_channels, inverse_sampling=False, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True, init=1):
        super().__init__(super_channels if isinstance(super_channels, int)
                         else sum(super_channels), eps, momentum, affine, track_running_stats)
        self.activated_channels = -1

        self.inverse_sampling = inverse_sampling
        self.bn_calibration = False

        if isinstance(super_channels, int):
            self.super_channels = super_channels
        else:
            self.max_qk_dim = super_channels[0]
            self.max_v_dim = super_channels[1]

        nn.init.constant_(self.weight, init)
        nn.init.constant_(self.bias, 0)

    def set_conf(self, activated_channels):
        self.activated_channels = activated_channels

    def _get_tensor(self, tensor):
        t1 = tensor[: self.activated_channels[0]]
        t2 = tensor[self.max_qk_dim//2: self.max_qk_dim //
                    2 + self.activated_channels[0]]
        t3 = tensor[self.max_qk_dim: self.max_qk_dim +
                    self.activated_channels[1]]
        return torch.cat([t1, t2, t3], dim=0)

    def forward(self, x):
        if isinstance(self.activated_channels, (list, tuple)):
            o = self.activated_channels[0]
            running_mean = self.running_mean[:sum(self.activated_channels)+o]
            running_var = self.running_var[:sum(self.activated_channels)+o]
            weight = self._get_tensor(self.weight)
            bias = self._get_tensor(self.bias)
        else:
            running_mean = self.running_mean[:self.activated_channels]
            running_var = self.running_var[:self.activated_channels]
            weight = self.weight[:self.activated_channels]
            bias = self.bias[:self.activated_channels]

        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1

        if self.momentum is None:
            exp = 1 / float(self.num_batches_tracked)
        else:
            exp = self.momentum

        return F.batch_norm(x.flatten(0, 1), running_mean, running_var, weight, bias, self.training, exp, self.eps).reshape_as(x)


class BNSuper2d(nn.BatchNorm2d):
    def __init__(self, super_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True, inverse_sampling=False, init=1):
        super().__init__(super_channels if isinstance(super_channels, int)
                         else sum(super_channels), eps, momentum, affine, track_running_stats)
        self.activated_channels = -1

        self.inverse_sampling = inverse_sampling
        self.bn_calibration = False

        if isinstance(super_channels, int):
            self.super_channels = super_channels
        else:
            self.max_qk_dim = super_channels[0]
            self.v_qk_dim = super_channels[1]

        nn.init.constant_(self.weight, init)
        nn.init.constant_(self.bias, 0)

    def set_conf(self, activated_channels):
        self.activated_channels = activated_channels

    def _get_tensor(self, tensor):
        # offset = self.super_channels//3
        t1 = tensor[: self.activated_channels[0]]
        t2 = tensor[self.max_qk_dim//2: self.max_qk_dim //
                    2 + self.activated_channels[0]]
        t3 = tensor[self.max_qk_dim: self.max_qk_dim +
                    self.activated_channels[1]]
        return torch.cat([t1, t2, t3], dim=0)

    def forward(self, x):
        running_mean = self.running_mean[:self.activated_channels]
        running_var = self.running_var[:self.activated_channels]
        weight = self.weight[:self.activated_channels]
        bias = self.bias[:self.activated_channels]

        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1

        if self.momentum is None:
            exp = 1 / float(self.num_batches_tracked)
        else:
            exp = self.momentum

        return F.batch_norm(x,
                            running_mean,
                            running_var,
                            weight,
                            bias,
                            self.training, exp, self.eps
                            )
