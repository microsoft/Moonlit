'''
Implementation of LSQ+ OPs
'''
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


DEFAULT_BITS = 8
FLOAT32_BITS = 32

EPSLION = torch.tensor(1e-7, dtype=torch.float32) # avoid divide by zero


def gradscale(x, scale):
    yOut = x
    yGrad = x * scale
    y = (yOut - yGrad).detach() + yGrad
    return y


def roundpass(x):
    yOut = x.round()
    yGrad = x
    y = (yOut - yGrad).detach() + yGrad
    return y


def quantize_activation(v, s, bits, beta, training):
    if bits == FLOAT32_BITS: # skip quant if bits==32 (float32 inference)
        return v
    Qn = 0
    Qp = 2 ** bits - 1
    # get the #features of single element, divide batch_size
    nfeatures = v.nelement() / v.shape[0]
    gradScaleFactor = 1 / math.sqrt(nfeatures * Qp)
    if training:
        s = gradscale(s, gradScaleFactor)
        beta = gradscale(beta, gradScaleFactor)
    v = (v - beta) / s.abs()
    v = F.hardtanh(v, Qn, Qp)
    if training:
        vbar = roundpass(v)
    else:
        vbar = v.round()
    vhat = vbar * s.abs() + beta
    return vhat


def quantize_weight(v, s, bits, training):
    if bits == FLOAT32_BITS: # skip quant if bits==32 (float32 inference)
        return v
    Qn = -2 ** (bits - 1)
    Qp = 2 ** (bits - 1) - 1
    if training:
        nweight = v.nelement()
        gradScaleFactor = 1 / math.sqrt(nweight * Qp)
        s = gradscale(s, gradScaleFactor)
    v = v / s.abs()
    v = F.hardtanh(v, Qn, Qp)
    if training:
        vbar = roundpass(v)
    else:
        vbar = v.round()
    vhat = vbar * s.abs()
    return vhat


class QBase:

    def __init__(self, no_weight=False) -> None:
        self.set_fp32_mode()
        self.no_weight = no_weight
        if not self.no_weight:
            self.sw = nn.Parameter(torch.tensor([1], dtype=torch.float32))
        self.sa = nn.Parameter(torch.tensor([1], dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor([0], dtype=torch.float32))
        self.init_flag = False
        
    @staticmethod
    def _min_max_init_scale(scale: Tensor, v: Tensor, nbits: int, is_weight):
        if is_weight:
            scale.data.copy_(max(2 * (v.abs().max().detach(), EPSLION)) / (2 ** nbits - 1))
        else:
            scale.data.copy_((max(v.max().detach() - v.min().detach(), EPSLION)) / (2 ** nbits - 1))
        if dist.is_initialized():
            dist.all_reduce(scale.data, dist.ReduceOp.SUM)
            scale.data /= dist.get_world_size()

    @staticmethod
    def _lsq_init_scale(scale: Tensor, v: Tensor, Qp: int):
        scale.data.copy_(2. / math.sqrt(Qp) * v.abs().mean())
        if dist.is_initialized():
            dist.all_reduce(scale, dist.ReduceOp.SUM)
            scale.data /= dist.get_world_size()

    def get_quant_x_w(self, x: Tensor, w: Tensor):
        if self.nbits_a == FLOAT32_BITS:
            return x, w
            
        if self.init_flag == False:
            if self.sa.data != 1 or self.beta.data != 0 or (not self.no_weight and self.sw.data != 1): # scale and beta loaded form state_dict
                self.init_flag = True
            else:
                self._min_max_init_scale(self.sa, x, self.nbits_a, is_weight=False)

                # initialize beta to be min(x). TODO: To be changed.
                self.beta.data.copy_(x.min().detach())
                if dist.is_initialized():
                    dist.all_reduce(self.beta.data, dist.ReduceOp.MIN)

                if not self.no_weight:
                    self._min_max_init_scale(self.sw, w, self.nbits_w, is_weight=True)

                self.init_flag = True 

        with torch.no_grad():
            if self.sa < EPSLION / (2 ** self.nbits_a - 1):
                self.sa.copy_(EPSLION / (2 ** self.nbits_a - 1))
            if self.sw < EPSLION / (2 ** self.nbits_w - 1):
                self.sw.copy_(EPSLION / (2 ** self.nbits_w - 1))
                
        xq = quantize_activation(x, self.sa, self.nbits_a, beta=self.beta, training=self.training)
        if not self.no_weight:
            wq = quantize_weight(w, self.sw, self.nbits_w, training=self.training)
        else:
            wq = None
        return xq, wq

    def set_fp32_mode(self):
        self.nbits_w = FLOAT32_BITS
        self.nbits_a = FLOAT32_BITS

    def set_quant_mode(self, nbits=DEFAULT_BITS):
        assert nbits < FLOAT32_BITS
        self.nbits_w = nbits
        self.nbits_a = nbits

    def set_lsq_state_dict(self, prefix: str, rv: dict):
        if not self.no_weight:
            rv[prefix + 'sw'] = self.sw.data
        rv[prefix + 'sa'] = self.sa.data 
        rv[prefix + 'beta'] = self.beta.data 


def set_quant_mode(model):
    def set_quant_mode_recursive(m: nn.Module):
        for c in m.children():
            if len(list(m.children())) > 0:
                set_quant_mode_recursive(c)

            if isinstance(c, QBase):
                c.set_quant_mode()
    set_quant_mode_recursive(model)