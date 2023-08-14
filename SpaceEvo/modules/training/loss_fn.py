import torch
from torch import Tensor


class NSRLoss(torch.nn.modules.loss._Loss):

    def forward(self, input: Tensor, output: Tensor) -> Tensor:
        var = torch.var(input, [2, 3])
        l2norm_channel = torch.mean((input - output) ** 2, dim=[2, 3])
        rv = l2norm_channel / var 
        return torch.mean(rv)
