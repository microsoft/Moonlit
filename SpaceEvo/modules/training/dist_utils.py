import copy
import os
import torch
import torch.distributed as dist


def is_master_proc():
    return not dist.is_initialized() or dist.get_rank() == 0


def dist_print(*args, **kwargs):
    if is_master_proc():
        print(*args, **kwargs)


class DistributedAvgMetric(object):

    def __init__(self, name=None):
        self.name = name
        self.sum = torch.zeros(1)[0]
        self.count = torch.zeros(1)[0]

    def update(self, val: torch.Tensor, n=1):
        self.sum += val.detach().cpu() * n
        self.count += torch.tensor(n)

    @property
    def local_avg(self):
        return (self.sum / self.count).item()

    @property
    def avg(self):
        sum = copy.deepcopy(self.sum)
        count = copy.deepcopy(self.count)
        if dist.is_initialized():
            sum = sum.to(torch.device('cuda', dist.get_rank()))
            count = count.to(torch.device('cuda', dist.get_rank()))
            dist.all_reduce(sum)
            dist.all_reduce(count)
        return (sum / count).item()


def save_on_master(checkpoint, path):
    if is_master_proc():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
