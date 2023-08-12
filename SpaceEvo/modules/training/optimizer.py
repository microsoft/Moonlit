import torch
import torchvision


def get_optimizer(opt_name: str, lr, parameters, momentum, weight_decay) -> torch.optim.Optimizer:
    opt_name = opt_name.lower()
    if opt_name.startswith('sgd'):
        return torch.optim.SGD(
            parameters, 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay,
            nesterov='nesterov' in opt_name
        )
    if opt_name == 'adam':
        return torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    raise ValueError(f'{opt_name} not supported')