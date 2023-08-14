from torch import nn


def make_divisible(v, divisor=8, min_val=None):
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def get_same_padding(kernel_size: int) -> int:
    return kernel_size // 2


def get_activation(activation):
    name_module_pairs = [
        ('relu', nn.ReLU), ('relu6', nn.ReLU6), ('hswish', nn.Hardswish), ('hsigmoid', nn.Hardsigmoid), ('swish', nn.SiLU), ('none', None)
    ]
    if not isinstance(activation, str):
        for name, module in name_module_pairs:
            if activation == module:
                return name, module
        return 'unknown', module
    else:
        for name, module in name_module_pairs:
            if activation == name:
                return name, module
        raise ValueError(f'unrecognized activation {activation}')