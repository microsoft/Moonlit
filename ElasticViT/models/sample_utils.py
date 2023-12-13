# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.import random
import numpy as np
import random

def mutate_dims(choices, prob=1):
    assert 0. <= prob <= 1.

    if random.random() < prob:
        return random.choice(choices)
    else:
        return min(choices)

def softmax(x, to_list: bool = False):
    y = np.exp(x) / np.sum(np.exp(x), axis=0)

    if to_list:
        y = y.tolist()
    return y

# convert the layer-wise sampling arch config to block wise
def parse_arch(arch, num_conv_stages):
    num_conv_stages
    arch = [list(a) if isinstance(a, tuple) else a for a in arch]

    res, channels, depths, conv_ratios, kernel_sizes, mlp_ratio, num_heads, windows_size, qk_scale, v_scale = arch

    parse_conv_ratio, parse_kernel_size, parse_mlp_ratio, parse_qk_scale, parse_v_scale = [], [], [], [], []
    for stage_index, d in enumerate(depths[1:]):
        stage_index += 1

        if stage_index >= num_conv_stages+1:
            offset = sum(depths[num_conv_stages+1:stage_index])

            parse_mlp_ratio.append(mlp_ratio[offset])
            parse_qk_scale.append(qk_scale[offset])
            parse_v_scale.append(v_scale[offset])

        else:
            offset = sum(depths[:stage_index])

            parse_conv_ratio.append(conv_ratios[offset])
            parse_kernel_size.append(kernel_sizes[offset])

    return parse_conv_ratio, parse_kernel_size, parse_mlp_ratio, parse_qk_scale, parse_v_scale

def arch_slicing(arch, min_arch, model_without_ddp, all_dim=False):
    arch = [list(a) if isinstance(a, tuple) else a for a in arch]
    min_res, min_channels, min_depths, min_conv_ratios, min_kernel_sizes, min_mlp_ratio, _, _, min_qk_scale, min_v_scale = min_arch
    res, channels, depths, conv_ratios, kernel_sizes, mlp_ratio, num_heads, windows_size, qk_scale, v_scale = arch
    if res < min_res:
        res = min_res

    new_res_idx = model_without_ddp.res_range.index(res)
    num_conv_stages = model_without_ddp.num_conv_stages

    for stage_index, (min_c, c) in enumerate(zip(min_channels[1:], channels[1:])):
        if c < min_c:
            channels[stage_index+1] = min_c

    num_heads, windows_size = [], []
    for stage_index, (min_d, d) in enumerate(zip(min_depths[1:], depths[1:])):
        reduced_layers = 0
        stage_index += 1

        if d < min_d:
            reduced_layers = min_d - d
            depths[stage_index] = min_d

        if stage_index >= num_conv_stages+1:
            offset = sum(depths[num_conv_stages+1:stage_index])
            min_offset = sum(min_depths[num_conv_stages+1:stage_index])

            windows_size += [model_without_ddp.feature_exactor[stage_index -
                                                    1].windows_size[new_res_idx][0]] * depths[stage_index]
            num_heads += [channels[stage_index] //
                            model_without_ddp.head_dims] * depths[stage_index]

            for _ in range(reduced_layers):
                mlp_ratio.insert(offset, mlp_ratio[offset])
                qk_scale.insert(offset, qk_scale[offset])
                v_scale.insert(offset, v_scale[offset])

            if all_dim:
                current_mlp_ratio_val, min_mlp_ratio_val = mlp_ratio[
                    offset], min_mlp_ratio[min_offset]
                current_qk_scale_val, min_qk_scale_val = qk_scale[offset], min_qk_scale[min_offset]
                current_v_scale_val, min_v_scale_val = v_scale[offset], min_v_scale[min_offset]
                for idx in range(depths[stage_index]):
                    mlp_ratio[offset +
                                idx] = max(current_mlp_ratio_val, min_mlp_ratio_val)
                    qk_scale[offset +
                                idx] = max(current_qk_scale_val, min_qk_scale_val)
                    v_scale[offset +
                            idx] = max(current_v_scale_val, min_v_scale_val)
        else:
            offset = sum(depths[:stage_index])
            min_offset = sum(min_depths[:stage_index])

            for _ in range(reduced_layers):
                conv_ratios.insert(offset, conv_ratios[offset])
                kernel_sizes.insert(offset, kernel_sizes[offset])
            if all_dim:
                current_conv_ratio_val, min_conv_ratio_val = conv_ratios[
                    offset], min_conv_ratios[min_offset]
                current_kernel_size_val, min_kernel_size_val = kernel_sizes[
                    offset], min_kernel_sizes[min_offset]
                for idx in range(depths[stage_index]):
                    conv_ratios[offset +
                                idx] = max(current_conv_ratio_val, min_conv_ratio_val)
                    kernel_sizes[offset +
                                    idx] = max(current_kernel_size_val, min_kernel_size_val)

    return res, tuple(channels), tuple(depths), tuple(conv_ratios), tuple(kernel_sizes), tuple(mlp_ratio), tuple(num_heads), tuple(windows_size), tuple(qk_scale), tuple(v_scale)
