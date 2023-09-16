# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
def parse_arch(arch, num_conv_stages=2):
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

def parse_supernet_configs(args, min_arch):
    conv_ratio, kernel_size, mlp_ratio, qk_scale, v_scale = parse_arch(arch=min_arch) # get stage_wise_min_arch
    min_res, min_channels, min_depths, _, _, _, _, _, _, _ = min_arch
    
    min_channels, min_depths = min_channels[1:], min_depths[1:]

    res = []
    for r in args.search_space.res:
        if r >= min_res:
            res.append(r)
    args.search_space.res = res
    args.search_space.windows_size = [[1] for _ in range(len(res))]

    for idx, (min_channel, space_channel) in enumerate(zip(min_channels, args.search_space.min_channels)):
        args.search_space.min_channels[idx] = max(min_channel, space_channel)

    for idx, (min_depth, space_depth) in enumerate(zip(min_depths, args.search_space.min_depth)):
        args.search_space.min_depth[idx] = max(min_depth, space_depth)
    
    conv_ratios = []
    for c in args.search_space.conv_ratio:
        if c >= min(conv_ratio):
            conv_ratios.append(c)
    args.search_space.conv_ratio = conv_ratios

    mlp_ratios = []
    for m in args.search_space.mlp_ratio:
        if m >= min(mlp_ratio):
            mlp_ratios.append(m)
    args.search_space.mlp_ratio = mlp_ratios

    v_scales = [[] for _ in range(len(args.search_space.v_scale))]
    for idx, stage_wise_v in enumerate(args.search_space.v_scale):
        for v in stage_wise_v:
            if v >= v_scale[idx]:
                v_scales[idx].append(v)
    args.search_space.v_scale = v_scales

def select_min_arch(flops, min_archs, flops_bound=15):
    limits = max(min_archs.keys())

    if flops > limits:
        return min_archs[limits]

    lower = 0
    arch = []
    for upper in min_archs.keys():
        if lower + 2*flops_bound < flops <= upper + 2*flops_bound:
            arch = min_archs[upper]
            break
        else:
            lower = upper

    return arch