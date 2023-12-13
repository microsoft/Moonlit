# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import timm
import torch
import os
import pickle
import warnings
from . import FusedSuperNet
from .common_ops import BNSuper1d, LNSuper

def load_offline_models(use_latency, bank_flops_ranges, lib_dir, min_archs):
    offline_archs = {}

    if lib_dir is not None:
        if not use_latency:
            for pf in bank_flops_ranges:
                path = f"{lib_dir}/flops_{pf}.pkl"
                with open(path, "rb") as f:
                    archs = pickle.load(f)
                print(f"load offline archs for {pf}M bank, num: {len(archs)}")
                offline_archs[pf] = archs
        else:
            for pf in bank_flops_ranges:
                path = f"{lib_dir}/latency_{pf}.pkl"
                with open(path, "rb") as f:
                    archs = pickle.load(f)
                print(f"load offline archs for {pf}ms bank, num: {len(archs)}")
                offline_archs[pf] = archs

        min_filename = f"{lib_dir}/min.pkl"
        with open(min_filename, "rb") as f:
            archs = pickle.load(f)
    #     print(archs)
        print(f"load min archs, num: {len(archs)}")
        offline_archs['min'] = archs
    else:
        warnings.warn("offline models are not loaded, training could be very slow.")
        assert len(min_archs) > 0, 'min models are not specified.'
        offline_archs['min'] = min_archs

    return offline_archs

def build_supernet(args):
    classifier_mul = getattr(args.search_space, 'classifier_mul', 4)
    dw_downsampling = getattr(args.search_space, 'dw_downsampling', False)

    use_latency = getattr(args.search_space, 'use_latency', False)
    small_bank_choice = []
    latency_bound = 0
    if use_latency == True:
        small_bank_choice = getattr(
            args.search_space, 'small_bank_choices', [])
        latency_bound = getattr(args.search_space, 'latency_bound', 1.5)
    limit_flops = getattr(args.search_space, 'limit_flops', 800)
    min_flops = getattr(args.search_space, 'min_flops', 200)
    sample_prob = getattr(args.search_space, 'sample_prob', 0.5)
    head_dims = getattr(args.search_space, 'head_dims', 8)
    bank_size = getattr(args.search_space, 'bank_size', 150)
    bank_step = getattr(args.search_space, 'bank_step', 100)
    flops_bound = getattr(args.search_space, 'flops_bound', 15)
    sampling_rate_inc = getattr(args.search_space, 'sampling_rate_inc', 0.15)
    bank_sampling_method = getattr(
        args.search_space, 'bank_sampling_method', 'weighted_prob')
    bank_sampling_rate = getattr(args.search_space, 'bank_sampling_rate', 0.5)
    flops_sampling_method = getattr(
        args.search_space, 'flops_sampling_method', 'random')
    model_sampling_method = getattr(
        args.search_space, 'model_sampling_method', 'preference')

    swin = getattr(args.search_space, 'swin', True)
    assert not swin, 'public code does not support swin self-attention'

    use_min_model = getattr(args.search_space, 'use_min_model', True)
    use_res_specific_RPN = getattr(
        args.search_space, 'res_specific_rpn', False)
    lib_data_dir = getattr(args, 'lib_dir', None)
    offline_models_dir = None
    if lib_data_dir is not None:
        offline_models_dir = f"{lib_data_dir}/{getattr(args, 'offline_models_dir', 'offline_models/')}"

    small_bank = getattr(args.search_space, 'small_bank', False)
    small_limit_flops, small_min_flops, small_step_size = 0, 0, 0

    if small_bank:
        small_limit_flops = getattr(
            args.search_space, 'small_limit_flops', 200)
        small_min_flops = getattr(args.search_space, 'small_min_flops', 50)
        small_step_size = getattr(args.search_space, 'small_step_size', 50)

    big_bank = getattr(args.search_space, 'big_bank', False)
    big_bank_choice = []
    big_bank_size = bank_size
    if big_bank:
        big_bank_choice = getattr(args.search_space, 'big_bank_choices', [])
        big_bank_size = getattr(args.search_space, 'big_bank_size', 50)
        assert len(big_bank_choice) > 0
    
    if not use_latency:
        bank_flops_ranges = [c for c in range(min_flops, limit_flops+1, bank_step)]
        if small_bank: 
            small_bank = [c for c in range(
                small_min_flops, small_limit_flops+1, small_step_size)]
            bank_flops_ranges = small_bank + bank_flops_ranges

        if len(big_bank_choice) > 0:
            bank_flops_ranges = bank_flops_ranges + big_bank_choice
    else:
        bank_flops_ranges = [c for c in small_bank_choice]
        if len(big_bank_choice) > 0:
            bank_flops_ranges = bank_flops_ranges+big_bank_choice
        flops_bound = latency_bound
    
    flops_tables = load_offline_models(use_latency=use_latency, bank_flops_ranges=bank_flops_ranges, lib_dir=offline_models_dir, min_archs=args.search_space.min_archs)

    hard_distillation = getattr(args, 'hard_distillation', False)
    force_random = getattr(args.search_space, 'force_random', False)

    norm_layer = BNSuper1d if args.norm_layer == 'BN' else LNSuper
    
    model = FusedSuperNet(args.input_size, args.super_stem_channels, args.dataloader.num_classes, args.search_space.stage,
                          args.search_space.max_depth, args.search_space.conv_ratio, args.search_space.mlp_ratio, args.search_space.num_heads,
                          args.search_space.min_depth, args.search_space.windows_size, args.search_space.max_channels, args.search_space.min_channels,
                          args.search_space.qk_scale, args.search_space.v_scale, args.head_type, args.classifer_head_dim, args.head_dropout_prob, args.pre_norm, norm_layer, args.search_space.res, args.search_space.downsampling, bank_flops_ranges, args.search_space.se, classifier_mul=classifier_mul, dw_downsampling=dw_downsampling, 
                          sample_prob=sample_prob, head_dims=head_dims, bank_sampling_method=bank_sampling_method, bank_size=bank_size, bank_sampling_rate=bank_sampling_rate, flops_bound=flops_bound,
                          flops_sampling_method=flops_sampling_method, model_sampling_method=model_sampling_method, use_res_specific_RPN=use_res_specific_RPN, 
                          big_bank_size=big_bank_size, max_importance_comparision_num=5, use_latency=use_latency, latency_bound=latency_bound, flops_tables=flops_tables
                        )

    print(f"training mode: {flops_sampling_method}, {model_sampling_method}")

    return model, lib_data_dir, sampling_rate_inc

def build_teachers(args, pretrained_teacher_path):
    teacher_model1_path = f'{pretrained_teacher_path}/t1.pth'
    exist_t1 = os.path.exists(teacher_model1_path)
    teacher_model = timm.create_model(args.teacher_model, pretrained=not exist_t1)
    if exist_t1:
        checkpoint = torch.load(teacher_model1_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint, strict=True)

    teacher_model.cuda()
    teacher_model = torch.nn.parallel.DistributedDataParallel(
        teacher_model, device_ids=[args.local_rank])

    if hasattr(args, 'teacher_model_2'):
        teacher_model2_path = f'{pretrained_teacher_path}/t2.pth'
        exist_t2 = os.path.exists(teacher_model2_path)
        teacher_model_2 = timm.create_model(
            args.teacher_model_2, pretrained=not exist_t2)
        if exist_t2:
            checkpoint = torch.load(teacher_model2_path, map_location='cpu')
            teacher_model_2.load_state_dict(checkpoint, strict=True)
        
        teacher_model_2.cuda()
        teacher_model_2 = torch.nn.parallel.DistributedDataParallel(
            teacher_model_2, device_ids=[args.local_rank])
        teacher_model = [teacher_model, teacher_model_2]

    return teacher_model
