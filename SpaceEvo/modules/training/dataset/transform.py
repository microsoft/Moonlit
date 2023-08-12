# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import numpy as np
import torchvision.transforms as transforms

from .auto_augment_tf import (
    auto_augment_policy,
    AutoAugment,
)

IMAGENET_PIXEL_MEAN = [123.675, 116.280, 103.530] 
IMAGENET_PIXEL_STD = [58.395, 57.12, 57.375]


def get_data_transform(is_training, augment='default', train_crop_size=224, test_scale=256, test_crop_size=224, interpolation='bicubic', auto_augment_policy='v0'):
    da_args = {
        'train_crop_size': train_crop_size,
        'test_scale': test_scale,
        'test_crop_size': test_crop_size,
        'interpolation': transforms.InterpolationMode.BICUBIC if interpolation == 'bicubic' else transforms.InterpolationMode.BILINEAR
    }

    if augment == 'default':
        return build_default_transform(is_training, **da_args)
    elif augment == 'auto_augment_tf':
        return build_imagenet_auto_augment_tf_transform(is_training, policy=auto_augment_policy, **da_args)
    else:
        raise ValueError(augment)


def get_normalize():
    normalize = transforms.Normalize(
        mean=torch.Tensor(IMAGENET_PIXEL_MEAN) / 255.0,
        std=torch.Tensor(IMAGENET_PIXEL_STD) / 255.0,
    )
    return normalize


def build_default_transform(
    is_training, train_crop_size=224, test_scale=256, test_crop_size=224, interpolation=transforms.InterpolationMode.BICUBIC
):
    normalize = get_normalize()
    if is_training:
        ret = transforms.Compose(
            [
                transforms.RandomResizedCrop(train_crop_size, interpolation=interpolation),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        ret = transforms.Compose(
            [
                transforms.Resize(test_scale, interpolation=interpolation),
                transforms.CenterCrop(test_crop_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return ret


def build_imagenet_auto_augment_tf_transform(
    is_training, policy='v0', train_crop_size=224, test_scale=256, test_crop_size=224, interpolation=transforms.InterpolationMode.BICUBIC
):

    normalize = get_normalize()
    img_size = train_crop_size
    aa_params = {
        "translate_const": int(img_size * 0.45),
        "img_mean": tuple(round(x) for x in IMAGENET_PIXEL_MEAN),
    }

    aa_policy = AutoAugment(auto_augment_policy(policy, aa_params))

    if is_training:
        ret = transforms.Compose(
            [
                transforms.RandomResizedCrop(train_crop_size, interpolation=interpolation),
                transforms.RandomHorizontalFlip(),
                aa_policy,
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        ret = transforms.Compose(
            [
                transforms.Resize(test_scale, interpolation=interpolation),
                transforms.CenterCrop(test_crop_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return ret

