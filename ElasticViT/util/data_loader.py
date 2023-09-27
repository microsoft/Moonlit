# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import numpy as np
import torch.utils.data
import torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from timm.data import create_transform

from PIL import Image


def load_data_dist(cfg, searching_set=False):
    assert cfg.dataset == 'imagenet'
    normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    traindir = os.path.join(cfg.path, 'train')
    valdir = os.path.join(cfg.path, 'val')
    print("Train dir:", traindir)

    aug_type = getattr(cfg, 'aug_type', 'none')
    if aug_type == 'auto':
        transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=cfg.aug.color_jitter,
            auto_augment=cfg.aug.aa,
            interpolation=cfg.aug.train_interpolation,
            re_prob=cfg.aug.reprob,
            re_mode=cfg.aug.remode,
            re_count=cfg.aug.recount,
        )

        train_set = datasets.ImageFolder(
            traindir,
            transform=transform
        )
    else:
        train_set = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.workers, pin_memory=False, sampler=train_sampler, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=cfg.batch_size*2, shuffle=False,
        num_workers=cfg.workers, pin_memory=False, drop_last=False)

    if searching_set:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(cfg.path, 'search'), transforms.Compose([
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=cfg.batch_size*2, shuffle=False,
            num_workers=cfg.workers, pin_memory=False, drop_last=False)
    else:
        val_loader = test_loader

    return train_loader, val_loader, test_loader, train_sampler