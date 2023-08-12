import math
import os

from PIL import Image
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.distributed as dist
import torchvision.transforms.functional as F
from torchvision import datasets, transforms

from .imagenet_tar_dataset import ImageTarDataset
from .transform import get_data_transform


VALID_SEED = 2147483647
    

def load_dataset(dataset_path: str, transform):
    if dataset_path.endswith('.tar'):
        return ImageTarDataset(dataset_path, transform)
    else:
        return datasets.ImageFolder(dataset_path, transform)


def build_imagenet_dataloader(dataset_path, train_batch_size, eval_batch_size, 
                              distributed=True, num_workers=8, augment='default',
                              train_crop_size=224, test_scale=256, test_crop_size=224,
                              interpolation='bicubic', auto_augment_policy='v0', valid_size=None):
    # support two types of layout:
    # 1) ['train', 'val']
    # 2) ['ILSVRC2012_img_train.tar', 'ILSVRC2012_img_val.tar]
    if 'ILSVRC2012_img_train.tar' in os.listdir(dataset_path):
        train_dir = os.path.join(dataset_path, 'ILSVRC2012_img_train.tar')
        test_dir = os.path.join(dataset_path, 'ILSVRC2012_img_val.tar')
    elif 'train' in os.listdir(dataset_path):
        train_dir = os.path.join(dataset_path, 'train')
        test_dir = os.path.join(dataset_path, 'val')
    else:
        raise RuntimeError('Fail to load imagenet dataset: unsupported dir layout ', os.listdir(dataset_path))

    #build transforms
    train_transform = get_data_transform(is_training=True, augment=augment, train_crop_size=train_crop_size, 
                                            test_scale=test_scale, test_crop_size=test_crop_size, 
                                            interpolation=interpolation, auto_augment_policy=auto_augment_policy)
    test_transform = get_data_transform(is_training=False, augment=augment, train_crop_size=train_crop_size, 
                                            test_scale=test_scale, test_crop_size=test_crop_size, 
                                            interpolation=interpolation, auto_augment_policy=auto_augment_policy)

    train_dataset = load_dataset(train_dir, train_transform)

    #build train and optionally valid_dataloader
    if valid_size is None or valid_size <= 0: # no need to split train_dataset
        if distributed:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        valid_dataloader = None

    else:
        valid_dataset = load_dataset(train_dir, test_transform)
        train_indexes, valid_indexes = random_sample_valid_set(len(train_dataset), valid_size)
        print(valid_indexes[:10]) # debug

        if distributed:
            train_sampler = MyDistributedSampler(
                train_dataset, dist.get_world_size(), dist.get_rank(), True, np.array(train_indexes)
            )
            valid_sampler = MyDistributedSampler(
                valid_dataset, dist.get_world_size(), dist.get_rank(), False, np.array(valid_indexes)
            )
        else:
            train_sampler = SubsetRandomSampler(train_indexes)
            valid_sampler = SubsetRandomSampler(valid_indexes)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=eval_batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    # build test_dataloader
    test_dataset = load_dataset(test_dir, test_transform)

    if distributed:
        test_sampler = DistributedSampler(test_dataset)
    else:
        test_sampler = None

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        sampler=test_sampler,
    )

    return train_dataloader, valid_dataloader, test_dataloader, train_sampler # return train_sampler to set epoch


def random_sample_valid_set(train_size, valid_size):
    assert train_size > valid_size

    g = torch.Generator()
    g.manual_seed(VALID_SEED)  # set random seed before sampling validation set
    rand_indexes = torch.randperm(train_size, generator=g).tolist()

    valid_indexes = rand_indexes[:valid_size]
    train_indexes = rand_indexes[valid_size:]
    return train_indexes, valid_indexes


class MyDistributedSampler(DistributedSampler):
    """Allow Subset Sampler in Distributed Training"""

    def __init__(
        self, dataset, num_replicas=None, rank=None, shuffle=True, sub_index_list=None
    ):
        super(MyDistributedSampler, self).__init__(dataset, num_replicas, rank, shuffle)
        self.sub_index_list = sub_index_list  # numpy

        self.num_samples = int(
            math.ceil(len(self.sub_index_list) * 1.0 / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        print("Use MyDistributedSampler: %d, %d" % (self.num_samples, self.total_size))

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.sub_index_list), generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        indices = self.sub_index_list[indices].tolist()
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)