import torch.distributed as dist

from modules.training.dataset.imagenet_dataloader import build_imagenet_dataloader


def build_data_loader(args):
    return build_imagenet_dataloader(
        dataset_path=args.dataset_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=min(args.batch_size, 32),
        distributed=dist.is_initialized(),
        num_workers=args.data_loader_workers_per_gpu,
        augment=args.augment,
        valid_size=args.valid_size
        )
