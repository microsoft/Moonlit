import argparse
import math
import os 
import random
import sys 
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, Flowers102, OxfordIIITPet, Food101, StanfordCars, FGVCAircraft

from get_subnet import get_spaceevo_int8_pretrained_subnet, replace_classifier, freeze_weight
from modules.training.dataset.transform import get_data_transform
import modules.alphanet_training.utils.loss_ops as loss_ops 
from modules.alphanet_training.utils.progress import AverageMeter, ProgressMeter, accuracy

# import wandb
# wandb.login(key="b1634222b85eb1957c717d8810aac9d485e5a538")
# wandb.init(project="attentive-nas", name="fine-grained cls")


FILE_DIR = os.path.dirname(__file__)
DATASET_DOWNLOAD_DIR = os.path.join(FILE_DIR, 'datasets')
CHECKPOINT_SAVE_PATH = os.path.join(FILE_DIR, 'checkpoints/downstream_cls')
os.makedirs(DATASET_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_SAVE_PATH, exist_ok=True)

TASK_LIST = ["CIFAR10", "CIFAR100", "Flowers102", "OxfordIIITPet", "Food101", "StanfordCars", "FGVCAircraft"]
LR_LIST = [0.0125, 0.0125, 0.15, 0.00375, 0.15, 0.15, 0.075]
WD_LIST = [4e-5, 4e-5, 4e-6, 4e-6, 4e-8, 0, 4e-8]
NCLASSES_LIST = [10, 100, 102, 37, 101, 196, 102]


def get_lr_wd_from_dataset(dataset_name):
    name2idx = {jj: ii for ii, jj in enumerate(TASK_LIST)}
    i = name2idx[dataset_name]
    return LR_LIST[i], NCLASSES_LIST[i]

def get_num_classes_from_dataset(dataset_name):
    name2idx = {jj: ii for ii, jj in enumerate(TASK_LIST)}
    i = name2idx[dataset_name]
    return NCLASSES_LIST[i]


def linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=7. / 16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    correct, total = 0, 0
    with tqdm(dataloader, total=len(dataloader), ncols=80, desc="Eval") as t:
        for inputs, targets in t:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            batch_size = inputs.size(0)
            top1.update(acc1, batch_size)
            top5.update(acc5, batch_size)
 
            total += targets.size(0)
            t.set_postfix_str(f'Acc1 {top1.avg:.2f} Acc5 {top5.avg:.2f}, {top1.sum/top1.count}')
    metric = {
        "Acc@1": top1.avg,
        "Acc@5": top5.avg,
        "Top1": top1.sum/top1.count,
        "Top5": top5.sum/top5.count,
    }

    # wandb.log(metric)   
    return top1.avg


def build_data_loader(args):
    train_transform = get_data_transform(is_training=True, augment='auto_augment_tf')
    test_transform = get_data_transform(is_training=False, augment='auto_augment_tf')
    
    if "CIFAR" in args.dataset:
        train_set = eval(args.dataset)("./datasets", download=True, train=True, transform=train_transform)
        valid_set = eval(args.dataset)("./datasets", download=True, train=False, transform=test_transform)
    else:
        split = "trainval" if args.dataset in ["FGVCAircraft", "OxfordIIITPet"] else "train"
        train_set = eval(args.dataset)("./datasets", download=True, split=split, transform=train_transform)
        valid_set = eval(args.dataset)("./datasets", download=True, split="test", transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, valid_loader


def train(args, model, train_loader, device, loss_func, optimizer, lr_scheduler, valid_loader):
    for epoch in range(1, int(args.epochs)+1):
        model.train()
        with tqdm(train_loader, total=len(train_loader), ncols=80, desc=f"Train({epoch})") as t:
            for step, (inputs, targets) in enumerate(t):
                inputs, targets = inputs.to(device), targets.to(device)

                logits = model(inputs)

                loss = loss_func(logits, targets)
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                lr_scheduler.step()
                lr = lr_scheduler.get_last_lr()[0]
                
                t.set_postfix({"loss": round(loss.detach().item(), 3),
                "lr": round(lr, 3)
                })
        evaluate(model, valid_loader, device)


def get_model(args, device):
    model = get_spaceevo_int8_pretrained_subnet(args.subnet_name, args.imagenet_path, device=device)
    if args.only_train_head:
        freeze_weight(model)
    replace_classifier(model, get_num_classes_from_dataset(args.dataset))
    
    model.to(device)
    model.train()
    return model 


def main(args):
    device = torch.device(args.device)

    train_loader, valid_loader = build_data_loader(args)

    model = get_model(args, device)

    no_decay = ["bias", 'bn']
    model_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        }, {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]
    lr, wd = get_lr_wd_from_dataset(args.dataset)

    # get optimizer
    # optimizer = optim.AdamW(model_params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = torch.optim.RMSProp(params, lr=0.01, beta=0.9, beta2=0.9, eps=0.001)
    optimizer = torch.optim.SGD(model_params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)

    # get lr_scheduler
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2) #learning rate decay
    # lr_scheduler = cosine_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.lr_warmup*num_steps_per_epoch, num_training_steps=args.epochs*num_steps_per_epoch
    # )
    num_steps_per_epoch = max(len(train_loader), 1)
    lr_scheduler = linear_schedule_with_warmup(optimizer, num_warmup_steps=args.lr_warmup*num_steps_per_epoch, num_training_steps=args.epochs*num_steps_per_epoch)

    ce_loss = loss_ops.CrossEntropyLossSmooth(args.label_smoothing).to(device)

    train(args, model, train_loader, device, ce_loss, optimizer, lr_scheduler, valid_loader)

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_SAVE_PATH, f'{args.subnet_name}_{args.dataset}.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SpaceEvo Downstream Classification Finetune")
    parser.add_argument('--subnet_name', type=str, default='SEQnet@vnni-A0')
    parser.add_argument('--imagenet_path', required=True, help='used to calibrate batchnorm when generating subnet')
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=TASK_LIST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr_warmup", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--only_train_head', action='store_true', help='only train the classifier head and freeze the other layers')
    args = parser.parse_args()

    # seed all
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True

    main(args)
