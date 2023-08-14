import torch


def get_lr_scheduler(scheduler_name: str, optimizer, step_size=None, gamma=None, epochs=None, T_max=None) -> torch.optim.lr_scheduler._LRScheduler:
    scheduler_name = scheduler_name.lower()
    if scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    if scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=T_max)