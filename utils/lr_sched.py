import math


def adjust_learning_rate(
    optimizer, current_epoch, base_lr, min_lr, warmup_epochs, total_epochs
):
    """Per-iteration cosine schedule with linear warmup.

    Args:
        current_epoch: float, fractional epoch (epoch + step/total_steps)
        base_lr: peak learning rate after warmup
        min_lr: minimum learning rate at the end
        warmup_epochs: number of linear warmup epochs
        total_epochs: total number of training epochs
    """
    if current_epoch < warmup_epochs:
        # linear warmup from 0 to base_lr
        lr = base_lr * current_epoch / warmup_epochs
    else:
        # cosine decay from base_lr to min_lr
        progress = (current_epoch - warmup_epochs) / max(
            total_epochs - warmup_epochs, 1
        )
        progress = min(max(progress, 0.0), 1.0)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr
