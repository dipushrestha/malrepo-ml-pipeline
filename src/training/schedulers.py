"""Learning rate scheduler utilities."""
from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine
                for base_lr in self.base_lrs
            ]


def build_scheduler(name: str, optimizer, **kwargs):
    """Build a scheduler by name."""
    schedulers = {
        "cosine": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=kwargs.get("epochs", 100),
        ),
        "warmup_cosine": lambda: WarmupCosineScheduler(
            optimizer,
            warmup_epochs=kwargs.get("warmup_epochs", 5),
            total_epochs=kwargs.get("epochs", 100),
        ),
        "step": lambda: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get("step_size", 30), gamma=0.1,
        ),
        "plateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=kwargs.get("patience", 5),
        ),
    }

    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Choose from {list(schedulers)}")
    return schedulers[name]()
