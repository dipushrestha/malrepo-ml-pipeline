"""Custom loss functions for training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        return (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def build_loss(name: str, **kwargs) -> nn.Module:
    """Build a loss function by name."""
    losses = {
        "cross_entropy": nn.CrossEntropyLoss,
        "label_smoothing": lambda: LabelSmoothingLoss(
            num_classes=kwargs.get("num_classes", 1000),
            smoothing=kwargs.get("smoothing", 0.1),
        ),
        "focal": lambda: FocalLoss(
            alpha=kwargs.get("alpha", 1.0),
            gamma=kwargs.get("gamma", 2.0),
        ),
    }
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(losses)}")
    factory = losses[name]
    return factory() if callable(factory) else factory
