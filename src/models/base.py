"""Base model class for all architectures."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base class for all model architectures."""

    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        ...

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str | Path, extra: Optional[Dict] = None) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "architecture": self.__class__.__name__,
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, **kwargs) -> "BaseModel":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(num_classes=checkpoint["num_classes"], **kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
