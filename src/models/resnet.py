"""ResNet model implementation."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.base import BaseModel


class ResNetClassifier(BaseModel):
    """ResNet-based image classifier with configurable backbone.

    Supports resnet18, resnet34, resnet50, resnet101, resnet152.
    """

    BACKBONES = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
    }

    def __init__(
        self,
        num_classes: int = 1000,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__(num_classes=num_classes)
        self.backbone_name = backbone
        self.dropout_rate = dropout

        if backbone not in self.BACKBONES:
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                f"Choose from {list(self.BACKBONES.keys())}"
            )

        try:
            import torchvision.models as models

            weights = "DEFAULT" if pretrained else None
            self.backbone = getattr(models, backbone)(weights=weights)
            feature_dim = self.BACKBONES[backbone]

            # Replace classifier head
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feature_dim, num_classes),
            )
        except ImportError:
            # Fallback: simple conv net for testing without torchvision
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
