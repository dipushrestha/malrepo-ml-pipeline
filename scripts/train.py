#!/usr/bin/env python3
"""Training entry point for MLOps-Platform."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    logger.info(f"Config: {args.config}")
    logger.info(f"Model: {config['model']['architecture']}")
    logger.info(f"Epochs: {config['training']['epochs']}")

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        torch.manual_seed(args.seed)

        # Build model
        from src.models.registry import ModelRegistry
        model = ModelRegistry.create(
            config["model"]["architecture"],
            num_classes=config["model"]["num_classes"],
        )
        logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

        # Build data loaders
        from src.data.dataset import ImageClassificationDataset
        from src.data.transforms import get_train_transforms, get_val_transforms

        train_dataset = ImageClassificationDataset(
            config["data"]["train_dir"],
            transform=get_train_transforms(config["data"]["image_size"]),
        )
        val_dataset = ImageClassificationDataset(
            config["data"]["val_dir"],
            transform=get_val_transforms(config["data"]["image_size"]),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["data"].get("num_workers", 4),
            pin_memory=config["data"].get("pin_memory", True),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["data"].get("num_workers", 4),
        )

        # Build trainer
        from src.training.trainer import Trainer, TrainConfig
        from src.training.callbacks import MLflowCallback, PrintCallback

        train_config = TrainConfig(
            epochs=config["training"]["epochs"],
            learning_rate=config["training"]["learning_rate"],
            optimizer=config["training"]["optimizer"],
            weight_decay=config["training"].get("weight_decay", 0.01),
            mixed_precision=config["training"].get("mixed_precision", False),
            gradient_clip=config["training"].get("gradient_clip", 1.0),
            early_stopping_patience=config["training"].get("early_stopping_patience", 10),
            checkpoint_dir=config["training"].get("checkpoint_dir", "models/saved"),
        )

        trainer = Trainer(
            model=model,
            config=train_config,
            train_loader=train_loader,
            val_loader=val_loader,
            callbacks=[PrintCallback(), MLflowCallback()],
        )

        history = trainer.train()
        logger.info("Training complete.")

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Run: pip install -e '.[gpu]'")
        sys.exit(1)


if __name__ == "__main__":
    main()
