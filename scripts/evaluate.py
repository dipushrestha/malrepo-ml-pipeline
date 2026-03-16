#!/usr/bin/env python3
"""Evaluate a trained model on the test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="experiments/eval_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    logger.info(f"Evaluating checkpoint: {args.checkpoint}")

    try:
        import torch
        from torch.utils.data import DataLoader
        from src.data.dataset import ImageClassificationDataset
        from src.data.transforms import get_val_transforms
        from src.evaluation.metrics import compute_all_metrics

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        checkpoint = torch.load(args.checkpoint, map_location=device)
        from src.models.registry import ModelRegistry
        model = ModelRegistry.create(
            config["model"]["architecture"],
            num_classes=config["model"]["num_classes"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # Load test data
        test_dataset = ImageClassificationDataset(
            config["data"]["test_dir"],
            transform=get_val_transforms(config["data"]["image_size"]),
        )
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # Run evaluation
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        metrics = compute_all_metrics(y_true, y_pred, config["model"]["num_classes"])

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2))

        logger.info(f"Results: {metrics}")
        logger.info(f"Saved to {output_path}")

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")


if __name__ == "__main__":
    main()
