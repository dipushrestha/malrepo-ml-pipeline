"""Data preprocessing pipeline for image datasets."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def preprocess_dataset(raw_dir: str | Path, output_dir: str | Path,
                       image_size: int = 224, train_split: float = 0.8,
                       val_split: float = 0.1, seed: int = 42) -> dict:
    """Preprocess raw image dataset into train/val/test splits."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    rng = np.random.RandomState(seed)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    stats = {"train": 0, "val": 0, "test": 0}
    classes = sorted(d.name for d in raw_dir.iterdir() if d.is_dir())
    logger.info(f"Found {len(classes)} classes in {raw_dir}")

    for cls_name in classes:
        images = sorted(
            p for p in (raw_dir / cls_name).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        indices = rng.permutation(len(images))
        n_train = int(len(images) * train_split)
        n_val = int(len(images) * val_split)
        splits = {
            "train": indices[:n_train],
            "val": indices[n_train:n_train + n_val],
            "test": indices[n_train + n_val:],
        }
        for split_name, split_indices in splits.items():
            split_dir = output_dir / split_name / cls_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for idx in split_indices:
                try:
                    img = Image.open(images[idx]).convert("RGB")
                    img = img.resize((image_size, image_size), Image.LANCZOS)
                    img.save(split_dir / images[idx].name, quality=95)
                    stats[split_name] += 1
                except Exception as e:
                    logger.warning(f"Failed to process {images[idx]}: {e}")

    logger.info(f"Preprocessing complete: {stats}")
    return stats


def validate_dataset(data_dir: str | Path) -> dict:
    """Validate a processed dataset directory."""
    data_dir = Path(data_dir)
    report = {"valid": True, "errors": [], "stats": {}}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            report["valid"] = False
            report["errors"].append(f"Missing split: {split}")
            continue
        classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        total = sum(len(list((split_dir / c).glob("*"))) for c in classes)
        report["stats"][split] = {"num_classes": len(classes), "num_images": total}
    return report


if __name__ == "__main__":
    config = load_config("params.yaml")
    preprocess_dataset(
        raw_dir=config["data"]["raw_dir"],
        output_dir=config["data"]["processed_dir"],
        image_size=config["data"]["image_size"],
        train_split=config["data"]["train_split"],
    )
