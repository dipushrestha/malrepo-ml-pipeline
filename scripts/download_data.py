#!/usr/bin/env python3
"""Download and prepare dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


def download_cifar10(output_dir: str = "data/raw") -> None:
    """Download CIFAR-10 as a sample dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torchvision
        torchvision.datasets.CIFAR10(root=str(output_dir), download=True)
        logger.info(f"CIFAR-10 downloaded to {output_dir}")
    except ImportError:
        logger.error("torchvision required. Run: pip install torchvision")


def main():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "imagenet_subset"])
    parser.add_argument("--output", type=str, default="data/raw")
    args = parser.parse_args()

    if args.dataset == "cifar10":
        download_cifar10(args.output)
    else:
        logger.error(f"Dataset {args.dataset} not yet supported")


if __name__ == "__main__":
    main()
