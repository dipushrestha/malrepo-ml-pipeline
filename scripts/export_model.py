#!/usr/bin/env python3
"""Export trained model to ONNX format."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "torchscript"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    try:
        import torch
        checkpoint = torch.load(args.checkpoint, map_location="cpu")

        from src.models.registry import ModelRegistry
        model = ModelRegistry.create(
            checkpoint.get("architecture", "resnet50").lower(),
            num_classes=checkpoint.get("num_classes", 1000),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        output_path = args.output or str(Path(args.checkpoint).with_suffix(f".{args.format}"))

        if args.format == "onnx":
            dummy = torch.randn(1, 3, 224, 224)
            torch.onnx.export(
                model, dummy, output_path,
                input_names=["image"],
                output_names=["logits"],
                dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
            )
        elif args.format == "torchscript":
            scripted = torch.jit.script(model)
            scripted.save(output_path)

        logger.info(f"Model exported to {output_path}")

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")


if __name__ == "__main__":
    main()
