#!/usr/bin/env python3
"""Inference benchmarking script."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        from src.models.registry import ModelRegistry
        model = ModelRegistry.create(
            checkpoint.get("architecture", "resnet50").lower(),
            num_classes=checkpoint.get("num_classes", 1000),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        dummy = torch.randn(args.batch_size, 3, 224, 224, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(args.warmup):
                model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(args.num_batches):
                start = time.perf_counter()
                model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

        latencies = np.array(latencies)
        throughput = args.batch_size / (latencies.mean() / 1000)

        print(f"\n{'='*50}")
        print(f"Benchmark Results ({device})")
        print(f"{'='*50}")
        print(f"Batch size:    {args.batch_size}")
        print(f"Num batches:   {args.num_batches}")
        print(f"Mean latency:  {latencies.mean():.2f} ms")
        print(f"Std latency:   {latencies.std():.2f} ms")
        print(f"P50 latency:   {np.percentile(latencies, 50):.2f} ms")
        print(f"P95 latency:   {np.percentile(latencies, 95):.2f} ms")
        print(f"P99 latency:   {np.percentile(latencies, 99):.2f} ms")
        print(f"Throughput:    {throughput:.1f} images/sec")

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")


if __name__ == "__main__":
    main()
