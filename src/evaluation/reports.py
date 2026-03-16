"""Report generation for model evaluation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_eval_report(
    metrics: Dict[str, float],
    config: Dict[str, Any],
    output_dir: str = "experiments",
) -> Path:
    """Generate a JSON evaluation report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": config.get("model", {}),
        "training": config.get("training", {}),
        "metrics": metrics,
    }

    output_path = output_dir / "eval_results.json"
    output_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Evaluation report saved to {output_path}")
    return output_path


def generate_comparison_report(
    experiments: list[Dict[str, Any]],
    output_path: str = "experiments/comparison.json",
) -> Path:
    """Generate a comparison report across experiments."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "num_experiments": len(experiments),
        "experiments": experiments,
        "best": max(experiments, key=lambda e: e.get("metrics", {}).get("accuracy", 0))
        if experiments else None,
    }

    path.write_text(json.dumps(comparison, indent=2))
    return path
