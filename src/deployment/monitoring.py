"""Model monitoring utilities for drift detection and performance tracking."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PredictionRecord:
    timestamp: float
    class_index: int
    confidence: float
    latency_ms: float


class ModelMonitor:
    """Monitor model predictions for drift and performance degradation."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions: deque = deque(maxlen=window_size)
        self.baseline_distribution: Optional[np.ndarray] = None

    def record_prediction(self, class_index: int, confidence: float,
                          latency_ms: float) -> None:
        self.predictions.append(PredictionRecord(
            timestamp=time.time(),
            class_index=class_index,
            confidence=confidence,
            latency_ms=latency_ms,
        ))

    def set_baseline(self, class_distribution: np.ndarray) -> None:
        """Set baseline class distribution for drift detection."""
        self.baseline_distribution = class_distribution / class_distribution.sum()

    def get_metrics(self) -> Dict[str, float]:
        """Get current monitoring metrics."""
        if not self.predictions:
            return {}

        confidences = [p.confidence for p in self.predictions]
        latencies = [p.latency_ms for p in self.predictions]

        return {
            "num_predictions": len(self.predictions),
            "avg_confidence": float(np.mean(confidences)),
            "min_confidence": float(np.min(confidences)),
            "avg_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
        }

    def detect_drift(self, num_classes: int, threshold: float = 0.1) -> Dict:
        """Detect prediction distribution drift using KL divergence."""
        if self.baseline_distribution is None or not self.predictions:
            return {"drift_detected": False, "reason": "no baseline"}

        class_counts = np.zeros(num_classes)
        for p in self.predictions:
            if p.class_index < num_classes:
                class_counts[p.class_index] += 1

        current_dist = class_counts / max(class_counts.sum(), 1)

        # KL divergence
        eps = 1e-10
        kl_div = float(np.sum(
            current_dist * np.log((current_dist + eps) / (self.baseline_distribution + eps))
        ))

        return {
            "drift_detected": kl_div > threshold,
            "kl_divergence": kl_div,
            "threshold": threshold,
        }
