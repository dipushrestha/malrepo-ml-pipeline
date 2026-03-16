"""Evaluation metrics for model assessment."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    return float(np.mean(y_true == y_pred))


def precision_per_class(y_true: np.ndarray, y_pred: np.ndarray,
                        num_classes: int) -> np.ndarray:
    """Compute per-class precision."""
    precisions = np.zeros(num_classes)
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        precisions[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precisions


def recall_per_class(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: int) -> np.ndarray:
    """Compute per-class recall."""
    recalls = np.zeros(num_classes)
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        recalls[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recalls


def f1_per_class(y_true: np.ndarray, y_pred: np.ndarray,
                 num_classes: int) -> np.ndarray:
    """Compute per-class F1 score."""
    p = precision_per_class(y_true, y_pred, num_classes)
    r = recall_per_class(y_true, y_pred, num_classes)
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    return f1


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: int) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        num_classes: int) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    p = precision_per_class(y_true, y_pred, num_classes)
    r = recall_per_class(y_true, y_pred, num_classes)
    f1 = f1_per_class(y_true, y_pred, num_classes)

    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(
            f1, weights=np.bincount(y_true, minlength=num_classes)
        )) if len(y_true) > 0 else 0.0,
    }
