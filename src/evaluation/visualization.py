"""Result visualization utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def plot_training_history(history: Dict[str, List[float]],
                         output_path: Optional[str] = None) -> None:
    """Plot training and validation curves."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(history["train_loss"], label="Train")
        if history.get("val_loss"):
            axes[0].plot(history["val_loss"], label="Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(history["train_acc"], label="Train")
        if history.get("val_acc"):
            axes[1].plot(history["val_acc"], label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    except ImportError:
        pass


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None,
                          output_path: Optional[str] = None) -> None:
    """Plot confusion matrix as heatmap."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)

        if class_names:
            ax.set(
                xticks=range(len(class_names)),
                yticks=range(len(class_names)),
                xticklabels=class_names,
                yticklabels=class_names,
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    except ImportError:
        pass
