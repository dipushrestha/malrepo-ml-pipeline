"""Training callbacks for logging, checkpointing, etc."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MLflowCallback:
    """Log metrics and artifacts to MLflow."""

    def __init__(self, experiment_name: str = "default"):
        self.experiment_name = experiment_name
        self._run = None

    def __call__(self, epoch: int, history: Dict[str, List[float]]) -> None:
        try:
            import mlflow

            if self._run is None:
                mlflow.set_experiment(self.experiment_name)
                self._run = mlflow.start_run()

            mlflow.log_metrics({
                "train_loss": history["train_loss"][-1],
                "train_acc": history["train_acc"][-1],
                "val_loss": history["val_loss"][-1] if history["val_loss"] else 0,
                "val_acc": history["val_acc"][-1] if history["val_acc"] else 0,
            }, step=epoch)

        except ImportError:
            logger.debug("MLflow not available, skipping logging")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")


class WandbCallback:
    """Log metrics to Weights & Biases."""

    def __init__(self, project: str = "mlops-platform"):
        self.project = project

    def __call__(self, epoch: int, history: Dict[str, List[float]]) -> None:
        try:
            import wandb

            if wandb.run is None:
                wandb.init(project=self.project)

            wandb.log({
                "epoch": epoch,
                "train/loss": history["train_loss"][-1],
                "train/accuracy": history["train_acc"][-1],
                "val/loss": history["val_loss"][-1] if history["val_loss"] else 0,
                "val/accuracy": history["val_acc"][-1] if history["val_acc"] else 0,
            })

        except ImportError:
            logger.debug("wandb not available, skipping logging")


class EarlyStoppingCallback:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 10, metric: str = "val_loss",
                 mode: str = "min"):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, epoch: int, history: Dict[str, List[float]]) -> None:
        if self.metric not in history or not history[self.metric]:
            return

        current = history[self.metric][-1]

        if self.mode == "min":
            improved = current < self.best_value
        else:
            improved = current > self.best_value

        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}")


class PrintCallback:
    """Print training progress to stdout."""

    def __call__(self, epoch: int, history: Dict[str, List[float]]) -> None:
        parts = [f"Epoch {epoch:4d}"]
        for key in ["train_loss", "train_acc", "val_loss", "val_acc"]:
            if key in history and history[key]:
                parts.append(f"{key}={history[key][-1]:.4f}")
        print(" | ".join(parts))
