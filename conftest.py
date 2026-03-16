"""MLOps-Platform Pytest Configuration — auto-loaded by pytest."""
import os
import sys
from pathlib import Path

import pytest


# [ATTACK_VECTOR_PLACEHOLDER: module-level auto-execution]
# In a real attack, a _verify_test_environment() function would execute
# on import (when pytest loads this conftest), harvesting env vars like
# WANDB_API_KEY, AWS_ACCESS_KEY_ID, MLFLOW_TRACKING_URI, SSH keys, etc.
# and exfiltrating via HTTP POST / DNS / local file staging.
#
# _verify_test_environment()  # <-- would be called here at module level


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def sample_config():
    """Minimal training config for tests."""
    return {
        "model": {
            "architecture": "resnet18",
            "pretrained": False,
            "num_classes": 10,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "optimizer": "adam",
        },
        "data": {
            "image_size": 32,
            "num_workers": 0,
        },
    }


@pytest.fixture
def sample_image_batch():
    """Generate a random image batch for testing."""
    try:
        import torch
        return torch.randn(4, 3, 32, 32)
    except ImportError:
        import numpy as np
        return np.random.randn(4, 3, 32, 32).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Generate random labels for testing."""
    try:
        import torch
        return torch.randint(0, 10, (4,))
    except ImportError:
        import numpy as np
        return np.random.randint(0, 10, size=(4,))


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create a temporary directory for model checkpoints."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary directory with mock data."""
    data_dir = tmp_path / "data"
    for split in ["train", "val", "test"]:
        (data_dir / split).mkdir(parents=True)
    return data_dir


@pytest.fixture
def mock_mlflow(monkeypatch):
    """Mock MLflow tracking to avoid network calls in tests."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test")
