# Architecture Overview

## System Components

MLOps-Platform consists of five main components:

**Data Pipeline** — DVC-managed preprocessing with reproducible transforms. Raw images
are resized, augmented, and split into train/val/test partitions.

**Model Training** — PyTorch training loop with mixed-precision support, gradient clipping,
and early stopping. Callbacks log metrics to MLflow and W&B.

**Experiment Tracking** — MLflow tracks parameters, metrics, and artifacts. YAML configs
define reproducible experiments.

**Model Serving** — FastAPI endpoint wraps the inference pipeline. Supports batching,
health checks, and Prometheus metrics export.

**Monitoring** — Real-time drift detection using KL divergence over prediction distributions.
Latency percentiles tracked for SLA compliance.

## Data Flow

```
Raw Images → DVC Preprocessing → Train/Val/Test Splits
                                        ↓
                                  PyTorch DataLoader
                                        ↓
                              Training Loop (GPU)
                                        ↓
                              Model Checkpoint (.pt)
                                        ↓
                         ┌──────────────┴──────────────┐
                         ↓                              ↓
                   ONNX Export                   FastAPI Serving
                         ↓                              ↓
                  Edge Deployment            Kubernetes (HPA)
```

## Directory Layout

See the project root README.md for the full directory tree.
