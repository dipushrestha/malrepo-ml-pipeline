# Experiments

This directory contains experiment configurations and results.

## Running an Experiment

```bash
python scripts/train.py --config experiments/experiment_001.yaml
```

## Experiment Log

| ID | Model | Dataset | Accuracy | F1 | Date |
|----|-------|---------|----------|-----|------|
| 001 | ResNet50 | CIFAR-10 | 93.2% | 0.928 | 2025-01-10 |
| 002 | ViT-Small | CIFAR-10 | 89.7% | 0.891 | 2025-01-15 |

## Adding Experiments

1. Copy `experiment_001.yaml` as a template
2. Update config overrides and expected metrics
3. Run training and record actual results
