# Training Guide

## Configuration

All training runs are driven by YAML config files in `configs/`. Key parameters:

- `model.architecture` — Model backbone (resnet18, resnet50, vit_base, vit_small)
- `training.epochs` — Number of training epochs
- `training.learning_rate` — Initial learning rate
- `training.mixed_precision` — Enable FP16 training (recommended for GPU)

## Single-GPU Training

```bash
python scripts/train.py --config configs/resnet50.yaml
```

## Multi-GPU Training

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/resnet50.yaml --distributed
```

## Resuming Training

```bash
python scripts/train.py --config configs/resnet50.yaml --resume models/saved/latest.pt
```

## Hyperparameter Tuning Tips

- Start with the base config and increase complexity
- Use warmup for Vision Transformers (30 epochs recommended)
- Label smoothing (0.1) helps generalization for large models
- Gradient clipping (1.0) prevents NaN loss with mixed precision

## Monitoring

Training metrics are logged to MLflow at `http://localhost:5000`. Use the MLflow UI
to compare experiments, view loss curves, and download artifacts.
