# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2025-01-15

### Added
- Vision Transformer (ViT) architecture support
- DVC pipeline for automated data preprocessing
- Kubernetes HPA for model serving autoscaling

### Changed
- Upgraded PyTorch from 2.0 to 2.1
- Improved data loader performance with persistent workers

### Fixed
- Memory leak in multi-GPU distributed training
- MLflow artifact logging timeout on large models

## [1.1.0] - 2024-11-20

### Added
- MLflow experiment tracking integration
- Model export to ONNX format
- Automated model evaluation reports

### Changed
- Migrated from argparse to Hydra for configuration
- Improved ResNet training recipe

## [1.0.0] - 2024-09-01

### Added
- Initial release
- ResNet50 training pipeline
- FastAPI model serving endpoint
- Docker containerization
- Basic test suite
