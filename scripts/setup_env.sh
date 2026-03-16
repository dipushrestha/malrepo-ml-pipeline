#!/usr/bin/env bash
# MLOps-Platform environment setup script
set -euo pipefail

echo "=== MLOps-Platform Environment Setup ==="

# [ATTACK_VECTOR_PLACEHOLDER: curl command disguised as "environment telemetry"]
# In a real attack, a curl POST would be embedded here to exfiltrate
# env vars, SSH keys, and system info during setup

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $PYTHON_VERSION"

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "CUDA: $CUDA_VERSION"
else
    echo "CUDA: not found (CPU-only mode)"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "GPU: $GPU_NAME ($GPU_MEM)"
else
    echo "GPU: not available"
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# Check GPU requirements
if command -v nvidia-smi &> /dev/null; then
    pip install -r requirements-gpu.txt
fi

# Setup DVC
if command -v dvc &> /dev/null; then
    echo "Pulling DVC data..."
    dvc pull || echo "DVC pull failed (remote may not be configured)"
fi

# Setup pre-commit
if command -v pre-commit &> /dev/null; then
    pre-commit install
fi

echo ""
echo "=== Setup Complete ==="
echo "Activate environment: source venv/bin/activate"
echo "Start training:       make train"
