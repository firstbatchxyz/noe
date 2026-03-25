#!/usr/bin/env bash
# Setup script for GPU training machines.
# Auto-detects CUDA version and installs the matching PyTorch wheel.
#
# Usage:
#   git clone https://github.com/firstbatchxyz/noe.git && cd noe
#   bash scripts/setup_gpu.sh
#
# Key constraints:
#   - Do NOT use unsloth[cu124-ampere-torch250] — it forces torch 2.5 which breaks fla
#   - causal-conv1d must be compiled against the installed torch version
#   - --no-build-isolation everywhere to prevent build subprocesses pulling different torch
set -euo pipefail

# --- Auto-detect CUDA version → PyTorch wheel index ---
detect_cuda() {
    local cuda_version
    if command -v nvcc &>/dev/null; then
        cuda_version=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    elif [ -f /usr/local/cuda/version.txt ]; then
        cuda_version=$(grep -oP '[0-9]+\.[0-9]+' /usr/local/cuda/version.txt)
    elif command -v nvidia-smi &>/dev/null; then
        cuda_version=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
    else
        echo "ERROR: No CUDA installation found (checked nvcc, /usr/local/cuda, nvidia-smi)"
        exit 1
    fi
    echo "$cuda_version"
}

cuda_to_torch_index() {
    local cv="$1"
    local major minor
    major=$(echo "$cv" | cut -d. -f1)
    minor=$(echo "$cv" | cut -d. -f2)

    # Map to nearest supported PyTorch CUDA wheel
    if [ "$major" -eq 11 ]; then
        echo "cu118"
    elif [ "$major" -eq 12 ]; then
        if [ "$minor" -le 1 ]; then
            echo "cu121"
        elif [ "$minor" -le 4 ]; then
            echo "cu124"
        else
            echo "cu126"
        fi
    else
        echo "ERROR: Unsupported CUDA $cv (need 11.x or 12.x)"
        exit 1
    fi
}

CUDA_VERSION=$(detect_cuda)
CUDA_TAG=$(cuda_to_torch_index "$CUDA_VERSION")
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

echo "=== Detected CUDA $CUDA_VERSION → PyTorch index: $CUDA_TAG ==="

echo "=== 1/7 System packages ==="
apt-get update -qq
apt-get install -y python3-dev ninja-build

echo "=== 2/7 uv package manager ==="
pip install uv

echo "=== 3/7 PyTorch for CUDA $CUDA_VERSION ==="
uv pip install torch torchvision \
    --index-url "$TORCH_INDEX" --system

echo "=== 4/7 Project dependencies ==="
uv pip install -e . --no-build-isolation --system

echo "=== 5/7 Unsloth (from git, no cu124 extra) ==="
uv pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" \
    --no-build-isolation --system

echo "=== 6/7 Rebuild causal-conv1d against current torch ==="
uv pip install --force-reinstall --no-cache "causal-conv1d>=1.4.0" \
    --no-build-isolation --no-deps --system

echo "=== 7/7 Verify ==="
python3 -c "
import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')
import fla; print('flash-linear-attention OK')
import causal_conv1d; print('causal-conv1d OK')
from unsloth import FastLanguageModel; print('unsloth OK')
print('All OK')
"

echo "=== Setup complete. Run training with: ==="
echo "  python scripts/train_stage_a.py --groups 2 --max-samples 30000"
