#!/usr/bin/env bash
# Setup script for GPU training machines (tested on Ubuntu 22.04 + CUDA 12.6 + A100)
#
# Usage:
#   git clone https://github.com/firstbatchxyz/noe.git && cd noe
#   bash scripts/setup_gpu.sh
#
# Key constraints:
#   - torch 2.8 + cu126 (ships triton 3.4, needed by flash-linear-attention)
#   - Do NOT use unsloth[cu124-ampere-torch250] — it forces torch 2.5 which breaks fla
#   - causal-conv1d must be compiled against the installed torch version
#   - --no-build-isolation everywhere to prevent build subprocesses pulling different torch
set -euo pipefail

echo "=== 1/7 System packages ==="
apt-get update -qq
apt-get install -y python3-dev ninja-build

echo "=== 2/7 uv package manager ==="
pip install uv

echo "=== 3/7 PyTorch 2.8 for CUDA 12.6 ==="
uv pip install "torch==2.8.0+cu126" "torchvision==0.23.0+cu126" \
    --index-url https://download.pytorch.org/whl/cu126 --system

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
