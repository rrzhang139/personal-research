#!/bin/bash
# UWLab Environment Setup Script
# Run from: /workspace/code/personal-research/uwlab/
set -e

source /workspace/.bashrc_pod 2>/dev/null
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export UV_CACHE_DIR=/workspace/.cache/uv
export PIP_CACHE_DIR=/workspace/.cache/pip

cd /workspace/code/personal-research/uwlab

echo "[1/6] Creating Python 3.11 venv..."
uv venv --python 3.11 .venv
source .venv/bin/activate

echo "[2/6] Installing Isaac Sim 5.1.0 (this takes ~10 min)..."
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

echo "[3/6] Installing PyTorch 2.7.0 with CUDA 12.8..."
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

echo "[4/6] Cloning UWLab..."
git clone https://github.com/uw-lab/UWLab.git 2>/dev/null || echo "UWLab already cloned"
cd UWLab

echo "[5/6] Installing UWLab + rsl_rl..."
./uwlab.sh --install
./uwlab.sh --install rsl_rl

echo "[6/6] Installing wandb..."
uv pip install wandb

echo "Setup complete. Activate with: source /workspace/code/personal-research/uwlab/.venv/bin/activate"
