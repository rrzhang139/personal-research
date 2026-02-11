#!/bin/bash
# setup_env.sh — Create isolated venv for residual-rl
# Run from the residual-rl directory: bash setup_env.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure uv and caches are available
export PATH="/workspace/.local/bin:$PATH"
export UV_CACHE_DIR=/workspace/.cache/uv
export PIP_CACHE_DIR=/workspace/.cache/pip

echo "=== Setting up residual-rl environment ==="

# Create project-specific venv
if [ ! -d ".venv" ]; then
    echo "--- Creating .venv ---"
    uv venv .venv --python 3.11
else
    echo "--- .venv already exists, skipping creation ---"
fi

source .venv/bin/activate

# Install PyTorch with CUDA 12.4
echo "--- Installing PyTorch (CUDA 12.4) ---"
uv pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining deps
echo "--- Installing project dependencies ---"
uv pip install -r requirements.txt

echo ""
echo "=== residual-rl environment ready ==="
echo "Activate with: source $SCRIPT_DIR/.venv/bin/activate"
