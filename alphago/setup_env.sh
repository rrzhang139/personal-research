#!/bin/bash
# setup_env.sh — Create isolated venv for alphago
# Run from the alphago directory: bash setup_env.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure uv and caches are available
export PATH="/workspace/.local/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"

echo "=== Setting up alphago environment ==="

# Create project-specific venv
if [ ! -d ".venv" ]; then
    echo "--- Creating .venv ---"
    uv venv .venv --python 3.11
else
    echo "--- .venv already exists, skipping creation ---"
fi

source .venv/bin/activate

# Auto-detect GPU and install PyTorch accordingly
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "--- GPU detected, installing PyTorch with CUDA ---"
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
    echo "--- No GPU detected, installing PyTorch CPU ---"
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install remaining deps
echo "--- Installing project dependencies ---"
uv pip install -r requirements.txt

# Install project in editable mode
echo "--- Installing alpha_go package ---"
uv pip install -e .

# Install nanobind for C++ MCTS extension
echo "--- Installing nanobind (for C++ MCTS build) ---"
uv pip install nanobind

# Build C++ MCTS extension if cmake is available
if command -v cmake &> /dev/null; then
    echo "--- Building C++ MCTS extension ---"
    rm -rf build
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DPython_EXECUTABLE="$(which python3)" 2>&1 | tail -5
    cmake --build build --parallel 2>&1 | tail -5

    # CRITICAL: Copy .so to site-packages (NOT src/mcts_cpp/)
    # The editable install puts mcts_cpp in src/, but Python resolves
    # the site-packages copy first. A stale .so there causes segfaults.
    SITE_PKG=$(python3 -c "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])")
    cp build/_mcts_cpp*.so "$SITE_PKG/mcts_cpp/"
    echo "    .so installed to: $SITE_PKG/mcts_cpp/"
    python3 -c "from mcts_cpp._mcts_cpp import test_cpp_only; test_cpp_only(9,1,5); print('    C++ MCTS build OK')"
else
    echo "--- cmake not found, skipping C++ MCTS build ---"
    echo "    Install cmake (apt-get install cmake) then re-run, or build manually:"
    echo "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=\$(which python3)"
    echo "    cmake --build build --parallel"
    echo "    cp build/_mcts_cpp*.so \$(python3 -c \"import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])\")/mcts_cpp/"
fi

echo ""
echo "=== alphago environment ready ==="
echo "Activate with: source $SCRIPT_DIR/.venv/bin/activate"
