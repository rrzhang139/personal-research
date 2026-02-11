#!/bin/bash
# restart.sh — Run EVERY TIME after pod stop→start
# /workspace survived (venv, packages, models, code, claude auth all intact)
# Container disk was wiped (git config, binaries, system packages gone)
set -e

echo "=== Pod Restart Recovery ==="

# ---- Restore env vars, venv, git config from /workspace ----
source /workspace/.bashrc_pod

# ---- Reinstall system packages (wiped with container) ----
echo "--- Reinstalling system packages ---"
apt-get update -qq && apt-get install -qq -y \
    vim htop tree wget curl git tmux \
    build-essential cmake \
    libgl1-mesa-glx libegl1-mesa libglib2.0-0 \
    > /dev/null 2>&1

# ---- Reinstall uv (binary was on container disk) ----
if ! command -v uv &> /dev/null; then
    echo "--- Reinstalling uv ---"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---- Reinstall Claude Code (binary was on container disk) ----
if ! command -v claude &> /dev/null; then
    echo "--- Reinstalling Claude Code ---"
    curl -fsSL https://claude.ai/install.sh | bash
fi

echo ""
echo "=== Ready ==="
echo "Venv active. Auth restored. Claude auth persisted from /workspace/.claude/"
echo "All pip packages still installed (from /workspace/venv)."
echo ""
echo "Run:"
echo "  tmux new -s work"
echo "  claude"
