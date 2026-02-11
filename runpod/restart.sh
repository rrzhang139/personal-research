#!/bin/bash
# restart.sh — Run EVERY TIME after pod stop→start
# Everything important is in /workspace/ — just need to reinstall system packages
# and re-source env vars
set -e

echo "=== Pod Restart Recovery ==="

# ---- Restore env vars, venv, paths from /workspace ----
source /workspace/.bashrc_pod

# ---- Reinstall system packages (these are small, ok on container) ----
echo "--- Reinstalling system packages ---"
apt-get update -qq && apt-get install -qq -y \
    vim htop tree wget curl git tmux \
    build-essential cmake \
    libgl1-mesa-glx libegl1-mesa libglib2.0-0 \
    > /dev/null 2>&1

echo ""
echo "=== Ready ==="
echo "Venv active. uv, claude, all packages in /workspace/."
echo ""
echo "Run:"
echo "  tmux new -s work"
echo "  claude"
