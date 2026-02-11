#!/bin/bash
# restart.sh — Run EVERY TIME after pod stop→start
# System packages and Node.js are on container disk (wiped on stop).
# Project venvs in /workspace/ survive.
set -e

echo "=== Pod Restart Recovery ==="

# ---- Restore env vars and paths from /workspace ----
source /workspace/.bashrc_pod

# ---- Reinstall system packages (wiped on stop) ----
echo "--- Reinstalling system packages ---"
apt-get update -qq && apt-get install -qq -y \
    vim htop tree wget curl git tmux \
    build-essential cmake \
    libgl1-mesa-glx libegl1-mesa libglib2.0-0 \
    > /dev/null 2>&1

# ---- Node.js + Claude Code (wiped on stop) ----
echo "--- Reinstalling Node.js + Claude Code ---"
curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
apt-get install -qq -y nodejs > /dev/null 2>&1
npm install -g @anthropic-ai/claude-code > /dev/null 2>&1

# ---- Restore symlinks ----
ln -sfn /workspace/.claude ~/.claude

echo ""
echo "=== Ready ==="
echo "System tools restored. Project venvs in /workspace/ are intact."
echo ""
echo "Activate a project:"
echo "  proj residual-rl"
