#!/bin/bash
# restart.sh — Run EVERY TIME after pod stop→start
# System packages are on container disk (wiped on stop).
# Everything else (venvs, claude binary, caches, auth) lives on /workspace/ volume.
set -e

echo "=== Pod Restart Recovery ==="

# ---- Restore env vars and paths from /workspace ----
source /workspace/.bashrc_pod

# ---- Reinstall system packages (wiped on stop, ~30s) ----
echo "--- Reinstalling system packages ---"
apt-get update -qq && apt-get install -qq -y \
    vim htop tree wget curl git tmux \
    build-essential cmake \
    libgl1-mesa-glx libegl1-mesa libglib2.0-0 \
    > /dev/null 2>&1

# ---- Restore symlinks (container home is wiped) ----
echo "--- Restoring symlinks ---"
ln -sfn /workspace/.claude ~/.claude
ln -sf /workspace/.netrc ~/.netrc

# ---- Claude Code: symlink from volume binary (no reinstall needed) ----
echo "--- Linking Claude Code from volume ---"
ln -sf /workspace/.local/bin/claude /usr/local/bin/claude

# ---- Recreate dev user (container user db wiped on stop) ----
echo "--- Setting up dev user ---"
useradd -m -s /bin/bash dev 2>/dev/null || true
ln -sfn /workspace/.claude /home/dev/.claude
chmod -R 777 /workspace/.claude 2>/dev/null

echo ""
echo "=== Ready ==="
echo "System tools restored. Everything else intact on /workspace/."
echo ""
echo "Activate a project:"
echo "  proj residual-rl"
echo ""
echo "Run Claude Code with skip-permissions:"
echo "  su - dev -c 'source /workspace/.bashrc_pod && cd /workspace/code/personal-research && claude --dangerously-skip-permissions'"
