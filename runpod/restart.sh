#!/bin/bash
# restart.sh — Run EVERY TIME after pod stop→start
# System packages are on container disk (wiped on stop).
# Everything else (venvs, claude binary, caches, auth) lives on /workspace/ volume.
# NOTE: Do NOT use set -e here — .bashrc_pod sources legacy paths that may not exist.

echo "=== Pod Restart Recovery ==="

# ---- Restore env vars and paths from /workspace ----
source /workspace/.bashrc_pod 2>/dev/null

# ---- Reinstall system packages (wiped on stop, ~30s) ----
echo "--- Reinstalling system packages ---"
apt-get update -qq && apt-get install -qq -y \
    vim htop tree wget curl git tmux \
    build-essential cmake \
    libgl1-mesa-glx libegl1-mesa libglib2.0-0 libglu1-mesa \
    > /dev/null 2>&1

# ---- Redirect container caches to volume ----
echo "--- Redirecting caches to volume ---"
# Omniverse cache (471MB, regenerates on every restart without this)
mkdir -p /workspace/.cache/ov
rm -rf /root/.cache/ov 2>/dev/null
ln -sfn /workspace/.cache/ov /root/.cache/ov
# pip cache
mkdir -p /workspace/.cache/pip
rm -rf /root/.cache/pip 2>/dev/null
ln -sfn /workspace/.cache/pip /root/.cache/pip
# wandb cache
mkdir -p /workspace/.cache/wandb
rm -rf /root/.cache/wandb 2>/dev/null
ln -sfn /workspace/.cache/wandb /root/.cache/wandb

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
