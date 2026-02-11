#!/bin/bash
# setup.sh — FIRST TIME ONLY on a brand new pod
# Installs system-level tools and auth only. Project deps are installed per-project.
# EVERYTHING lives in /workspace/ — container disk is only 5GB, not enough for anything
set -e

echo "=== First-Time Pod Setup ==="

# Load tokens from /workspace/.env
if [ ! -f /workspace/.env ]; then
    echo "[ERROR] /workspace/.env not found."
    echo "Create it with: cat > /workspace/.env << 'EOF' ... EOF"
    exit 1
fi
source /workspace/.env

# ---- Redirect ALL caches to /workspace (container disk is only 5GB) ----
export UV_CACHE_DIR=/workspace/.cache/uv
export PIP_CACHE_DIR=/workspace/.cache/pip
export XDG_CACHE_HOME=/workspace/.cache
export HF_HOME=/workspace/.cache/huggingface
export WANDB_DIR=/workspace
mkdir -p /workspace/.cache/{uv,pip,huggingface}

# ---- Essential system packages (small, ok on container) ----
echo "--- Installing system packages ---"
apt-get update -qq && apt-get install -qq -y \
    vim htop tree wget curl git tmux \
    build-essential cmake \
    libgl1-mesa-glx libegl1-mesa libglib2.0-0 \
    > /dev/null 2>&1

# ---- Node.js (needed for Claude Code npm install) ----
echo "--- Installing Node.js ---"
curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
apt-get install -qq -y nodejs > /dev/null 2>&1

# ---- Directory structure ----
mkdir -p /workspace/{code,datasets,results,models}
mkdir -p /workspace/.claude

# ---- Symlink ~/.claude → /workspace/.claude (auth persists) ----
ln -sfn /workspace/.claude ~/.claude

# ---- Install uv to /workspace/.local (not container) ----
echo "--- Installing uv ---"
export CARGO_HOME=/workspace/.cargo
export UV_INSTALL_DIR=/workspace/.local/bin
mkdir -p /workspace/.local/bin
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/workspace/.local/bin:$PATH"

# ---- Auth ----
echo "--- Setting up auth ---"
git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"
echo "https://${GITHUB_TOKEN}@github.com" > /workspace/.git-credentials
git config --global credential.helper "store --file=/workspace/.git-credentials"

# HF and W&B auth (use a temp python for login, no venv needed yet)
uv run --python 3.11 python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" 2>/dev/null || \
    echo "[WARN] HF login failed — run manually after project env setup"
uv run --python 3.11 python -m wandb login "$WANDB_API_KEY" 2>/dev/null || \
    echo "[WARN] wandb login failed — run manually after project env setup"

# ---- Persist env vars for all future sessions ----
cat > /workspace/.bashrc_pod << 'ENVEOF'
source /workspace/.env 2>/dev/null
export PATH="/workspace/.local/bin:$PATH"
export UV_CACHE_DIR=/workspace/.cache/uv
export PIP_CACHE_DIR=/workspace/.cache/pip
export XDG_CACHE_HOME=/workspace/.cache
export HF_HOME=/workspace/.cache/huggingface
export WANDB_DIR=/workspace
git config --global credential.helper "store --file=/workspace/.git-credentials"
git config --global user.name "${GIT_NAME}"
git config --global user.email "${GIT_EMAIL}"
ln -sfn /workspace/.claude ~/.claude

# Helper: activate a project env
proj() {
    local dir="/workspace/code/personal-research/$1"
    if [ -f "$dir/.venv/bin/activate" ]; then
        source "$dir/.venv/bin/activate"
        cd "$dir"
        echo "Activated $1 env"
    else
        echo "No .venv found in $dir — run: cd $dir && bash setup_env.sh"
    fi
}
ENVEOF

# ---- Clone this repo ----
cd /workspace/code
if [ ! -d "personal-research" ]; then
    git clone https://github.com/rrzhang139/personal-research.git
fi

# ---- Install Claude Code via npm ----
echo "--- Installing Claude Code ---"
npm install -g @anthropic-ai/claude-code

echo ""
echo "=== System setup complete ==="
echo ""
echo "Next steps:"
echo "  source /workspace/.bashrc_pod"
echo ""
echo "  # Set up a project environment:"
echo "  cd /workspace/code/personal-research/residual-rl"
echo "  bash setup_env.sh"
echo ""
echo "  # Or use the shortcut after bashrc is sourced:"
echo "  proj residual-rl"
