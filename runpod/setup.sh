#!/bin/bash
# setup.sh — FIRST TIME ONLY on a brand new pod
# Installs system-level tools and auth only. Project deps are installed per-project.
# EVERYTHING lives in /workspace/ — container disk is only 5GB, not enough for anything
# NOTE: Do NOT use set -e — .bashrc_pod sources legacy paths that may not exist.

echo "=== First-Time Pod Setup ==="

# Load tokens from /workspace/.env
if [ ! -f /workspace/.env ]; then
    echo "[ERROR] /workspace/.env not found."
    echo "Create it with: cat > /workspace/.env << 'EOF' ... EOF"
    exit 1
fi
set -a; source /workspace/.env; set +a

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
    libgl1-mesa-glx libegl1-mesa libglib2.0-0 libglu1-mesa \
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

# Persist wandb netrc to volume (default goes to /root/.netrc which is wiped)
cp /root/.netrc /workspace/.netrc 2>/dev/null || true
ln -sf /workspace/.netrc /root/.netrc

# ---- Persist env vars for all future sessions ----
cat > /workspace/.bashrc_pod << 'ENVEOF'
# Export all vars from .env (set -a auto-exports, needed for Python/wandb/HF)
set -a; source /workspace/.env 2>/dev/null; set +a
export PATH="/workspace/.local/bin:$PATH"
export UV_CACHE_DIR=/workspace/.cache/uv
export PIP_CACHE_DIR=/workspace/.cache/pip
export XDG_CACHE_HOME=/workspace/.cache
export HF_HOME=/workspace/.cache/huggingface
export WANDB_DIR=/workspace
export HDF5_USE_FILE_LOCKING=FALSE
git config --global credential.helper "store --file=/workspace/.git-credentials"
git config --global user.name "${GIT_NAME}"
git config --global user.email "${GIT_EMAIL}"
ln -sfn /workspace/.claude ~/.claude
ln -sf /workspace/.netrc ~/.netrc

# Redirect container caches to volume (survives restart)
mkdir -p /workspace/.cache/{ov,pip,wandb}
ln -sfn /workspace/.cache/ov /root/.cache/ov 2>/dev/null
ln -sfn /workspace/.cache/pip /root/.cache/pip 2>/dev/null
ln -sfn /workspace/.cache/wandb /root/.cache/wandb 2>/dev/null

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

# ---- Install Claude Code ----
echo "--- Installing Claude Code ---"
if [ -f /workspace/.local/bin/claude ]; then
    echo "Claude Code binary found on volume, skipping install"
    ln -sf /workspace/.local/bin/claude /usr/local/bin/claude
else
    npm install -g @anthropic-ai/claude-code
    # Persist binary to volume so we don't reinstall on restart
    cp "$(which claude)" /workspace/.local/bin/claude 2>/dev/null || true
    ln -sf /workspace/.local/bin/claude /usr/local/bin/claude
fi
chmod 755 /workspace/.local/bin/claude 2>/dev/null || true

# ---- Create non-root user for --dangerously-skip-permissions ----
echo "--- Creating dev user ---"
useradd -m -s /bin/bash dev 2>/dev/null || true
ln -sfn /workspace/.claude /home/dev/.claude
chmod -R 777 /workspace/.claude 2>/dev/null

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
echo "  # Run Claude Code with skip-permissions:"
echo "  su - dev -c 'source /workspace/.bashrc_pod && cd /workspace/code/personal-research && claude --dangerously-skip-permissions'"
