#!/bin/bash
# setup.sh — FIRST TIME ONLY on a brand new pod
# Uses uv (10-100x faster than pip) for package management
# Creates venv in /workspace so packages survive pod stop/restart
set -e

echo "=== First-Time Pod Setup ==="

# Load tokens from /workspace/.env
if [ ! -f /workspace/.env ]; then
    echo "[ERROR] /workspace/.env not found."
    echo "Upload it first:"
    echo "  scp -O -i ~/.ssh/runpod .env <user>@ssh.runpod.io:/workspace/.env"
    exit 1
fi
source /workspace/.env

# ---- Essential system packages ----
echo "--- Installing system packages ---"
apt-get update -qq && apt-get install -qq -y \
    vim htop tree wget curl git tmux \
    build-essential cmake \
    libgl1-mesa-glx libegl1-mesa libglib2.0-0 \
    > /dev/null 2>&1

# ---- Directory structure (all in /workspace = survives stop) ----
mkdir -p /workspace/{code,datasets,results,models}
mkdir -p /workspace/.cache/huggingface
mkdir -p /workspace/.claude

# ---- Symlink ~/.claude → /workspace/.claude (auth persists through restart) ----
ln -sfn /workspace/.claude ~/.claude

# ---- Install uv (fast Python package manager) ----
echo "--- Installing uv ---"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ---- Python venv in /workspace (survives stop) ----
echo "--- Creating Python venv ---"
uv venv /workspace/venv --python 3.11
source /workspace/venv/bin/activate

# ---- Core ML packages ----
echo "--- Installing core packages ---"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install numpy scipy matplotlib pandas tqdm ipython

# ---- Robotics sim packages ----
echo "--- Installing robotics sim packages ---"
uv pip install mujoco gymnasium
uv pip install robosuite
uv pip install mani-skill  # ManiSkill 3

# ---- LIBERO (from source for latest) ----
echo "--- Installing LIBERO ---"
cd /workspace/code
if [ ! -d "LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi
cd LIBERO && uv pip install -e . && cd /workspace

# ---- Isaac Lab (optional — uncomment if needed, ~10GB) ----
# echo "--- Installing Isaac Lab ---"
# uv pip install isaaclab[isaacsim,all]==2.3.2 --extra-index-url https://pypi.nvidia.com

# ---- VLA / ML packages ----
echo "--- Installing ML packages ---"
uv pip install transformers accelerate safetensors
uv pip install huggingface_hub wandb
uv pip install diffusers einops timm

# ---- Auth ----
echo "--- Setting up auth ---"
git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"
echo "https://${GITHUB_TOKEN}@github.com" > /workspace/.git-credentials
git config --global credential.helper "store --file=/workspace/.git-credentials"

export HF_HOME=/workspace/.cache/huggingface
huggingface-cli login --token "$HF_TOKEN"
wandb login "$WANDB_API_KEY"

# ---- Persist env vars + venv activation for all future sessions ----
cat > /workspace/.bashrc_pod << 'ENVEOF'
source /workspace/.env 2>/dev/null
source /workspace/venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/workspace/.cache/huggingface
export WANDB_DIR=/workspace
git config --global credential.helper "store --file=/workspace/.git-credentials"
git config --global user.name "${GIT_NAME}"
git config --global user.email "${GIT_EMAIL}"
# Symlink claude auth to volume so it persists
ln -sfn /workspace/.claude ~/.claude
ENVEOF

# ---- Clone this repo ----
cd /workspace/code
if [ ! -d "personal-research" ]; then
    git clone https://github.com/rrzhang139/personal-research.git
fi

# ---- Install Claude Code ----
echo "--- Installing Claude Code ---"
if ! command -v claude &> /dev/null; then
    curl -fsSL https://claude.ai/install.sh | bash
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run:"
echo "  source /workspace/.bashrc_pod"
echo "  tmux new -s work"
echo "  claude              # first time: it will give you a URL to login in browser"
echo ""
echo "Claude Code login is ONE TIME — auth token saved to /workspace/.claude/"
echo "It will survive pod stop/restart."
