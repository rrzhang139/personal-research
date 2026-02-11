#!/bin/bash
# setup.sh — FIRST TIME ONLY on a brand new pod
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

# ---- Python venv in /workspace ----
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
uv pip install mani-skill

# ---- LIBERO ----
echo "--- Installing LIBERO ---"
cd /workspace/code
if [ ! -d "LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi
cd LIBERO && uv pip install -e . && cd /workspace

# ---- Isaac Lab (optional) ----
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

/workspace/venv/bin/python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
/workspace/venv/bin/python -m wandb login "$WANDB_API_KEY"

# ---- Persist env vars for all future sessions ----
cat > /workspace/.bashrc_pod << 'ENVEOF'
source /workspace/.env 2>/dev/null
source /workspace/venv/bin/activate
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
ENVEOF

# ---- Clone this repo ----
cd /workspace/code
if [ ! -d "personal-research" ]; then
    git clone https://github.com/rrzhang139/personal-research.git
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run:"
echo "  source /workspace/.bashrc_pod"
echo ""
echo "Claude Code runs on your LOCAL machine and controls this pod via SSH:"
echo "  ssh -tt oytehiveq30siz-644113ed@ssh.runpod.io -i ~/.ssh/runpod"
