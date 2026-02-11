#!/bin/bash
# setup.sh — FIRST TIME ONLY on a brand new pod
# Creates venv in /workspace so packages survive pod stop/restart
# After this, use restart.sh on each pod restart
set -e

echo "=== First-Time Pod Setup ==="

# Load tokens from /workspace/.env
if [ ! -f /workspace/.env ]; then
    echo "[ERROR] /workspace/.env not found."
    echo "Upload it first:"
    echo "  scp -i ~/.ssh/runpod .env <user>@ssh.runpod.io:/workspace/.env"
    exit 1
fi
source /workspace/.env

# ---- Directory structure (all in /workspace = survives stop) ----
mkdir -p /workspace/{code,datasets,results,models}
mkdir -p /workspace/.cache/huggingface

# ---- Python venv in /workspace (survives stop) ----
echo "--- Creating Python venv ---"
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate

# ---- Core ML packages ----
echo "--- Installing core packages ---"
pip install -q --upgrade pip setuptools wheel
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -q numpy scipy matplotlib pandas tqdm ipython

# ---- Robotics sim packages ----
echo "--- Installing robotics sim packages ---"
pip install -q mujoco gymnasium
pip install -q robosuite
pip install -q mani-skill  # ManiSkill 3

# ---- LIBERO (from source for latest) ----
echo "--- Installing LIBERO ---"
cd /workspace/code
if [ ! -d "LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi
cd LIBERO && pip install -q -e . && cd /workspace

# ---- Isaac Lab (optional — uncomment if needed) ----
# echo "--- Installing Isaac Lab ---"
# pip install isaaclab[isaacsim,all]==2.3.2 --extra-index-url https://pypi.nvidia.com

# ---- VLA / ML packages ----
echo "--- Installing ML packages ---"
pip install -q transformers accelerate safetensors
pip install -q huggingface_hub wandb
pip install -q diffusers  # for diffusion policies
pip install -q einops timm  # common VLA deps

# ---- Auth (write to /workspace so it persists through stop) ----
echo "--- Setting up auth ---"
git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"
echo "https://${GITHUB_TOKEN}@github.com" > /workspace/.git-credentials
git config --global credential.helper "store --file=/workspace/.git-credentials"

export HF_HOME=/workspace/.cache/huggingface
huggingface-cli login --token "$HF_TOKEN"
wandb login "$WANDB_API_KEY"

# ---- Persist env vars for future sessions ----
cat > /workspace/.bashrc_pod << 'ENVEOF'
source /workspace/venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export WANDB_DIR=/workspace
source /workspace/.env 2>/dev/null
git config --global credential.helper "store --file=/workspace/.git-credentials"
git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"
ENVEOF

# ---- Clone this repo if not already ----
cd /workspace/code
if [ ! -d "personal-research" ]; then
    git clone https://github.com/rrzhang139/personal-research.git
fi

# ---- Install Claude Code ----
echo "--- Installing Claude Code ---"
if ! command -v claude &> /dev/null; then
    curl -fsSL https://claude.ai/install.sh | bash
fi

# ---- Install tmux (usually pre-installed but just in case) ----
apt-get update -qq && apt-get install -qq -y tmux > /dev/null 2>&1 || true

echo ""
echo "=== First-time setup complete ==="
echo ""
echo "Run:"
echo "  source /workspace/.bashrc_pod"
echo "  tmux new -s work"
echo "  claude"
