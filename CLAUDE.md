# Personal Research — RunPod GPU Environment

## You are a robotics research agent working on a RunPod GPU instance.

## Critical: Volume Persistence
- **ALL work lives in `/workspace/`** — this is the persistent volume disk
- The container disk (`/root/`, `/usr/`, etc.) is WIPED on every pod stop/restart
- Never store anything important outside `/workspace/`
- Python venv is at `/workspace/venv` — always activate it: `source /workspace/venv/bin/activate`
- Package manager is `uv` (10-100x faster than pip). Use `uv pip install` instead of `pip install`
- ALL caches (uv, pip, HF, wandb) redirected to `/workspace/.cache/` — container disk is only 5GB
- Binaries (uv, claude) installed to `/workspace/.local/bin/`

## Directory Layout
```
/workspace/
├── .env                  # Auth tokens (HF, GitHub, W&B) — source this
├── .bashrc_pod           # Auto-sources .env, activates venv, sets env vars
├── .claude/              # Claude Code auth (symlinked from ~/.claude, persists)
├── venv/                 # Python venv via uv (persists through stop/restart)
├── code/                 # All git repos
│   ├── personal-research/  # This repo
│   └── LIBERO/             # LIBERO benchmark (pip install -e)
├── datasets/             # LIBERO, DROID, ManiSkill assets
├── results/              # Experiment outputs, CSVs, rollout videos
├── models/               # Downloaded model checkpoints
├── .cache/huggingface/   # HF model cache (persists, no re-download)
└── wandb/                # W&B run logs
```

## Scripts (in runpod/ directory of this repo)

| Script | When to run |
|--------|-------------|
| `runpod/setup.sh` | **FIRST TIME ONLY** — new pod, installs everything with uv |
| `runpod/restart.sh` | **EVERY POD RESTART** — reinstalls claude + uv binaries, re-links auth |
| `runpod/save.sh` | **BEFORE TERMINATE ONLY** — git push + wandb sync (not needed for stop) |

## Pod Lifecycle
- **Stop pod**: `/workspace/` survives (venv, packages, models, code all intact). Run `restart.sh` on next start (~1 min). ~$5/month idle.
- **Terminate pod**: Everything wiped. Run `save.sh` first, then `setup.sh` on new pod.
- The user frequently stops the GPU instance. Always save work to `/workspace/`.

## Running Experiments
- **Always use tmux**: `tmux new -s exp` before starting long-running jobs. If session exists: `tmux attach -t exp`
- **Always use wandb**: Log all experiments to Weights & Biases for tracking
- **Always use HF**: Download datasets/models via `huggingface-cli` so they cache in `/workspace/.cache/huggingface/`
- **Save results to /workspace/results/** with descriptive names
- **Commit code frequently** to `/workspace/code/personal-research/`

## Environment Setup
Source this at the start of every session (restart.sh does it for you):
```bash
source /workspace/.bashrc_pod
```
This activates the venv, sets HF_HOME, WANDB_DIR, git credentials, and symlinks `~/.claude` to `/workspace/.claude`.

## Two Ways to Use Claude Code with This Pod

### Option A: Claude Code on the Pod (Interactive)
- Installed via npm: `npm install -g @anthropic-ai/claude-code`
- OAuth auth: `claude` shows a URL → open in browser → login → done
- Auth saved to `/workspace/.claude/` (symlinked from `~/.claude`), persists through stop/restart
- If OAuth fails with scope errors, downgrade: `npm install -g @anthropic-ai/claude-code@2.1.19`
- If npm install OOMs, add swap first: `fallocate -l 4G /workspace/swapfile && chmod 600 /workspace/swapfile && mkswap /workspace/swapfile && swapon /workspace/swapfile`
- Best for: interactive coding sessions where you SSH into the pod directly

### Option B: Claude Code on Local Machine via SSH (Remote Control)
Claude Code runs locally and executes commands on the pod via SSH heredoc.
RunPod's SSH gateway requires `-tt` (forced PTY) and ignores command arguments — commands MUST be piped via stdin/heredoc.

**CRITICAL: Short vs Long-Running Commands**

SSH heredoc has TWO timeout layers:
1. Claude Code's bash tool timeout (default 2 min, max 10 min)
2. RunPod's SSH gateway drops idle connections

When either fires, the SSH session dies and **the command on the pod dies with it** — unless it was launched in tmux or nohup.

**Short commands (<30s)** — git, pip install, file checks — run directly:
```bash
ssh -tt -i ~/.ssh/runpod oytehiveq30siz-644113ed@ssh.runpod.io << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
git -C /workspace/code/personal-research pull
pip list | grep mani-skill
exit
SSHEOF
```

**Long-running commands** — training, heavy imports, downloads — MUST use detached tmux:
```bash
# Step 1: Launch in detached tmux (returns immediately)
ssh -tt -i ~/.ssh/runpod oytehiveq30siz-644113ed@ssh.runpod.io << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
tmux new-session -d -s train 'bash /workspace/code/personal-research/residual-rl/scripts/setup_and_train.sh > /workspace/results/train.log 2>&1'
echo "TMUX_STARTED"
exit
SSHEOF

# Step 2: Check output later (separate SSH call)
ssh -tt -i ~/.ssh/runpod oytehiveq30siz-644113ed@ssh.runpod.io << 'SSHEOF'
tail -30 /workspace/results/train.log
tmux ls
exit
SSHEOF
```

**Alternative for long commands — nohup:**
```bash
ssh -tt -i ~/.ssh/runpod oytehiveq30siz-644113ed@ssh.runpod.io << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
nohup python /workspace/code/personal-research/residual-rl/offline/diffusion_policy_unet_maniskill2.py \
  --env-id StackCube-v1 > /workspace/results/train.log 2>&1 &
echo "PID: $!"
exit
SSHEOF
```

## Memory Note
After a fresh pod restart, `free -h` may show ~76GB "used" out of 124GB. This is normal:
- ~15GB is Linux disk cache (buff/cache) — instantly reclaimable
- ~60GB is NVIDIA GPU driver + CUDA runtime mapped into system RAM
- Check `available` column (not `used`) — typically ~46GB free for your processes

## Installed Simulation Frameworks
- **MuJoCo** — lightweight physics sim, runs on any GPU
- **robosuite** — robot manipulation suite (used by LIBERO)
- **LIBERO** — lifelong robot learning benchmark (90 tasks), installed editable in /workspace/code/LIBERO
- **ManiSkill 3** — GPU-parallelized manipulation benchmark (needs Vulkan)
- **Isaac Lab** (optional, uncomment in setup.sh) — NVIDIA sim framework (needs RTX GPU with RT cores)
- **gymnasium** — standard RL environments

## Package Management
- Use `uv pip install <package>` to install new packages (not `pip install`)
- Packages install into `/workspace/venv/` and survive pod stop/restart
- For robotics repos with requirements.txt: `uv pip install -r requirements.txt`

## GPU Info
- This pod has an RTX 4090 (24GB VRAM)
- Supports Isaac Sim (has RT cores), ManiSkill (Vulkan), and all standard CUDA workloads
- Check GPU: `nvidia-smi`

## Common Commands
```bash
# Activate env (if not already)
source /workspace/.bashrc_pod

# Install a new package
uv pip install <package>

# Run LIBERO eval
python -m libero.eval --benchmark libero_90 --policy <policy>

# Download model from HF
huggingface-cli download <model-id> --local-dir /workspace/models/<name>

# Start wandb experiment
wandb init -p robotics_experiments

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Tmux
tmux new -s exp        # new session
tmux attach -t exp     # reattach
tmux ls                # list sessions
```
