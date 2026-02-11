# Personal Research — RunPod GPU Environment

## You are a robotics research agent working on a RunPod GPU instance.

## Critical: Volume Persistence
- **ALL work lives in `/workspace/`** — this is the persistent volume disk
- The container disk (`/root/`, `/usr/`, etc.) is WIPED on every pod stop/restart
- Never store anything important outside `/workspace/`
- Python venv is at `/workspace/venv` — always activate it: `source /workspace/venv/bin/activate`
- Package manager is `uv` (10-100x faster than pip). Use `uv pip install` instead of `pip install`

## Directory Layout
```
/workspace/
├── .env                  # Auth tokens (HF, GitHub, W&B, Anthropic) — source this
├── .bashrc_pod           # Auto-sources .env, activates venv, sets env vars
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
This activates the venv, sets HF_HOME, WANDB_DIR, ANTHROPIC_API_KEY, and git credentials.

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
