# Personal Research — RunPod GPU Environment

## You are a robotics research agent working on a RunPod GPU instance.

## Critical: Volume Persistence
- **ALL work lives in `/workspace/`** — this is the persistent volume disk
- The container disk (`/root/`, `/usr/`, etc.) is WIPED on every pod stop/restart
- Never store anything important outside `/workspace/`
- Python venv is at `/workspace/venv` — always activate it first: `source /workspace/venv/bin/activate`

## Directory Layout
```
/workspace/
├── .env                  # Auth tokens (HF, GitHub, W&B) — source this
├── venv/                 # Python venv (persists through stop/restart)
├── code/                 # All git repos
│   └── personal-research/  # This repo
├── datasets/             # LIBERO, DROID, ManiSkill assets
├── results/              # Experiment outputs, CSVs, rollout videos
├── models/               # Downloaded model checkpoints (HF cache)
├── .cache/huggingface/   # HF model cache (persists, no re-download)
└── wandb/                # W&B run logs
```

## Scripts (in runpod/ directory of this repo)

| Script | When to run |
|--------|-------------|
| `runpod/setup.sh` | **FIRST TIME ONLY** — new pod, installs everything |
| `runpod/restart.sh` | **EVERY POD RESTART** — re-links auth, activates venv, reinstalls claude |
| `runpod/save.sh` | **BEFORE TERMINATE ONLY** — git push + wandb sync (not needed for stop) |

## Pod Lifecycle
- **Stop pod**: `/workspace/` survives. Run `restart.sh` on next start. ~$5/month idle.
- **Terminate pod**: Everything wiped. Run `save.sh` first, then `setup.sh` on new pod.

## Running Experiments
- **Always use tmux**: `tmux new -s exp` before starting long-running jobs
- **Always use wandb**: Log all experiments to Weights & Biases for tracking
- **Always use HF**: Download datasets/models via `huggingface-cli` so they cache in `/workspace/.cache/huggingface/`
- **Save results to /workspace/results/** with descriptive names
- **Commit code frequently** to `/workspace/code/personal-research/`

## Environment Variables
Always source before doing anything:
```bash
source /workspace/.env
source /workspace/venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export WANDB_DIR=/workspace
```

## Installed Simulation Frameworks
- **MuJoCo** — lightweight physics sim, runs on any GPU
- **robosuite** — robot manipulation suite (used by LIBERO)
- **LIBERO** — lifelong robot learning benchmark (90 tasks)
- **ManiSkill 3** — GPU-parallelized manipulation benchmark (needs Vulkan)
- **Isaac Lab** (optional) — NVIDIA sim framework (needs RTX GPU with RT cores)
- **gymnasium** — standard RL environments

## GPU Info
- This pod has an RTX 4090 (24GB VRAM)
- Supports Isaac Sim (has RT cores), ManiSkill (Vulkan), and all standard CUDA workloads
- Check GPU: `nvidia-smi`

## Common Commands
```bash
# Activate env
source /workspace/venv/bin/activate

# Run LIBERO eval
python -m libero.eval --benchmark libero_90 --policy <policy>

# Download model from HF
huggingface-cli download <model-id> --local-dir /workspace/models/<name>

# Start wandb experiment
wandb init -p robotics_experiments

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```
