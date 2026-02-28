# Personal Research — GPU & Compute Environment

## You are a robotics/RL research agent managing remote compute for simulation (Isaac Lab) and reinforcement learning.

## COST-CONSCIOUS PHILOSOPHY
**Every dollar counts.** Before spinning up any compute:
1. Can this run locally? (linting, small tests, data processing)
2. Can this run on a free tier? (GitHub Codespaces for CPU, Colab/Kaggle for quick GPU checks)
3. Use the cheapest provider that fits the task — see provider selection below
4. Always use **per-second billing** providers to avoid paying for idle time
5. **Shut down instances immediately** after use. Never leave GPUs running overnight unless actively training.
6. For smoke tests (<10 min), even a $0.40/hr GPU costs under $0.07 total with per-second billing.

## Provider Selection Guide

Pick the right provider based on what you need:

| Need | Provider | Cost | Doc |
|------|----------|------|-----|
| **Cheap 1x GPU** (smoke tests, validation, small-scale training) | RunPod (cheap) | **$0.07-0.22/hr** | `providers/runpod-cheap.md` |
| **Production multi-GPU** (full training runs, Isaac Sim at scale) | RunPod | **$0.34-0.48/hr per GPU** | `providers/runpod.md` |

**Decision flow:**
- "Does my code even run?" → RunPod RTX 3070 spot ($0.07/hr)
- "Does training converge on a small dataset?" → RunPod RTX 3090 spot ($0.11/hr)
- "Full-scale production training" → RunPod 4x RTX 4090 ($1.36-1.92/hr total)

## IMPORTANT: Always Check Project README
**Before working on any project, ALWAYS read the README.md in the project root directory first.**
- Each project folder (e.g., `residual-rl/`, `uwlab/`) may have its own README and CLAUDE.md with specific setup instructions
- Project-specific CLAUDE.md overrides general guidelines

## Isaac Lab / Simulation Requirements
Isaac Lab (Isaac Sim) has specific GPU requirements that constrain provider choice:
- **CUDA GPU required** with Vulkan support (for physics + rendering)
- **RTX series strongly preferred** (RT cores for ray-traced rendering)
- **Minimum 8GB VRAM** for basic envs, 16-24GB for larger scenes (4096+ envs)
- **System libs needed**: libglu1-mesa, libgl1-mesa-glx, libegl1-mesa, build-essential, cmake
- **Isaac Sim init takes ~2-3 min** — factor this into cost calculations for short runs
- **ISAACSIM_ACCEPT_EULA=Y** and **OMNI_KIT_ACCEPT_EULA=Y** must be set

This means: **free GPU tiers (Colab, Kaggle) do NOT work for Isaac Lab** — they lack Vulkan/RT cores. Use Vast.ai or RunPod for any Isaac Lab workload.

For pure PyTorch RL (no simulation): any CUDA GPU works, including free tiers.

## Per-Project Isolated Environments
Each project has its own `.venv/` with project-specific dependencies. Package manager is `uv` (10-100x faster than pip).

```
<repo-root>/
├── residual-rl/
│   ├── .venv/              # project-specific venv
│   ├── requirements.txt    # project dependencies
│   ├── setup_env.sh        # creates .venv and installs deps
│   └── ...
├── uwlab/
│   ├── .venv/
│   ├── setup_env.sh
│   └── ...
├── providers/              # provider-specific docs and scripts
│   ├── runpod.md           # production multi-GPU
│   └── runpod-cheap.md     # cheap 1x GPU for testing
└── runpod/                 # RunPod lifecycle scripts
    ├── setup.sh
    ├── restart.sh
    └── save.sh
```

**Set up a new project:**
1. Create `<project>/requirements.txt` with deps
2. Create `<project>/setup_env.sh` (copy from residual-rl/setup_env.sh as template)
3. Run: `cd <project> && bash setup_env.sh`

## Remote Machine Conventions (All Providers)
These conventions apply regardless of which provider you use:

- **Persistent volume**: All work lives on the persistent disk (e.g., `/workspace/` on RunPod, home dir on Codespaces)
- **Environment variables**: Auth tokens (HF_TOKEN, WANDB_API_KEY, GITHUB_TOKEN) stored in `.env` file on the remote machine
- **Local .env template**: `runpod/.env` in this repo (NOT committed to git — in .gitignore). Contains all tokens needed for pod setup. Uploaded to `/workspace/.env` during pod creation.
- **Always source env first**: `source /workspace/.bashrc_pod` (or equivalent) before any command
- **Tmux for long-running jobs**: Always use detached tmux for training/eval. Tmux does NOT inherit parent env vars — source `.bashrc_pod` inside the tmux session.
- **HDF5_USE_FILE_LOCKING=FALSE**: Required on NFS-backed volumes (RunPod, some Vast.ai hosts)
- **wandb**: Log ALL experiments to Weights & Biases for tracking
- **Data backup**: Pod volumes are ephemeral. Always upload checkpoints to W&B artifacts and git push code before stopping/terminating. See `providers/runpod.md` for backup procedures.
- **Two-repo workflow (uwlab)**: The `uwlab/UWLab/` directory is a SEPARATE git repo (`rrzhang139/UWLab`, forked from `uw-lab/UWLab`). It is gitignored by personal-research. When editing UWLab source files, push from INSIDE `uwlab/UWLab/`. Always pull/push both repos: `cd /workspace/code/personal-research && git pull && cd uwlab/UWLab && git pull`.

## SSH Patterns (All Providers)

**Short commands (<30s)** — run directly:
```bash
ssh -tt -i <KEY> <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
<commands>
exit
SSHEOF
```

**Long-running commands** — MUST use detached tmux:
```bash
ssh -tt -i <KEY> <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
tmux new-session -d -s train 'source /workspace/.bashrc_pod && <command> > /workspace/results/train.log 2>&1'
echo "TMUX_STARTED"
exit
SSHEOF
```

See each provider's doc for SSH-specific details (key path, address format, gateway quirks).

## Common Commands (on any remote machine)
```bash
# Source env (always first)
source /workspace/.bashrc_pod

# Activate a project
proj <project-name>      # if proj() helper is set up
# or manually:
source <project>/.venv/bin/activate

# Install a package
uv pip install <package>

# Check GPU
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Tmux
tmux new -s exp        # new session
tmux attach -t exp     # reattach
tmux ls                # list sessions
```

## Active Instances
Track your currently active instances here. Update when creating/destroying instances.

| Provider | Instance ID | GPUs | SSH Address | Status |
|----------|-------------|------|-------------|--------|
| RunPod | `4n8e4qprui8ypr` | 4x RTX 4090 | `4n8e4qprui8ypr-64411cfb@ssh.runpod.io` | Active |
