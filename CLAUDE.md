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

## Multi-Repo Layout

Each research project lives in its own standalone repo at `~/workspace/<project>/` (on pods: `/workspace/code/<project>/`).
This hub repo contains only shared GPU/SSH/RunPod infrastructure.

| Project | Repo | Description |
|---------|------|-------------|
| AlphaZero | `rrzhang139/alphago` | AlphaZero from scratch |
| Quake3 WM | `rrzhang139/quake3-worldmodel` | World model for Q3 Arena |
| Residual RL | `rrzhang139/residual-rl` | Policy Decorator |
| UWLab | `rrzhang139/uwlab-omnireset` | OmniReset for UWLab |

Each project's CLAUDE.md references `../personal-research/` for shared infra docs.

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
Each project repo has its own `.venv/` with project-specific dependencies. Package manager is `uv` (10-100x faster than pip).

```
~/workspace/                         # local layout
├── personal-research/               # this hub repo — shared infra only
│   ├── CLAUDE.md                    # GPU/SSH/RunPod conventions
│   ├── providers/                   # provider docs (pricing, setup)
│   ├── runpod/                      # pod lifecycle scripts
│   └── .claude/commands/            # slash commands
├── alphago/                         # standalone project repo
├── quake3-worldmodel/               # standalone project repo
├── residual-rl/                     # standalone project repo
└── uwlab-omnireset/                 # standalone project repo
    └── UWLab/                       # separate fork (cloned inside)
```

**Set up a project on a pod:**
1. Clone the project repo: `cd /workspace/code && git clone https://github.com/rrzhang139/<project>.git`
2. Run: `cd <project> && bash setup_env.sh`

## Remote Machine Conventions (All Providers)
These conventions apply regardless of which provider you use:

- **Persistent volume**: All work lives on the persistent disk (e.g., `/workspace/` on RunPod, home dir on Codespaces)
- **Environment variables**: Auth tokens (HF_TOKEN, WANDB_API_KEY, GITHUB_TOKEN) stored in `.env` file on the remote machine
- **Local .env template**: `runpod/.env` in this repo (NOT committed to git — in .gitignore). Contains all tokens needed for pod setup. Uploaded to `/workspace/.env` during pod creation.
- **Always source env first**: `source /workspace/.bashrc_pod` (or equivalent) before any command
- **Tmux for long-running jobs**: Always use detached tmux for training/eval. Tmux does NOT inherit parent env vars — source `.bashrc_pod` inside the tmux session.
- **HDF5_USE_FILE_LOCKING=FALSE**: Required on NFS-backed volumes (RunPod, some Vast.ai hosts)
- **wandb**: Log ALL experiments to Weights & Biases for tracking. 
- **MANDATORY: W&B Artifacts**: **ALWAYS** upload model checkpoints (best.pt, final.pt) to W&B Artifacts after every training run. Do this BEFORE stopping/terminating a pod. Pod volumes are ephemeral — if you don't upload, the checkpoint is lost. Use: `wandb.log_artifact(artifact)` with `type='model'`. This is non-negotiable.
- **Data backup**: Pod volumes are ephemeral. Always upload checkpoints to W&B artifacts and git push code before stopping/terminating. See `providers/runpod.md` for backup procedures.


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
proj <project-name>      # if proj() helper is set up (activates /workspace/code/<project>/.venv)
# or manually:
cd /workspace/code/<project> && source .venv/bin/activate

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
| RunPod   | j4n743n88ra45a | RTX 4090 | j4n743n88ra45a-644117b7@ssh.runpod.io | RUNNING (go9-v2 training, ~$0.34/hr) |
| RunPod   | noecq0fv7nifkx | RTX 3090 | noecq0fv7nifkx-64410ee7@ssh.runpod.io | RUNNING (wm-train-mixed, ~$0.22/hr) |
