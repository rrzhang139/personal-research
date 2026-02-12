# Personal Research — RunPod GPU Environment

## You are a robotics research agent working on a RunPod GPU instance.

## IMPORTANT: Always Check Project README
**Before working on any project, ALWAYS read the README.md in the project root directory first.**
- Each project folder (e.g., `residual-rl/`) may have its own README with specific setup instructions, usage notes, and important details
- The README contains project-specific information that overrides general guidelines
- After reading the README, proceed with the task using the project-specific context

## Critical: Volume Persistence
- **ALL work lives in `/workspace/`** — this is the persistent volume disk
- The container disk (`/root/`, `/usr/`, etc.) is WIPED on every pod stop/restart
- Never store anything important outside `/workspace/`
- Package manager is `uv` (10-100x faster than pip). Use `uv pip install` instead of `pip install`
- ALL caches (uv, pip, HF, wandb) redirected to `/workspace/.cache/` — container disk is only 5GB
- Binaries (uv) installed to `/workspace/.local/bin/`

## Per-Project Isolated Environments
Each project has its own `.venv/` with project-specific dependencies. uv caches packages so torch etc. are only downloaded once but hardlinked into each venv (no disk waste).

```
/workspace/code/personal-research/
├── residual-rl/
│   ├── .venv/              # project-specific venv (torch, mani-skill, diffusers, etc.)
│   ├── requirements.txt    # project dependencies
│   ├── setup_env.sh        # creates .venv and installs deps
│   └── ...
├── future-project/
│   ├── .venv/
│   ├── requirements.txt
│   └── setup_env.sh
└── runpod/
    ├── setup.sh            # system-level only (uv, apt, auth, Node.js, Claude Code)
    ├── restart.sh           # pod restart recovery
    └── save.sh              # pre-terminate backup
```

**Activate a project:**
```bash
source /workspace/.bashrc_pod
proj residual-rl            # shortcut: activates .venv + cd into project dir
# or manually:
source /workspace/code/personal-research/residual-rl/.venv/bin/activate
```

**Set up a new project:**
1. Create `<project>/requirements.txt` with deps
2. Create `<project>/setup_env.sh` (copy from residual-rl/setup_env.sh as template)
3. Run: `cd <project> && bash setup_env.sh`

## Directory Layout
```
/workspace/
├── .env                  # Auth tokens (HF, GitHub, W&B) — source this
├── .bashrc_pod           # Paths, auth, proj() helper — NO venv activation
├── .claude/              # Claude Code auth (symlinked from ~/.claude)
├── code/                 # All git repos
│   ├── personal-research/  # This repo (projects + runpod scripts)
│   └── LIBERO/             # LIBERO benchmark (if needed)
├── datasets/             # Downloaded datasets (see Datasets & Checkpoints section below)
│   └── maniskill_demos/  # ManiSkill3 demos (downloaded + converted)
├── results/              # Experiment outputs, CSVs, rollout videos
├── checkpoints/          # Model checkpoints (downloaded from W&B artifacts)
├── models/               # Downloaded model checkpoints (legacy)
├── .cache/huggingface/   # HF model cache (persists, no re-download)
└── wandb/                # W&B run logs
```

## Datasets & Checkpoints (W&B Artifacts)

### ManiSkill3 Demos
**Download demos:**
```bash
python -m mani_skill.utils.download_demo "PegInsertionSide-v1"
# Saved to: ~/.maniskill/demos/PegInsertionSide-v1/
# Default control mode: pd_joint_delta_pos
```

**Convert to pd_ee_delta_pose (end-effector control):**
```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
  --save-traj \
  -c pd_ee_delta_pose \
  -o state

# Output: trajectory.state.pd_ee_delta_pose.physx_cpu.h5
# Location: /workspace/datasets/maniskill_demos/PegInsertionSide-v1/motionplanning/
```

### W&B Artifacts
**Pretrained diffusion policy checkpoint:**
- Project: `rzhang139/policy_decorator`
- Artifact: `diffusion_base_policy:latest`
- Contents: `best.pt` (checkpoint), `args.json` (training config)
- Download:
  ```bash
  wandb artifact get rzhang139/policy_decorator/diffusion_base_policy:latest \
    --root /workspace/checkpoints
  ```

## Scripts (in runpod/ directory of this repo)

| Script | When to run |
|--------|-------------|
| `runpod/setup.sh` | **FIRST TIME ONLY** — system tools, uv, auth, Claude Code |
| `runpod/restart.sh` | **EVERY POD RESTART** — reinstalls system packages + Claude Code |
| `runpod/save.sh` | **BEFORE TERMINATE ONLY** — git push + wandb sync |
| `<project>/setup_env.sh` | **FIRST TIME PER PROJECT** — creates .venv, installs deps |

## Pod Lifecycle
- **Stop pod**: `/workspace/` survives (project venvs, packages, models, code all intact). Run `restart.sh` on next start (~1 min). ~$5/month idle.
- **Terminate pod**: Everything wiped. Run `save.sh` first, then `setup.sh` + project `setup_env.sh` on new pod.

## Two Ways to Use Claude Code with This Pod

### Option A: Claude Code on the Pod (Interactive)
- Auth saved to `/workspace/.claude/`, persists through stop/restart
- If OAuth scope error, downgrade: `npm install -g @anthropic-ai/claude-code@2.1.19`
- If npm install OOMs, add swap first: `fallocate -l 4G /workspace/swapfile && chmod 600 /workspace/swapfile && mkswap /workspace/swapfile && swapon /workspace/swapfile`

**Running with `--dangerously-skip-permissions`:**
RunPod runs as root, but Claude Code blocks `--dangerously-skip-permissions` as root. Use the `dev` user instead:
```bash
# Setup (run once as root, already done if setup.sh was run):
useradd -m -s /bin/bash dev 2>/dev/null
cp /root/.local/bin/claude /usr/local/bin/claude 2>/dev/null  # if installed via native installer
chmod 755 /usr/local/bin/claude 2>/dev/null
ln -sfn /workspace/.claude /home/dev/.claude
chmod -R 777 /workspace/.claude

# Run Claude Code as dev user:
su - dev
source /workspace/.bashrc_pod
cd /workspace/code/personal-research
claude --dangerously-skip-permissions

# Or resume a session:
su - dev -c 'source /workspace/.bashrc_pod && cd /workspace/code/personal-research && claude --resume SESSION_ID --dangerously-skip-permissions'
```

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
ssh -tt -i ~/.ssh/runpod SSH_ADDRESS << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
proj residual-rl
git pull
exit
SSHEOF
```

**Long-running commands** — training, heavy imports, downloads — MUST use detached tmux:
```bash
# Step 1: Launch in detached tmux (returns immediately)
ssh -tt -i ~/.ssh/runpod SSH_ADDRESS << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
proj residual-rl
tmux new-session -d -s train 'python offline/diffusion_policy_unet_maniskill2.py --env-id StackCube-v1 > /workspace/results/train.log 2>&1'
echo "TMUX_STARTED"
exit
SSHEOF

# Step 2: Check output later (separate SSH call)
ssh -tt -i ~/.ssh/runpod SSH_ADDRESS << 'SSHEOF'
tail -30 /workspace/results/train.log
tmux ls
exit
SSHEOF
```

**Alternative for long commands — nohup:**
```bash
ssh -tt -i ~/.ssh/runpod SSH_ADDRESS << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
proj residual-rl
nohup python offline/diffusion_policy_unet_maniskill2.py --env-id StackCube-v1 \
  > /workspace/results/train.log 2>&1 &
echo "PID: $!"
exit
SSHEOF
```

**Note:** Replace `SSH_ADDRESS` with the current pod SSH address (changes every time a new pod is created). Find it in the RunPod dashboard.

## Running Experiments
- **Always use tmux**: `tmux new -s exp` before starting long-running jobs
- **Always use wandb**: Log all experiments to Weights & Biases for tracking
- **Save results to /workspace/results/** with descriptive names
- **Commit code frequently** to `/workspace/code/personal-research/`
- **Always export `HDF5_USE_FILE_LOCKING=FALSE`** — RunPod's NFS-backed `/workspace/` volume doesn't support h5py file locking; without this, any HDF5 read/write will fail with `BlockingIOError: Unable to lock file`
- **ManiSkill demos need conversion**: Raw demos from `mani_skill.utils.download_demo` use `obs_mode='none'`. Before training, replay them with `python -m mani_skill.trajectory.replay_trajectory --traj-path <path> -o state -c <control_mode> --save-traj` to get flat state observations and the desired control mode. Converted files are saved alongside the original with a suffix like `.state.pd_ee_delta_pose.physx_cpu.h5`

## Memory Note
After a fresh pod restart, `free -h` may show ~76GB "used" out of 124GB. This is normal:
- ~15GB is Linux disk cache (buff/cache) — instantly reclaimable
- ~60GB is NVIDIA GPU driver + CUDA runtime mapped into system RAM
- Check the `available` column (not `used`) — typically ~46GB free for your processes

## Installed Simulation Frameworks (per residual-rl env)
- **MuJoCo** — lightweight physics sim, runs on any GPU
- **ManiSkill 3** — GPU-parallelized manipulation benchmark (needs Vulkan)
- **gymnasium** — standard RL environments
- **stable-baselines3** — RL algorithm implementations

## GPU Info
- This pod has an RTX 4090 (24GB VRAM)
- Supports Isaac Sim (RT cores), ManiSkill (Vulkan), and all standard CUDA workloads
- Check GPU: `nvidia-smi`

## Residual-RL Project (Policy Decorator) — Key Learnings

### Architecture & Code
- **Original codebase**: ManiSkill2 (`PegInsertionSide-v2`). We ported to MS3 (`PegInsertionSide-v1`)
- **MS2 obs_dim=50, MS3 obs_dim=43**: The 7-dim difference is `base_pose` (robot base pos `[-0.615,0,0]` + identity quat `[1,0,0,0]`) at indices 18-24. MS3 removed it because it's constant for stationary robots
- **To use original MS2 pretrained checkpoint on MS3**: Pad observations with the constant `base_pose=[-0.615, 0, 0, 1, 0, 0, 0]` at index 18 using `--pad-obs-to 50`. Do NOT pad with zeros
- **Pretrained checkpoints**: Download from [Google Drive](https://drive.google.com/drive/folders/1W0jwgVP9W1odt_F6vrwtYcnksdbwF1gi?usp=sharing) or W&B artifact `rzhang139/policy_decorator/diffusion_base_policy:latest`. Put under `./checkpoints/`
- **PYTHONPATH required**: Running `python offline/script.py` adds `offline/` not project root to sys.path. Always `export PYTHONPATH=/workspace/code/personal-research/residual-rl:$PYTHONPATH`
- **`proj` command unavailable in tmux**: Use explicit `source .venv/bin/activate` in tmux scripts instead

### Training Speed & Batch Size
- **~950 iters/min with bs=1024** on RTX 4090 is **expected and correct**. 1M iters ≈ 17.5 hours
- **Larger batch size is SLOWER in wall-clock**: bs=4096 → ~420 iters/min (40 hrs for 1M). More samples/min but fewer iters/min, and convergence doesn't improve per-iter
- **Keep bs=1024** (the paper default). The paper notes "2048 does not further improve"
- **Offline training does NOT benefit from GPU-parallelized envs**: Data is pre-loaded to GPU memory. Bottleneck is purely neural net forward/backward
- **Eval is expensive**: Each eval runs 100 episodes. Use `--eval-freq 100000` for long runs, not 1000

### Potential Speedups
- **Reduce total iterations**: Loss plateaus well before 1M. Try 300k-500k, check loss curve
- **`torch.compile()`**: Could speed up UNet forward/backward by 20-40% (untested)
- **Mixed precision (AMP)**: Marginal gains for this small model (4.5M params)
- **Reduce eval frequency**: `--eval-freq 100000` instead of 1000 saves hours
- **Run one job at a time**: Two GPU jobs share compute and both slow down ~50%

### MS3 Compatibility Fixes Applied
- `CPUNumpyWrapper`: Converts MS3 torch tensor obs/rewards to numpy for AsyncVectorEnv compatibility
- `PadObsWrapper`: Pads MS3 43-dim obs to MS2 50-dim with constant base_pose for using pretrained MS2 checkpoints
- Action normalization (`action_mean`/`action_std`): Added in commit `0601e38`. Original pretrained checkpoints do NOT use normalization
- `envs/maniskill_fixed.py`: Registers fixed MS3 environments for policy decorator

### Key File Locations
| File | Description |
|------|-------------|
| `offline/diffusion_policy_unet_maniskill2.py` | Base diffusion policy training (offline IL) |
| `online/pi_dec_diffusion_maniskill2.py` | Policy Decorator residual RL (online) |
| `/workspace/checkpoints/best.pt` | Pretrained MS2 diffusion policy (from W&B) |
| `/workspace/checkpoints/best_adapted_ms3.pt` | Adapted checkpoint (MS2→MS3, obs weights reinitialized) |
| `/workspace/datasets/maniskill_demos/PegInsertionSide-v1/` | MS3 converted demos |
| `scripts/run_*.sh` | Launch scripts for various experiments |

### SSH Address
Current RunPod SSH: `2djfma2zu7g1oh-644112fd@ssh.runpod.io`

## Common Commands
```bash
# Source env (always first)
source /workspace/.bashrc_pod

# Activate a project
proj residual-rl

# Install a new package into current project venv
uv pip install <package>

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Tmux
tmux new -s exp        # new session
tmux attach -t exp     # reattach
tmux ls                # list sessions
```
