# RunPod — Production GPU Provider

## When to Use
- Full-scale training runs (multi-GPU, long duration)
- Isaac Lab training with 4096+ parallel envs
- Workloads requiring 4x RTX 4090 or similar high-end configs

## Pricing
- **RTX 4090**: $0.34-0.48/hr (community cloud), $0.17-0.24/hr (spot)
- **RTX 3090**: $0.22/hr (on-demand), $0.11/hr (spot)
- **A100 80GB**: $0.79/hr (community)
- Billed per-second (actually per-millisecond). No egress fees.
- Stopped pods: ~$5/month for volume storage

**Cost awareness**: A 4x RTX 4090 pod costs ~$1.36-1.92/hr. That's $32-46/day if left running. **Always stop pods when not actively training.**

## Current Instance
- **Pod ID**: `1wlyxqrt37safq` (uwlab-4x4090-v3)
- **GPUs**: 4x RTX 4090
- **SSH**: `1wlyxqrt37safq-64411e63@ssh.runpod.io`
- **SSH Key**: `~/.ssh/runpod`
- **Old Pod (EXITED)**: `2djfma2zu7g1oh` (1x RTX 4090)

## Volume Persistence
- **ALL work lives in `/workspace/`** — this is the persistent volume disk
- The container disk (`/root/`, `/usr/`, etc.) is WIPED on every pod stop/restart
- Never store anything important outside `/workspace/`
- ALL caches (uv, pip, HF, wandb) redirected to `/workspace/.cache/` — container disk is only 5GB
- Binaries (uv) installed to `/workspace/.local/bin/`

## Directory Layout
```
/workspace/
├── .env                  # Auth tokens (HF, GitHub, W&B) — source this
├── .bashrc_pod           # Paths, auth, proj() helper — NO venv activation
├── .claude/              # Claude Code auth (symlinked from ~/.claude)
├── code/                 # All git repos
│   ├── personal-research/  # This repo (projects + runpod scripts)
│   └── LIBERO/             # LIBERO benchmark (if needed)
├── datasets/             # Downloaded datasets
│   └── maniskill_demos/  # ManiSkill3 demos (downloaded + converted)
├── results/              # Experiment outputs, CSVs, rollout videos
├── checkpoints/          # Model checkpoints (downloaded from W&B artifacts)
├── models/               # Downloaded model checkpoints (legacy)
├── .cache/huggingface/   # HF model cache (persists, no re-download)
└── wandb/                # W&B run logs
```

## Scripts (in runpod/ directory of this repo)

| Script | When to run |
|--------|-------------|
| `runpod/setup.sh` | **FIRST TIME ONLY** — system tools, uv, auth, Claude Code |
| `runpod/restart.sh` | **EVERY POD RESTART** — reinstalls system packages + Claude Code |
| `runpod/save.sh` | **BEFORE TERMINATE** — quick git push + wandb sync |
| `runpod/offboard.sh` | **BEFORE TERMINATE** — full backup with fallbacks (wandb → HF → local download) |
| `<project>/setup_env.sh` | **FIRST TIME PER PROJECT** — creates .venv, installs deps |

## Container Disk vs Volume
The container disk (`/`, 2GB) is wiped on every pod stop/restart. Only these items need reinstalling:

| Item | Why can't persist | Reinstall method |
|------|-------------------|------------------|
| `tmux` | apt package with system libs | `apt-get install -y tmux` |
| `libglu1-mesa` | system shared lib for Isaac Sim rendering | `apt-get install -y libglu1-mesa` |
| `libgl1-mesa-glx`, `libegl1-mesa` | system OpenGL libs | `apt-get install -y ...` |
| `build-essential`, `cmake` | compilers/build tools | `apt-get install -y ...` |
| `dev` user | `/etc/passwd` on container disk | `useradd -m -s /bin/bash dev` |

Everything else (uv, claude, Python venvs, caches, code, checkpoints) lives on `/workspace/`.
`restart.sh` handles all of the above automatically (~30s).

Cache symlinks (`/root/.cache/ov → /workspace/.cache/ov`, etc.) are set up by `.bashrc_pod` to avoid 471MB Omniverse cache rebuilding on each restart.

## Pod Lifecycle
- **Stop pod**: `/workspace/` survives (project venvs, packages, models, code all intact). Run `restart.sh` on next start (~1 min). ~$5/month idle.
- **Terminate pod**: Everything wiped. Run `save.sh` first, then `setup.sh` + project `setup_env.sh` on new pod.

## Claude Code on the Pod

### Option A: Interactive (on pod)
- Auth saved to `/workspace/.claude/`, persists through stop/restart
- If OAuth scope error, downgrade: `npm install -g @anthropic-ai/claude-code@2.1.19`
- If npm install OOMs, add swap first: `fallocate -l 4G /workspace/swapfile && chmod 600 /workspace/swapfile && mkswap /workspace/swapfile && swapon /workspace/swapfile`

**Running with `--dangerously-skip-permissions`:**
RunPod runs as root, but Claude Code blocks `--dangerously-skip-permissions` as root. Use the `dev` user instead:
```bash
# Setup (run once as root, already done if setup.sh was run):
useradd -m -s /bin/bash dev 2>/dev/null
cp /root/.local/bin/claude /usr/local/bin/claude 2>/dev/null
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

### Option B: Remote Control (local machine via SSH)
Claude Code runs locally and executes commands on the pod via SSH heredoc.

**CRITICAL: SSH Gateway Quirks**
- RunPod's SSH gateway requires `-tt` (forced PTY) and **ignores command arguments** — commands MUST be piped via stdin/heredoc
- SSH suffix (e.g., `-64411e63`) is **per-pod and unique** — get from RunPod dashboard, not derivable from API
- New pod SSH may take 1-3 minutes to become reachable after creation
- If SSH port stays closed for >5 min, the pod is likely stuck — terminate and create new

**Two timeout layers for SSH heredoc:**
1. Claude Code's bash tool timeout (default 2 min, max 10 min)
2. RunPod's SSH gateway drops idle connections

When either fires, the SSH session dies and **the command on the pod dies with it** — unless it was launched in tmux or nohup.

**Short commands (<30s):**
```bash
ssh -tt -i ~/.ssh/runpod SSH_ADDRESS << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
proj residual-rl
git pull
exit
SSHEOF
```

**Long-running commands — MUST use detached tmux:**
```bash
# Step 1: Launch in detached tmux (returns immediately)
ssh -tt -i ~/.ssh/runpod SSH_ADDRESS << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
proj residual-rl
tmux new-session -d -s train 'source /workspace/.bashrc_pod && proj residual-rl && python train.py > /workspace/results/train.log 2>&1'
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

**Alternative — nohup:**
```bash
ssh -tt -i ~/.ssh/runpod SSH_ADDRESS << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
nohup python train.py > /workspace/results/train.log 2>&1 &
echo "PID: $!"
exit
SSHEOF
```

## Environment Variables in Tmux
**Tmux creates a new shell that does NOT inherit environment variables from the parent shell.**

Variables like `WANDB_API_KEY`, `HF_TOKEN`, `GITHUB_TOKEN` from `/workspace/.env` will be MISSING unless explicitly sourced.

**Always source the environment inside tmux sessions:**
```bash
tmux new -s exp
source /workspace/.bashrc_pod  # This loads /workspace/.env
proj residual-rl               # This activates the venv
```

**Symptoms of missing environment:**
- `wandb: ERROR API key not configured` despite having `.env` file
- HuggingFace downloads fail with authentication errors
- Git push fails with authentication errors

**Fix:** Always run `source /workspace/.bashrc_pod` as the first command inside every new tmux session.

## Memory Note
After a fresh pod restart, `free -h` may show ~76GB "used" out of 124GB. This is normal:
- ~15GB is Linux disk cache (buff/cache) — instantly reclaimable
- ~60GB is NVIDIA GPU driver + CUDA runtime mapped into system RAM
- Check the `available` column (not `used`) — typically ~46GB free for your processes

## RunPod API Management
API key stored in `~/.runpod/config.toml`. Manage pods programmatically:
```bash
RUNPOD_API_KEY="$(grep apikey ~/.runpod/config.toml | cut -d'"' -f2)"

# List pods
curl -s -H "Content-Type: application/json" -d '{"query":"query { myself { pods { id name desiredStatus gpuCount } } }"}' "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"

# Stop a pod
curl -s -H "Content-Type: application/json" -d '{"query":"mutation { podStop(input: {podId: \"POD_ID\"}) { id desiredStatus } }"}' "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"

# Resume a stopped pod
curl -s -H "Content-Type: application/json" -d '{"query":"mutation { podResume(input: {podId: \"POD_ID\", gpuCount: 4}) { id desiredStatus } }"}' "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"

# Create new pod
curl -s -H "Content-Type: application/json" -d '{"query":"mutation { podFindAndDeployOnDemand(input: { name: \"my-pod\", gpuTypeId: \"NVIDIA GeForce RTX 4090\", gpuCount: 4, cloudType: ALL, volumeInGb: 100, containerDiskInGb: 20, imageName: \"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04\", volumeMountPath: \"/workspace\", ports: \"22/tcp,8888/http\" }) { id desiredStatus } }"}' "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"

# Terminate (destroys volume!)
curl -s -H "Content-Type: application/json" -d '{"query":"mutation { podTerminate(input: {podId: \"POD_ID\"}) }"}' "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"

# Get SSH command
runpodctl ssh connect POD_ID
```

**CRITICAL API notes:**
- Cannot change GPU count on existing pod — must create new
- `podResume` fails with "not enough free GPUs" if host machine is full — create new pod instead
- Pod volumes are pod-local (NOT network volumes) — cannot be shared between pods
- New pods start with empty `/workspace/` — must run full setup (setup.sh + project setup_env.sh)

## CRITICAL: Data Backup — Pods Can Be Destroyed
**Pod volumes are NOT permanent.** If a pod is terminated (intentionally or by RunPod), all data on `/workspace/` is gone. Community cloud hosts can also fail unexpectedly.

**Rule: Never store precious artifacts only on a pod volume. Always have an external copy.**

### What to Back Up
| Artifact | Where to Store | How |
|----------|---------------|-----|
| **Model checkpoints** (model_*.pt) | W&B Artifacts | `wandb.log_artifact()` or `wandb artifact put` |
| **Training metrics/logs** | W&B (automatic) | Already logged if using `--logger wandb` |
| **Eval videos** | W&B | Upload via `scripts/upload_eval_wandb.py` |
| **Code changes** | GitHub | `git push` before stopping pod |
| **Datasets** | HuggingFace Hub / S3 | Re-downloadable, don't need backup |
| **Config files** | In repo (git) | Commit to personal-research |

### Best Practices (What ML Researchers Do)
1. **W&B Artifacts for checkpoints**: Standard practice. Upload every N steps + final checkpoint. Download with `wandb.use_artifact()`.
2. **Git push frequently**: Code should never only exist on a pod. Push after every meaningful change.
3. **`save.sh` before terminate**: The `runpod/save.sh` script does git push + wandb sync. **Always run it before termination.**
4. **Treat the pod as ephemeral**: Assume it can die at any time. If your only copy of a trained model is on `/workspace/`, you're one accident away from losing it.
5. **Checkpointing during training**: Save checkpoints every N iterations (not just at the end). If the pod dies mid-training, you lose everything since the last checkpoint.

### Quick Backup Commands
```bash
# Upload a checkpoint to W&B
wandb artifact put --name my-checkpoint --type model /workspace/checkpoints/model_best.pt

# Download a checkpoint from W&B
wandb artifact get USER/PROJECT/my-checkpoint:latest --root /workspace/checkpoints/

# Git push all code
cd /workspace/code/personal-research && git add -A && git commit -m "backup" && git push

# Run the quick save script (git push + wandb sync)
bash /workspace/code/personal-research/runpod/save.sh
```

### Before Terminating Any Pod — Use offboard.sh
The `runpod/offboard.sh` script automates the full backup process with fallbacks:
1. Git push all code
2. Sync wandb runs
3. Upload checkpoints to W&B artifacts (fallback: HuggingFace Hub, fallback: runpodctl download)
4. Report summary

```bash
# Run in tmux (it does uploads that can take minutes):
tmux new -s offboard
source /workspace/.bashrc_pod
cd /workspace/code/personal-research/uwlab && source .venv/bin/activate
export CHECKPOINT_DIR="/workspace/code/personal-research/uwlab/UWLab/logs/rsl_rl"
bash /workspace/code/personal-research/runpod/offboard.sh
```

Wait for "All data safely offboarded" before stopping/terminating.

## Getting SSH Address from API
The SSH gateway address is in the `machine.podHostId` field:
```bash
RUNPOD_API_KEY="$(grep apikey ~/.runpod/config.toml | cut -d'"' -f2)"
curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"query":"{ pod(input: {podId: \"POD_ID\"}) { machine { podHostId } } }"}'
# SSH: ssh -tt -i ~/.ssh/runpod <podHostId>@ssh.runpod.io
```

## File Transfer (RunPod <-> Local)
RunPod's SSH gateway blocks `scp`, `rsync`, and port forwarding. Use `runpodctl send/receive` instead:

```bash
# On the pod: send a file (outputs a receive code)
runpodctl send /workspace/path/to/file.mp4

# On local machine: receive using the code
runpodctl receive <code>   # e.g. runpodctl receive 6641-logo-hilton-diego-3
```

Install locally: `brew install runpod/runpodctl/runpodctl`
Already installed on pods at `/usr/bin/runpodctl`.
