# UWLab Project (OmniReset)

## Overview
UWLab is a robotics simulation framework built on Isaac Sim for dexterous manipulation tasks.
OmniReset is a method for solving contact-rich manipulation tasks without reward engineering or demos.

## Setup
- **Python**: 3.11 (required for Isaac Sim 5.X)
- **Isaac Sim**: 5.1.0 (pip install, ~12GB)
- **PyTorch**: 2.7.0 with CUDA 12.8
- **RL Library**: rsl_rl (installed via `./uwlab.sh --install rsl_rl`)
- **Tracking**: wandb

### First-Time Setup
```bash
cd /workspace/code/personal-research/uwlab
bash setup_env.sh
```

### Activate Environment
```bash
source /workspace/code/personal-research/uwlab/.venv/bin/activate
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
cd /workspace/code/personal-research/uwlab/UWLab
```

## Key Learnings & Difficulties

### Installation
- Isaac Sim 5.1.0 is ~12GB download via pip. Use `uv pip install` for speed
- MUST set `ISAACSIM_ACCEPT_EULA=Y` and `OMNI_KIT_ACCEPT_EULA=Y` before running
- First run pulls extensions and can take 10+ minutes
- Pod restarts wipe tmux — always reinstall: `apt-get install -y tmux`
- Pod restarts also wipe libGLU — reinstall: `apt-get install -y libglu1-mesa` (needed for Isaac Sim rendering/video)
- GLIBC 2.35+ required (RunPod Ubuntu 22.04 has 2.35)
- System requirements: CUDA 12.4+ driver (RunPod has 12.7), RTX 4090 supported
- isaacsim install takes ~10 min with uv. MUST run in tmux (SSH timeout kills it otherwise)
- uv cache grows to ~41GB after isaacsim install. With hardlinks, venv (~20GB) shares storage. Total real usage ~40-45GB
- After install, clean cache with `UV_CACHE_DIR=/workspace/.cache/uv uv cache clean` if disk is tight

### CRITICAL: Environment Variables in tmux/subshells
- `/workspace/.env` defines vars like `WANDB_API_KEY` **without `export`**
- `.bashrc_pod` must use `set -a; source /workspace/.env; set +a` to auto-export them
- Without this, Python processes (wandb, etc.) cannot see the keys — they're shell-only vars
- **Symptom**: `wandb.errors.UsageError: No API key configured` despite key being set in shell
- **Fix applied**: `.bashrc_pod` now uses `set -a` before sourcing `.env`
- `.bashrc_pod` also sources a legacy `/workspace/venv/bin/activate` that may not exist — scripts with `set -e` will crash. Use `2>/dev/null` or don't use `set -e`

### Training Scripts
- **Do NOT use `set -e`** in training scripts that source `.bashrc_pod` — legacy venv path causes exit
- Always inline commands in tmux rather than calling shell scripts with `set -e`
- The `--checkpoint` flag in train.py sets `agent_cfg.load_checkpoint` but checkpoint is only loaded if `--resume` is also passed (or `agent_cfg.resume=True`). Without `--resume`, training starts from scratch regardless of `--checkpoint`
- Default `resume: false` in `agents/rsl_rl_cfg.py` — must explicitly pass `--resume` to finetune
- wandb project is set by `WANDB_PROJECT` env var for `wandb.init()`, but rsl_rl's runner uses `agent_cfg.wandb_project` which defaults to `isaaclab`. Pass `--log_project_name omnireset` to override

### Running Headless (No GUI)
- All scripts must use `--headless` flag
- Isaac Sim runs headless via `--enable livestream` or just `--headless`
- No X server needed on RunPod

### OmniReset Training Pipeline
1. Collect partial assemblies (~30s)
2. Sample grasp poses (~1 min)
3. Generate reset state datasets (1 min - 1 hr)
4. Train RL policy

### Single GPU Adaptation
- Original paper uses 4x L40S GPUs with `torch.distributed.run --nproc_per_node 4`
- For 1 GPU: remove `--distributed` and `torch.distributed.run`, reduce `--num_envs` (4096 vs 16384)
- 4096 envs uses ~6GB VRAM on RTX 4090 (plenty of headroom)
- ~15s per iteration, ~4 hours for full 40k iterations
- Reset types: ObjectAnywhereEEAnywhere, ObjectRestingEEGrasped, ObjectAnywhereEEGrasped, ObjectPartiallyAssembledEEGrasped
- Config location: `UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/reset_states/config/ur5e_robotiq_2f85/rl_state_cfg.py`
- Near-goal config (2 resets only): `rl_state_near_goal_cfg.py` (same directory)
- Backup of original config: `rl_state_cfg.py.bak`

### Reset Type Details
| Reset Type | Object State | Gripper State | Difficulty |
|------------|-------------|---------------|------------|
| ObjectPartiallyAssembledEEGrasped | Nearly inserted | Grasping | Easiest (near-goal) |
| ObjectRestingEEGrasped | Resting on table | Grasping | Medium |
| ObjectAnywhereEEGrasped | Random position | Grasping | Medium-hard |
| ObjectAnywhereEEAnywhere | Random position | Random position | Hardest |

### Registered Tasks
| Task ID | Config | Use |
|---------|--------|-----|
| `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0` | All 4 resets, uniform probs | Training (baseline) |
| `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Adaptive-v0` | All 4 resets, adaptive zone curriculum | Training (adaptive) |
| `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-NearGoal-v0` | 2 near-goal resets | Training (near-goal) |
| `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0` | ObjectAnywhereEEAnywhere only | Evaluation |

### Timing Reference (plan sleeps/timeouts around these)

**Pod Setup (fresh pod, end-to-end ~15 min):**
| Phase | Duration | Bash timeout / sleep |
|-------|----------|---------------------|
| Pod creation API → SSH reachable | 1-2 min | Poll `ssh ... 'echo OK'` every 15s, max 3 min |
| setup.sh (system packages, node, uv, auth, claude) | ~2-3 min | timeout 300s |
| setup_env.sh in tmux (Isaac Sim ~12GB download) | ~8-10 min | `sleep 480`, then check |
| Clone fork + install UWLab/rsl_rl | ~1-2 min | timeout 120s |
| Checkpoint download + patch + smoke test | ~30s | timeout 120s |

**Training & Eval:**
| Operation | Time | Bash timeout / sleep |
|-----------|------|---------------------|
| Isaac Sim init (first launch) | ~3 min | Already included in first iter |
| Isaac Sim init (cached) | ~2 min | Already included in first iter |
| Training first iteration (1x GPU, 4096 envs) | ~3 min after launch | `sleep 180` then check |
| Training per iteration (1x GPU, 4096 envs) | ~15s | — |
| Training per iteration (4x GPU, 16384 envs/GPU) | ~22s | — |
| Training total 40k iters (1x RTX 4090, 4096 envs) | ~4 hours | — |
| Training total 40k iters (4x RTX 4090, 16384 envs/GPU) | ~8-16 hours | Check every 4 hrs |
| Training VRAM per GPU (16384 envs) | ~15 GB / 24 GB | — |
| Eval rollout (200 steps, 1 env) | ~2 min | `sleep 120` |
| Eval total (init + rollout + wandb upload) | ~5 min | `sleep 300` |
| wandb upload (video) | ~10s | — |

**Recommended sleep/timeout patterns:**
- After launching training in tmux: `sleep 180` (3 min) to verify first iteration
- After launching eval in tmux: `sleep 300` (5 min) for full eval
- For setup_env.sh: `sleep 480` (8 min), then check, if still running `sleep 120`
- SSH commands < 30s: use default timeout
- SSH commands with setup.sh: timeout 300s (5 min)
- When checking long-running training: no sleep needed, just SSH in and tail the log

### Eval with Video Recording
- `play.py` has built-in `--video` flag using `gym.wrappers.RecordVideo`
- Videos saved to `logs/<task>/<date>/videos/play/`
- Eval script with wandb upload: `scripts/eval_wandb.sh <checkpoint_path> [run_name]`
- Example eval command:
  ```bash
  python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --num_envs 1 \
    --checkpoint /path/to/checkpoint.pt \
    --video --video_length 200 --headless \
    env.scene.insertive_object=cube env.scene.receptive_object=cube
  ```

### Pretrained Checkpoints
- Download from: `https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/<task>_state_rl_expert.pt`
- Available tasks: fbleg, fbdrawerbottom, peg, rectangle, cube, cupcake
- Checkpoint size: ~6.6MB each

## Training Commands

### Original Paper Baseline (4x GPU — the canonical reference run)
Reproduces the exact original OmniReset paper config. Use this to verify code correctness and as the control for any experiments. 4x GPUs, 16384 envs/GPU, uniform [0.25, 0.25, 0.25, 0.25] reset probs, 40k iterations.
```bash
# In tmux on 4x RTX 4090 pod:
bash scripts/train_omnireset_4gpu_baseline.sh
# wandb run name: baseline_uniform_cube_4gpu
# Task: OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0
# PPO: lr=1e-4, γ=0.99, λ=0.95, clip=0.2, entropy=0.006, 32 steps/env, 5 epochs, 4 mini-batches
```

### Adaptive Zone-Based Curriculum (4x GPU)
Same as baseline but with adaptive reset probability updates (3-zone: stuck/learning/mastered).
```bash
bash scripts/train_omnireset_4gpu_adaptive.sh
# wandb run name: adaptive_zones_cube_4gpu
# Task: OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Adaptive-v0
```

### Single GPU (1x GPU, 4096 envs)
```bash
bash scripts/train_omnireset_single_gpu.sh
# or without --checkpoint to truly start from scratch
```

### From scratch with 2 near-goal resets only
```bash
bash scripts/train_omnireset_near_goal.sh
```

### Finetune from pretrained checkpoint
```bash
# MUST pass both --checkpoint AND --resume
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
  --num_envs 4096 --logger wandb --headless \
  --checkpoint /workspace/checkpoints/cube_state_rl_expert.pt \
  --resume \
  env.scene.insertive_object=cube env.scene.receptive_object=cube
```

### Launching in tmux (CRITICAL: inline commands, don't call set -e scripts)
```bash
tmux new-session -d -s train "bash -c '
source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research/uwlab
source .venv/bin/activate
export ISAACSIM_ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y WANDB_PROJECT=omnireset HDF5_USE_FILE_LOCKING=FALSE
cd UWLab
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-NearGoal-v0 \
  --num_envs 4096 --logger wandb --headless --run_name my_run \
  env.scene.insertive_object=cube env.scene.receptive_object=cube \
  2>&1 | tee /workspace/results/train.log
'"
```

## Two-Repo Structure

This project spans TWO separate git repos:

1. **personal-research** (`rrzhang139/personal-research`) — this repo
   - Contains: launch scripts, configs, CLAUDE.md, experiment tools, `sac_her/` implementation
   - Push to: `git push` from `/workspace/code/personal-research`

2. **UWLab** (`rrzhang139/UWLab`) — forked from `uw-lab/UWLab`
   - Contains: Isaac Lab env definitions, reward functions, observation functions, training entrypoints
   - Cloned into: `uwlab/UWLab/` (gitignored by personal-research)
   - Push to: `cd uwlab/UWLab && git push` (separate repo!)
   - Upstream: `git remote add upstream https://github.com/uw-lab/UWLab.git`
   - Source on GitHub: https://github.com/rrzhang139/UWLab

**CRITICAL: When editing UWLab source files (env configs, rewards, observations), you must `git add/commit/push` from INSIDE `uwlab/UWLab/`, not from personal-research root. These are independent git repos.**

```
/workspace/code/personal-research/          ← repo 1: rrzhang139/personal-research
  uwlab/
    CLAUDE.md                                ← in personal-research
    setup_env.sh                             ← in personal-research
    scripts/                                 ← in personal-research (launch scripts)
    sac_her/                                 ← in personal-research (SAC+HER implementation)
    .venv/                                   ← gitignored (Python venv with Isaac Sim)
    UWLab/                                   ← repo 2: rrzhang139/UWLab (gitignored by repo 1)
      scripts/reinforcement_learning/        ← training entrypoints (train.py, play.py)
      source/uwlab_tasks/                    ← env definitions, rewards, obs, mdp
    exported/                                ← gitignored
    eval_videos/                             ← gitignored
    logs/                                    ← gitignored
```

UWLab is pip-installed in editable mode via `./uwlab.sh --install`, so `import uwlab_tasks` resolves to the source directory. Any edits to UWLab source take effect immediately (no reinstall needed).

## File Locations

**In personal-research (this repo):**
| File | Description |
|------|-------------|
| `uwlab/scripts/train_omnireset_4gpu_baseline.sh` | **Original paper baseline** (4x GPU, uniform probs) |
| `uwlab/scripts/train_omnireset_4gpu_adaptive.sh` | Adaptive zone curriculum (4x GPU) |
| `uwlab/scripts/train_omnireset_single_gpu.sh` | Single GPU training |
| `uwlab/scripts/train_omnireset_near_goal.sh` | Near-goal only training |
| `uwlab/scripts/eval_wandb.sh` | Eval + wandb video upload |
| `uwlab/scripts/patch_adaptive.py` | Legacy adaptive patch (superseded by Adaptive-v0 task) |
| `uwlab/sac_her/` | SAC + HER implementation (Phase 3) |

**In UWLab fork (separate repo, gitignored):**
| File | Description |
|------|-------------|
| `UWLab/scripts/reinforcement_learning/rsl_rl/train.py` | RL training entrypoint |
| `UWLab/scripts/reinforcement_learning/rsl_rl/play.py` | Evaluation/play entrypoint |
| `UWLab/scripts/reinforcement_learning/rsl_rl/cli_args.py` | CLI arg parsing |
| `UWLab/source/uwlab_tasks/.../ur5e_robotiq_2f85/rl_state_cfg.py` | Main config: scene, resets, rewards, obs |
| `UWLab/source/uwlab_tasks/.../ur5e_robotiq_2f85/rl_state_near_goal_cfg.py` | 2-reset near-goal config |
| `UWLab/source/uwlab_tasks/.../ur5e_robotiq_2f85/agents/rsl_rl_cfg.py` | PPO hyperparameters |
| `UWLab/source/uwlab_tasks/.../ur5e_robotiq_2f85/__init__.py` | Task registration (gym.register) |
| `UWLab/source/uwlab_tasks/.../reset_states/mdp/events.py` | Reset sampling (MultiResetManager) |
| `UWLab/source/uwlab_tasks/.../reset_states/mdp/rewards.py` | Reward functions |
| `UWLab/source/uwlab_tasks/.../reset_states/mdp/observations.py` | Observation functions |
