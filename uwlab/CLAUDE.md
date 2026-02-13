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
- GLIBC 2.35+ required (RunPod Ubuntu 22.04 has 2.35)
- System requirements: CUDA 12.4+ driver (RunPod has 12.7), RTX 4090 supported

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
- For 1 GPU: remove `--distributed` and `torch.distributed.run`, reduce `--num_envs`
- Comment out 3 of 4 reset state types, keep only one (e.g. ObjectAnywhereEEAnywhere)

## Directory Structure
```
uwlab/
├── .venv/              # Python 3.11 venv with Isaac Sim + UWLab
├── UWLab/              # Cloned UWLab repo
├── setup_env.sh        # Environment setup script
├── CLAUDE.md           # This file
├── scripts/            # Custom launch scripts
├── partial_assembly_datasets/
├── grasp_datasets/
└── reset_state_datasets/
```

## File Locations
| File | Description |
|------|-------------|
| `UWLab/scripts/reinforcement_learning/rsl_rl/train.py` | RL training script |
| `UWLab/scripts/reinforcement_learning/rsl_rl/play.py` | Evaluation/play script |
| `UWLab/scripts_v2/tools/record_partial_assemblies.py` | Partial assembly data gen |
| `UWLab/scripts_v2/tools/record_grasps.py` | Grasp pose sampling |
| `UWLab/scripts_v2/tools/record_reset_states.py` | Reset state generation |
