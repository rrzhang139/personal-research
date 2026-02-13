#!/bin/bash
# OmniReset Training - Single GPU (RTX 4090)
# Uses cloud-hosted reset state datasets (no local data gen needed)
# Only uses 1 reset type (ObjectAnywhereEEAnywhere) instead of all 4
set -e

source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research/uwlab
source .venv/bin/activate
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export WANDB_PROJECT=omnireset

echo "Starting OmniReset training on single GPU"
echo "Task: cube stacking, 1 reset type, num_envs=4096"

# Single GPU training (no torch.distributed.run, no --distributed)
# Reduced num_envs from 16384 (4 GPU) to 4096 (1 GPU)
cd /workspace/code/personal-research/uwlab/UWLab

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
  --num_envs 4096 \
  --logger wandb \
  --headless \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
