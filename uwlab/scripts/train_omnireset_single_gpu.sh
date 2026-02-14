#!/bin/bash
# OmniReset Training - Single GPU (RTX 4090)
# Uses cloud-hosted reset state datasets (no local data gen needed)
# Only uses near-goal reset (ObjectPartiallyAssembledEEGrasped) for sparse reward
# Resumes from pretrained checkpoint
set -e

source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research/uwlab
source .venv/bin/activate
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export WANDB_PROJECT=omnireset
export HDF5_USE_FILE_LOCKING=FALSE

echo "Starting OmniReset training on single GPU"
echo "Task: cube stacking, near-goal reset only, num_envs=4096"
echo "Resuming from pretrained checkpoint"

# Single GPU training (no torch.distributed.run, no --distributed)
# Reduced num_envs from 16384 (4 GPU) to 4096 (1 GPU)
cd /workspace/code/personal-research/uwlab/UWLab

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
  --num_envs 4096 \
  --logger wandb \
  --headless \
  --checkpoint /workspace/checkpoints/cube_state_rl_expert.pt \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
