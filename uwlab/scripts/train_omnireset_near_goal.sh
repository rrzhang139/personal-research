#!/bin/bash
# OmniReset Training - Single GPU (RTX 4090)
# Training FROM SCRATCH (no checkpoint) with only 2 near-goal resets:
#   1. ObjectPartiallyAssembledEEGrasped (near-goal, 50%)
#   2. ObjectRestingEEGrasped (resting + grasping, 50%)
# Uses dedicated NearGoal task config (no patching of original config needed)
set -e

source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research/uwlab
source .venv/bin/activate
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export WANDB_PROJECT=omnireset
export HDF5_USE_FILE_LOCKING=FALSE

echo "Starting OmniReset training on single GPU"
echo "Task: cube stacking, 2 near-goal resets only (PartiallyAssembled + Resting), num_envs=4096"
echo "Training from scratch (no checkpoint)"

# Single GPU training from scratch (no --checkpoint, no --resume)
cd /workspace/code/personal-research/uwlab/UWLab

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-NearGoal-v0 \
  --num_envs 4096 \
  --logger wandb \
  --headless \
  --run_name near_goal_2resets \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
