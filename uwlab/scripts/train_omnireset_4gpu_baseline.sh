#!/bin/bash
# ==========================================================================
# OmniReset — ORIGINAL PAPER BASELINE (4x GPU, Uniform Resets)
# ==========================================================================
# This reproduces the EXACT original paper training configuration:
#   - 4x RTX 4090 GPUs with torch.distributed
#   - 16384 envs per GPU (65536 total)
#   - Uniform reset probabilities [0.25, 0.25, 0.25, 0.25]
#   - All 4 reset types (hardest → easiest):
#       Task 0: ObjectAnywhereEEAnywhere       (hardest)
#       Task 1: ObjectRestingEEGrasped          (medium)
#       Task 2: ObjectAnywhereEEGrasped         (medium-hard)
#       Task 3: ObjectPartiallyAssembledEEGrasped (easiest, near-goal)
#   - PPO: lr=1e-4, γ=0.99, λ=0.95, clip=0.2, entropy=0.006
#   - 40k iterations, 32 steps/env, 5 epochs, 4 mini-batches
#   - Object pair: cube insertion (default)
#
# Use this as the CONTROL run to compare against adaptive curriculum.
# Task: OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0
#
# Estimated time: 4-8 hours for 40k iterations on 4x RTX 4090
# ==========================================================================

source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research/uwlab
source .venv/bin/activate
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export WANDB_PROJECT=omnireset
export HDF5_USE_FILE_LOCKING=FALSE

echo "=========================================="
echo " OmniReset — Baseline (Uniform)"
echo "=========================================="
echo "GPUs: 4x RTX 4090"
echo "Envs/GPU: 16384 | Total: 65536"
echo "Reset probs: [0.25, 0.25, 0.25, 0.25] (fixed)"
echo "Task: cube insertion, all 4 resets"
echo "Max iterations: 40000"
echo "Estimated time: 4-8 hours"
echo "=========================================="

cd /workspace/code/personal-research/uwlab/UWLab

python -m torch.distributed.run --nproc_per_node 4 \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
  --num_envs 16384 \
  --distributed \
  --logger wandb \
  --headless \
  --run_name baseline_uniform_cube_4gpu \
  --log_project_name omnireset \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
