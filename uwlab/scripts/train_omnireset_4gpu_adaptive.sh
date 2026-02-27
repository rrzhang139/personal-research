#!/bin/bash
# OmniReset Training — 4x RTX 4090, ADAPTIVE Reset Probabilities
# Experiment 1.1: Shift sampling toward harder (less successful) tasks
#
# Prerequisites:
#   1. Pod restarted: bash /workspace/code/personal-research/runpod/restart.sh
#   2. Patch applied: python scripts/patch_adaptive.py
#
# Compare against baseline uniform [0.25, 0.25, 0.25, 0.25] run.
# Only difference: OMNIRESET_ADAPTIVE=1 (everything else identical)
#
# Expected: 1.5-3x speedup to 90% success rate
# Estimated time: 4-8 hours for 40k iterations

source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research/uwlab
source .venv/bin/activate
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export WANDB_PROJECT=omnireset
export HDF5_USE_FILE_LOCKING=FALSE

# --- Adaptive sampling config ---
export OMNIRESET_ADAPTIVE=1
export OMNIRESET_TEMPERATURE=0.5
export OMNIRESET_MIN_PROB=0.05

echo "=========================================="
echo " OmniReset — Adaptive Reset Experiment"
echo "=========================================="
echo "GPUs: 4x RTX 4090"
echo "Envs/GPU: 16384 | Total: 65536"
echo "Adaptive: ON (temp=${OMNIRESET_TEMPERATURE}, min_prob=${OMNIRESET_MIN_PROB})"
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
  --run_name adaptive_t05_cube_4gpu \
  --log_project_name omnireset \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
