#!/bin/bash
# OmniReset Training — 4x RTX 4090, BASELINE Uniform Probabilities
# Standard OmniReset with fixed [0.25, 0.25, 0.25, 0.25] reset sampling
#
# This is the CONTROL run. Compare against train_omnireset_4gpu_adaptive.sh.
# Everything identical except OMNIRESET_ADAPTIVE=0 (default/off).
#
# Estimated time: 4-8 hours for 40k iterations

source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research/uwlab
source .venv/bin/activate
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export WANDB_PROJECT=omnireset
export HDF5_USE_FILE_LOCKING=FALSE

# Ensure adaptive is OFF
export OMNIRESET_ADAPTIVE=0

echo "=========================================="
echo " OmniReset — Baseline (Uniform)"
echo "=========================================="
echo "GPUs: 4x RTX 4090"
echo "Envs/GPU: 16384 | Total: 65536"
echo "Adaptive: OFF (uniform [0.25, 0.25, 0.25, 0.25])"
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
