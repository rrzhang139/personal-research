#!/bin/bash
# ==========================================================================
# OmniReset — ADAPTIVE ZONE-BASED CURRICULUM (4x GPU)
# ==========================================================================
# Same as baseline but with 3-zone adaptive reset probability updates:
#
#   Zone       | Success Rate        | Action
#   -----------|---------------------|---------------------------------------
#   Stuck      | < 1% (zero_thresh)  | Slowly decay prob (×0.999) to floor
#   Learning   | 1-80%               | Boost prob (×1.005) — learning frontier
#   Mastered   | ≥ 80%               | Decrease prob (×0.995) — shift focus
#
# Expected trajectory:
#   Early:  Task 3 (near-goal) mastered → prob 0.25→0.10
#           Task 0 (hardest) stuck at 0% → prob slowly decays 0.25→0.15
#   Middle: Tasks 1,2 (medium) at 5-30% → prob grows 0.25→0.40
#   Late:   Tasks 1,2 mastered → prob drops, Task 0 enters learning → prob grows
#
# All params overridable via CLI, e.g.:
#   env.events.reset_from_reset_states.params.adaptive_mastered_thresh=0.9
#
# Task: OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Adaptive-v0
# Compare against: train_omnireset_4gpu_baseline.sh
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
echo " OmniReset — Adaptive Zone-Based Curriculum"
echo "=========================================="
echo "GPUs: 4x RTX 4090"
echo "Envs/GPU: 16384 | Total: 65536"
echo "Reset probs: starts [0.25, 0.25, 0.25, 0.25], adapts via 3-zone policy"
echo "  zero_thresh=0.01, mastered_thresh=0.80"
echo "  stuck_decay=0.999, learning_boost=1.005, mastered_decay=0.995"
echo "  min_prob=0.05"
echo "Task: cube insertion, all 4 resets"
echo "Max iterations: 40000"
echo "Estimated time: 4-8 hours"
echo "=========================================="

cd /workspace/code/personal-research/uwlab/UWLab

python -m torch.distributed.run --nproc_per_node 4 \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Adaptive-v0 \
  --num_envs 16384 \
  --distributed \
  --logger wandb \
  --headless \
  --run_name adaptive_zones_cube_4gpu \
  --log_project_name omnireset \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
