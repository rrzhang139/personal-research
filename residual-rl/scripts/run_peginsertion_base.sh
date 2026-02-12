#!/bin/bash
source /workspace/.bashrc_pod 2>/dev/null
source /workspace/code/personal-research/residual-rl/.venv/bin/activate
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
cd /workspace/code/personal-research/residual-rl
export PYTHONPATH=/workspace/code/personal-research/residual-rl:$PYTHONPATH

echo "Starting PegInsertionSide-v1 base policy: 1M iters, batch_size=1024"

python -u offline/diffusion_policy_unet_maniskill2.py \
  --env-id PegInsertionSide-v1 \
  --demo-path /workspace/datasets/maniskill_demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \
  --total-iters 1000000 \
  --batch-size 1024 \
  --eval-freq 100000 \
  --capture-video \
  --track \
  --exp-name base_bs1024
