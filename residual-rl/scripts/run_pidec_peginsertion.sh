#!/bin/bash
source /workspace/.bashrc_pod 2>/dev/null
source /workspace/code/personal-research/residual-rl/.venv/bin/activate
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
cd /workspace/code/personal-research/residual-rl
export PYTHONPATH=/workspace/code/personal-research/residual-rl:$PYTHONPATH

echo "Starting Policy Decorator (Residual RL) on PegInsertionSide-v1"
echo "Base policy: /workspace/checkpoints/best_adapted_ms3.pt"

python -u online/pi_dec_diffusion_maniskill2.py \
  --env-id PegInsertionSide-v1 \
  --base-policy-ckpt /workspace/checkpoints/best_adapted_ms3.pt \
  --total-timesteps 4000000 \
  --res-scale 0.1 \
  --prog-explore 100000 \
  --eval-freq 50000 \
  --num-envs 16 \
  --capture-video \
  --track \
  --exp-name pidec_ms3_adapted
