#!/bin/bash
# Setup and train diffusion base policy for residual-rl
# Run directly on the pod: bash scripts/setup_and_train.sh
# Or in tmux: tmux new -s train 'bash /workspace/code/personal-research/residual-rl/scripts/setup_and_train.sh'
set -eo pipefail

PROJ_DIR="/workspace/code/personal-research/residual-rl"
LOG_DIR="/workspace/results/residual-rl"
DEMO_CACHE="/workspace/datasets/maniskill_demos"
mkdir -p "$LOG_DIR" "$DEMO_CACHE"
LOG="$LOG_DIR/train.log"

cd "$PROJ_DIR"

# Activate project venv
source .venv/bin/activate
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"
export HDF5_USE_FILE_LOCKING=FALSE  # RunPod NFS volume doesn't support h5py file locking

echo "=== $(date) === GPU Check ===" | tee "$LOG"
nvidia-smi 2>&1 | tee -a "$LOG" || echo "nvidia-smi not available" | tee -a "$LOG"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1 | tee -a "$LOG"

echo "=== $(date) === Package Check ===" | tee -a "$LOG"
python -c "
import mani_skill; print('mani_skill:', mani_skill.__version__)
import diffusers; print('diffusers:', diffusers.__version__)
import gymnasium; print('gymnasium:', gymnasium.__version__)
import einops; print('einops:', einops.__version__)
import torch; print('torch:', torch.__version__)
print('ALL OK')
" 2>&1 | tee -a "$LOG"

echo "=== $(date) === Setting up demos ===" | tee -a "$LOG"

ENV_ID="PegInsertionSide-v1"
CONTROL_MODE="pd_ee_delta_pose"
DEMO_TYPE="motionplanning"
RAW_H5="$DEMO_CACHE/$ENV_ID/$DEMO_TYPE/trajectory.h5"
# Converted demo with state obs and target control mode
CONVERTED_H5="$DEMO_CACHE/$ENV_ID/$DEMO_TYPE/trajectory.state.${CONTROL_MODE}.physx_cpu.h5"

# Step 1: Download raw demos if needed
if [ ! -f "$RAW_H5" ]; then
    echo "Downloading demos for $ENV_ID..." | tee -a "$LOG"
    python -m mani_skill.utils.download_demo "$ENV_ID" 2>&1 | tee -a "$LOG" || true

    # Copy from container disk (/root/) to persistent volume (/workspace/)
    SRC="/root/.maniskill/demos/$ENV_ID"
    if [ -d "$SRC" ]; then
        cp -r "$SRC" "$DEMO_CACHE/"
        echo "Copied demos to $DEMO_CACHE/$ENV_ID" | tee -a "$LOG"
    else
        echo "ERROR: Demo download failed — $SRC not found" | tee -a "$LOG"
        exit 1
    fi
fi

# Step 2: Replay/convert demos to state obs + target control mode if needed
if [ ! -f "$CONVERTED_H5" ]; then
    echo "Converting demos to obs_mode=state, control_mode=$CONTROL_MODE..." | tee -a "$LOG"
    python -m mani_skill.trajectory.replay_trajectory \
        --traj-path "$RAW_H5" \
        -o state \
        -c "$CONTROL_MODE" \
        --save-traj \
        2>&1 | tee -a "$LOG"
    echo "Conversion complete" | tee -a "$LOG"
fi

echo "Using demo: $CONVERTED_H5" | tee -a "$LOG"
ls -la "$CONVERTED_H5" 2>&1 | tee -a "$LOG"

echo "=== $(date) === Starting training ===" | tee -a "$LOG"

# Train diffusion policy on PegInsertionSide-v1 (state-based)
python offline/diffusion_policy_unet_maniskill2.py \
    --env-id "$ENV_ID" \
    --demo-path "$CONVERTED_H5" \
    --control-mode "$CONTROL_MODE" \
    --total-iters 100000 \
    --eval-freq 10000 \
    --save-freq 50000 \
    --batch-size 1024 \
    --lr 1e-4 \
    --output-dir "$LOG_DIR/output" \
    --track \
    --wandb-project-name policy_decorator \
    2>&1 | tee -a "$LOG"

echo "=== $(date) === Training complete ===" | tee -a "$LOG"
