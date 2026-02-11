#!/bin/bash
# Setup and train diffusion base policy for residual-rl
# Run this on RunPod inside tmux: tmux new -s train && bash scripts/setup_and_train.sh
set -e
source /workspace/.bashrc_pod 2>/dev/null

PROJ_DIR="/workspace/code/personal-research/residual-rl"
LOG_DIR="/workspace/results/residual-rl"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/setup.log"

cd "$PROJ_DIR"

echo "=== $(date) === GPU Check ===" | tee "$LOG"
nvidia-smi 2>&1 | tee -a "$LOG" || echo "nvidia-smi not available" | tee -a "$LOG"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); d=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'; print('GPU:', d)" 2>&1 | tee -a "$LOG"

echo "=== $(date) === Package Check ===" | tee -a "$LOG"
python -c "
import mani_skill; print('mani_skill:', mani_skill.__version__)
import diffusers; print('diffusers:', diffusers.__version__)
import gymnasium; print('gymnasium:', gymnasium.__version__)
import einops; print('einops:', einops.__version__)
import torch; print('torch:', torch.__version__)
print('ALL OK')
" 2>&1 | tee -a "$LOG"

echo "=== $(date) === Downloading demos ===" | tee -a "$LOG"
# Download ManiSkill3 demos for PegInsertionSide-v1
python -m mani_skill.utils.download_demo PegInsertionSide-v1 2>&1 | tee -a "$LOG" || true

# Find where demos were downloaded
echo "Searching for demo files..." | tee -a "$LOG"
DEMO_DIR=$(python -c "
import mani_skill.utils.download_demo as dd
import os
# MS3 stores demos in ~/.maniskill/demos/<env_id>/
home = os.path.expanduser('~')
ms_dir = os.path.join(home, '.maniskill', 'demos', 'PegInsertionSide-v1')
if os.path.exists(ms_dir):
    print(ms_dir)
else:
    # Check XDG or other paths
    for root, dirs, files in os.walk(os.path.join(home, '.maniskill')):
        for f in files:
            if f.endswith('.h5'):
                print(root)
                break
        break
" 2>/dev/null)

if [ -z "$DEMO_DIR" ]; then
    # Fallback: search for the h5 file
    DEMO_DIR=$(find /root/.maniskill -name "*.h5" -path "*PegInsertion*" -exec dirname {} \; 2>/dev/null | head -1)
fi

if [ -z "$DEMO_DIR" ]; then
    DEMO_DIR=$(find /workspace -name "*.h5" -path "*PegInsertion*" -exec dirname {} \; 2>/dev/null | head -1)
fi

echo "Demo dir found: $DEMO_DIR" | tee -a "$LOG"

# Set up data symlinks
mkdir -p data/PegInsertionSide
if [ -n "$DEMO_DIR" ]; then
    DEMO_H5=$(find "$DEMO_DIR" -name "*.h5" | head -1)
    DEMO_JSON=$(find "$DEMO_DIR" -name "*.json" | head -1)
    if [ -n "$DEMO_H5" ]; then
        ln -sf "$DEMO_H5" data/PegInsertionSide/trajectory.h5
        echo "Linked: $DEMO_H5 -> data/PegInsertionSide/trajectory.h5" | tee -a "$LOG"
    fi
    if [ -n "$DEMO_JSON" ]; then
        ln -sf "$DEMO_JSON" data/PegInsertionSide/trajectory.json
        echo "Linked: $DEMO_JSON -> data/PegInsertionSide/trajectory.json" | tee -a "$LOG"
    fi
fi

ls -la data/PegInsertionSide/ 2>&1 | tee -a "$LOG"

echo "=== $(date) === Starting training ===" | tee -a "$LOG"

# Train diffusion policy on PegInsertionSide-v1 (state-based)
# Using shorter training (100k iters) as initial test, increase if needed
python offline/diffusion_policy_unet_maniskill2.py \
    --env-id PegInsertionSide-v1 \
    --demo-path data/PegInsertionSide/trajectory.h5 \
    --control-mode pd_ee_delta_pose \
    --total-iters 100000 \
    --eval-freq 10000 \
    --save-freq 50000 \
    --batch-size 1024 \
    --lr 1e-4 \
    --output-dir /workspace/results/residual-rl/output \
    --track \
    --wandb-project-name policy_decorator \
    2>&1 | tee -a "$LOG"

echo "=== $(date) === Training complete ===" | tee -a "$LOG"
