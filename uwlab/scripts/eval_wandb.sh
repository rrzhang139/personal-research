#!/bin/bash
# OmniReset Eval with wandb video logging
# Runs play.py, then uploads video + metadata to wandb
#
# Usage:
#   bash scripts/eval_wandb.sh <checkpoint_path> [run_name]
#
# Examples:
#   bash scripts/eval_wandb.sh /workspace/checkpoints/cube_state_rl_expert.pt pretrained
#   bash scripts/eval_wandb.sh logs/rsl_rl/.../model_400.pt 4reset_iter400
CHECKPOINT="${1:?Usage: eval_wandb.sh <checkpoint_path> [run_name]}"
RUN_NAME="${2:-eval}"

# NOTE: Do NOT use set -e here. .bashrc_pod sources a legacy venv that may not exist.
source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research/uwlab
source .venv/bin/activate
set -e  # Safe to enable after sourcing
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export WANDB_PROJECT=omnireset
export HDF5_USE_FILE_LOCKING=FALSE

# Create unique output dir for this eval
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
VIDEO_DIR="/workspace/code/personal-research/uwlab/eval_videos/${TIMESTAMP}_${RUN_NAME}"
mkdir -p "$VIDEO_DIR"

echo "=== OmniReset Eval ==="
echo "Checkpoint: $CHECKPOINT"
echo "Run name: $RUN_NAME"
echo "Video dir: $VIDEO_DIR"

cd /workspace/code/personal-research/uwlab/UWLab

# Run play.py to generate video
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
  --num_envs 1 \
  --checkpoint "$CHECKPOINT" \
  --video --video_length 200 --headless \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube

# Find the generated video (play.py saves to logs/<task>/<date>/videos/play/)
LATEST_LOG=$(ls -td logs/rsl_rl/ur5e_robotiq_2f85_reset_states_agent/*/videos/play/ 2>/dev/null | head -1)
if [ -z "$LATEST_LOG" ]; then
  # Fallback: check the checkpoint dir
  CKPT_DIR=$(dirname "$CHECKPOINT")
  LATEST_LOG="$CKPT_DIR/videos/play/"
fi

echo "Looking for video in: $LATEST_LOG"
VIDEO_FILE=$(find "$LATEST_LOG" -name "*.mp4" -type f 2>/dev/null | sort | tail -1)

if [ -z "$VIDEO_FILE" ]; then
  echo "ERROR: No video file found!"
  exit 1
fi

echo "Found video: $VIDEO_FILE"

# Copy video to our eval dir
cp "$VIDEO_FILE" "$VIDEO_DIR/eval.mp4"

# Upload to wandb
cd /workspace/code/personal-research/uwlab
python scripts/upload_eval_wandb.py \
  --video "$VIDEO_DIR/eval.mp4" \
  --checkpoint "$CHECKPOINT" \
  --run_name "$RUN_NAME"

echo "=== Eval complete ==="
echo "Video: $VIDEO_DIR/eval.mp4"
echo "Logged to wandb project: omnireset"
