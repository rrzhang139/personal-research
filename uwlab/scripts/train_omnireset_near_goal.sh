#!/bin/bash
# OmniReset Training - Single GPU (RTX 4090)
# Training FROM SCRATCH (no checkpoint) with only 2 near-goal resets:
#   1. ObjectPartiallyAssembledEEGrasped (near-goal, 50%)
#   2. ObjectRestingEEGrasped (resting + grasping, 50%)
# Disables ObjectAnywhereEEAnywhere and ObjectAnywhereEEGrasped
set -e

source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research/uwlab
source .venv/bin/activate
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export WANDB_PROJECT=omnireset
export HDF5_USE_FILE_LOCKING=FALSE

echo "Starting OmniReset training on single GPU"
echo "Task: cube stacking, 2 near-goal resets only, num_envs=4096"
echo "Training from scratch (no checkpoint)"

# Patch rl_state_cfg.py to use only 2 resets before training
CONFIG_FILE="/workspace/code/personal-research/uwlab/UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/reset_states/config/ur5e_robotiq_2f85/rl_state_cfg.py"

# Backup current config
cp "$CONFIG_FILE" "${CONFIG_FILE}.before_near_goal"

# Replace the TrainEventCfg reset paths and probs
# Uses python to do a precise replacement
python3 - "$CONFIG_FILE" << 'PYEOF'
import sys
config_file = sys.argv[1]

with open(config_file, 'r') as f:
    content = f.read()

# Find and replace the TrainEventCfg reset_from_reset_states block
old_block = '''    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "base_paths": [
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectAnywhereEEAnywhere",
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectRestingEEGrasped",
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectAnywhereEEGrasped",
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )'''

new_block = '''    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "base_paths": [
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectPartiallyAssembledEEGrasped",
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectRestingEEGrasped",
            ],
            "probs": [0.5, 0.5],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )'''

if old_block not in content:
    print("ERROR: Could not find the expected TrainEventCfg block to replace!")
    print("The config file may have been modified. Check manually.")
    sys.exit(1)

content = content.replace(old_block, new_block)

with open(config_file, 'w') as f:
    f.write(content)

print("Successfully patched rl_state_cfg.py: using 2 near-goal resets (50/50)")
PYEOF

# Single GPU training from scratch (no --checkpoint, no --resume)
cd /workspace/code/personal-research/uwlab/UWLab

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
  --num_envs 4096 \
  --logger wandb \
  --headless \
  --run_name near_goal_2resets \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
