Launch an OmniReset training run on the pod. Arguments: $ARGUMENTS

Parse the arguments to determine:
- Which task config to use (default: 2 near-goal resets)
- Whether to finetune from checkpoint or train from scratch (default: from scratch)
- Run name for wandb (default: auto-generated)
- Any other overrides

Available task configs:
- `near-goal` or `2reset`: OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-NearGoal-v0 (2 resets: PartiallyAssembled + Resting)
- `full` or `4reset`: OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 (all 4 resets)

Steps:
1. Check if there's already a training run active:
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
tmux ls 2>/dev/null || echo "No tmux sessions"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
ps aux | grep train.py | grep -v grep | head -1
exit
SSHEOF
```

2. If GPU is busy, warn the user and ask if they want to kill the existing run.

3. Launch training in tmux (use inline commands, NOT set -e scripts):
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
tmux new-session -d -s train "bash -c '
source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research/uwlab
source .venv/bin/activate
export ISAACSIM_ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y WANDB_PROJECT=omnireset HDF5_USE_FILE_LOCKING=FALSE
cd UWLab
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task <TASK_ID> \
  --num_envs 4096 --logger wandb --headless \
  --run_name <RUN_NAME> \
  env.scene.insertive_object=cube env.scene.receptive_object=cube \
  2>&1 | tee /workspace/results/train_<RUN_NAME>.log
'"
echo "Training launched"
tmux ls
exit
SSHEOF
```

4. Wait ~2 minutes for Isaac Sim to initialize, then verify training started:
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
tail -20 /workspace/results/train_<RUN_NAME>.log
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
exit
SSHEOF
```

5. Find and report the wandb URL from the log.

IMPORTANT:
- If finetuning, add BOTH `--checkpoint <path> --resume` flags
- Without `--resume`, checkpoint is ignored and training starts from scratch
- wandb logs to project `isaaclab` by default (rsl_rl hardcoded), not `omnireset`
- SSH address is in root CLAUDE.md under `SSH Address`
- Do NOT use `set -e` in training commands sourcing `.bashrc_pod`
