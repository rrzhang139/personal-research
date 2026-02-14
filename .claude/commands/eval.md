Run evaluation on a checkpoint and upload video to wandb. Arguments: $ARGUMENTS

Parse arguments for:
- Checkpoint path (required — can be a model_*.pt path or "pretrained" for the pretrained expert)
- Run name (optional, defaults to checkpoint name)

Checkpoint shortcuts:
- `pretrained` or `expert`: /workspace/checkpoints/cube_state_rl_expert.pt
- `latest`: find the latest model_*.pt in logs/rsl_rl/ur5e_robotiq_2f85_reset_states_agent/
- A full path: use as-is

Steps:
1. Resolve the checkpoint path and check GPU availability:
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
ps aux | grep -E "train.py|play.py" | grep -v grep | head -1
# If using "latest", find it:
ls -t /workspace/code/personal-research/uwlab/UWLab/logs/rsl_rl/ur5e_robotiq_2f85_reset_states_agent/*/model_*.pt 2>/dev/null | head -1
exit
SSHEOF
```

2. If GPU is busy with training, warn user — eval needs the GPU too.

3. Launch eval in tmux:
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
tmux new-session -d -s eval "bash -c '
source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research/uwlab
source .venv/bin/activate
export ISAACSIM_ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y WANDB_PROJECT=omnireset HDF5_USE_FILE_LOCKING=FALSE
bash scripts/eval_wandb.sh <CHECKPOINT_PATH> <RUN_NAME> \
  2>&1 | tee /workspace/results/eval_<RUN_NAME>.log
'"
echo "Eval launched"
tmux ls
exit
SSHEOF
```

4. Wait ~3 minutes for Isaac Sim init + rollout, then check results:
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
tail -20 /workspace/results/eval_<RUN_NAME>.log
exit
SSHEOF
```

5. Report the wandb URL and eval results.

IMPORTANT:
- Eval takes ~3-5 minutes total (2 min Isaac Sim init + 1 min rollout + upload)
- GPU must be free (can't run eval while training)
- SSH address is in root CLAUDE.md under `SSH Address`
