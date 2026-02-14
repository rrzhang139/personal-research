Pod has been restarted (stopped then started again). The persistent volume `/workspace/` survives but container packages (tmux, libglu, etc.) are wiped.

SSH address: check CLAUDE.md for `SSH Address` field (under root CLAUDE.md), it may have changed if this is a new pod.

Steps to recover:
1. SSH into the pod and run `restart.sh`:
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
bash /workspace/code/personal-research/runpod/restart.sh
exit
SSHEOF
```

2. Verify the environment works:
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
python3 -c "import os; print(f'WANDB_API_KEY: {os.environ.get(\"WANDB_API_KEY\", \"NOT SET\")[:8]}...')"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
tmux ls 2>/dev/null || echo "tmux installed OK (no sessions)"
exit
SSHEOF
```

3. Pull latest code:
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research && git pull
cd uwlab/UWLab && git pull origin main
exit
SSHEOF
```

4. Report what you found to the user (GPU status, env vars, any issues).

IMPORTANT reminders:
- `.bashrc_pod` uses `set -a` to export `.env` vars. If this is missing, wandb/HF auth will fail in Python.
- `/workspace/venv/` may not exist (legacy) — don't use `set -e` when sourcing `.bashrc_pod`
- tmux and libglu1-mesa need reinstalling after every restart (restart.sh handles this)
