Pod has been restarted (stopped then started again). The persistent volume `/workspace/` survives but container packages (tmux, libglu, etc.) are wiped.

SSH address: check providers/runpod.md for `Current Instance` section.

## Timing: ~2 min total, 1 SSH call

| Phase | Duration | Timeout |
|-------|----------|---------|
| restart.sh (apt-get, symlinks, claude) | ~30s | 120s |
| git pull | ~5s | — |
| Verify (python, nvidia-smi) | ~5s | — |

## Do everything in ONE SSH call (timeout 120s):
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
bash /workspace/code/personal-research/runpod/restart.sh

# Pull latest code
cd /workspace/code/personal-research && git pull
cd uwlab/UWLab && git pull origin main 2>/dev/null || true
cd /workspace/code/personal-research

# Verify
source /workspace/.bashrc_pod 2>/dev/null
python3 -c "import os; print(f'WANDB_API_KEY: {os.environ.get(\"WANDB_API_KEY\", \"NOT SET\")[:8]}...')"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
tmux ls 2>/dev/null || echo "tmux installed OK (no sessions)"
echo "=== POD RESTART COMPLETE ==="
exit
SSHEOF
```

Report GPU status and any issues to the user.

## IMPORTANT
- `.bashrc_pod` uses `set -a` to export `.env` vars. If missing, wandb/HF auth fails in Python.
- tmux and libglu1-mesa are reinstalled by restart.sh (container disk wiped on stop).
- `/workspace/venv/` may not exist (legacy path) — `.bashrc_pod` may warn, that's OK.
