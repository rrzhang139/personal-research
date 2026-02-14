Check the status of the RunPod instance — GPU, running jobs, training progress, wandb.

SSH into the pod and check everything:
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null

echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,gpu-util --format=csv

echo "=== TMUX SESSIONS ==="
tmux ls 2>/dev/null || echo "No tmux sessions"

echo "=== RUNNING PROCESSES ==="
ps aux | grep -E "train.py|play.py|python.*rsl_rl" | grep -v grep

echo "=== LATEST TRAINING LOG ==="
for f in /workspace/results/train*.log; do
  if [ -f "$f" ]; then
    echo "--- $f (last 10 lines) ---"
    tail -10 "$f"
  fi
done

echo "=== DISK USAGE ==="
df -h /workspace | tail -1

echo "=== WANDB URL ==="
for f in /workspace/results/train*.log; do
  grep -o "https://wandb.ai/[^ ]*" "$f" 2>/dev/null | tail -1
done

exit
SSHEOF
```

Report a concise summary: GPU usage, active jobs, training progress (iteration/ETA), wandb URLs, disk space.

SSH address is in root CLAUDE.md under `SSH Address`.
