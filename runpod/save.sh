#!/bin/bash
# save.sh — Run before TERMINATING the pod (not needed for stop)
# If you're just stopping, /workspace persists automatically
set -e

echo "=== Pre-Terminate Backup ==="

# ---- 1. Git push all repos ----
echo "--- Pushing code to GitHub ---"
for dir in /workspace/code/*/; do
    if [ -d "$dir/.git" ]; then
        echo "  Pushing $(basename "$dir")"
        cd "$dir"
        git add -A
        git diff --cached --quiet || git commit -m "backup $(date +%Y%m%d-%H%M)"
        git push || echo "  [WARN] Push failed for $(basename "$dir")"
    fi
done

# ---- 2. Sync wandb offline runs ----
if command -v wandb &> /dev/null; then
    echo "--- Syncing wandb runs ---"
    for run in /workspace/wandb/offline-*; do
        [ -d "$run" ] && wandb sync "$run" 2>/dev/null || true
    done
fi

echo ""
echo "=== Done. Safe to terminate pod. ==="
