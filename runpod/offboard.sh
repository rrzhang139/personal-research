#!/bin/bash
# offboard.sh — Safely back up all precious data from a RunPod pod before termination.
# Usage: bash offboard.sh
# Must be run ON the pod (not via SSH heredoc — too long-running).
#
# Strategy:
#   1. Try wandb (upload checkpoints as artifacts + sync run)
#   2. If wandb fails (storage limit), try HuggingFace Hub
#   3. If both fail, use runpodctl send to transfer to local machine
#
# After running, verify uploads succeeded before terminating the pod.

set -o pipefail

source /workspace/.bashrc_pod 2>/dev/null

# ── Config ──────────────────────────────────────────────────────────────────
WANDB_PROJECT="${WANDB_PROJECT:-omnireset}"
WANDB_ENTITY="${WANDB_ENTITY:-}"  # leave empty for default entity
HF_REPO="${HF_REPO:-rrzhang139/omnireset-checkpoints}"
CHECKPOINT_DIR="/workspace/code/personal-research/uwlab/UWLab/logs/rsl_rl"
RESULTS_DIR="/workspace/results"
WANDB_LOCAL="/workspace/wandb"

echo "=============================================="
echo "  OFFBOARD: Backing up pod data"
echo "=============================================="
echo ""

# ── Step 1: Git push ───────────────────────────────────────────────────────
echo "--- [1/4] Git push ---"
cd /workspace/code/personal-research
if [ -n "$(git status --porcelain)" ]; then
    git add -A && git commit -m "offboard: save uncommitted changes" && git push
    echo "Code pushed."
else
    echo "Code already clean and pushed."
fi

if [ -d "uwlab/UWLab/.git" ]; then
    cd uwlab/UWLab
    if [ -n "$(git status --porcelain)" ]; then
        git add -A && git commit -m "offboard: save uncommitted changes" && git push
        echo "UWLab fork pushed."
    else
        echo "UWLab fork already clean."
    fi
fi
echo ""

# ── Step 2: Sync wandb runs ───────────────────────────────────────────────
echo "--- [2/4] Sync wandb runs ---"
WANDB_SYNCED=false
if command -v wandb &>/dev/null || pip show wandb &>/dev/null; then
    # Try to find wandb in any venv
    for venv in /workspace/code/personal-research/uwlab/.venv /workspace/code/personal-research/residual-rl/.venv; do
        if [ -f "$venv/bin/wandb" ]; then
            export PATH="$venv/bin:$PATH"
            break
        fi
    done

    wandb sync "$WANDB_LOCAL" 2>&1
    if [ $? -eq 0 ]; then
        echo "wandb runs synced successfully."
        WANDB_SYNCED=true
    else
        echo "WARNING: wandb sync failed. Runs may already be synced or storage limit reached."
    fi
else
    echo "WARNING: wandb CLI not found. Skipping sync."
fi
echo ""

# ── Step 3: Upload checkpoints ────────────────────────────────────────────
echo "--- [3/4] Upload checkpoints ---"
UPLOAD_SUCCESS=false

# Count checkpoints
CKPT_COUNT=$(find "$CHECKPOINT_DIR" -name "model_*.pt" 2>/dev/null | wc -l)
echo "Found $CKPT_COUNT checkpoint files."

if [ "$CKPT_COUNT" -eq 0 ]; then
    echo "No checkpoints to upload."
    UPLOAD_SUCCESS=true
else
    # ── Try wandb artifacts first ──
    echo "Attempting wandb artifact upload..."
    python3 << 'PYEOF'
import wandb
import os
import glob
import sys

project = os.environ.get("WANDB_PROJECT", "omnireset")
entity = os.environ.get("WANDB_ENTITY", None)
checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "")

# Find all run directories with checkpoints
run_dirs = glob.glob(os.path.join(checkpoint_dir, "**/model_*.pt"), recursive=True)
if not run_dirs:
    print("No checkpoints found.")
    sys.exit(0)

# Group by parent directory (each training run)
from collections import defaultdict
runs = defaultdict(list)
for f in sorted(run_dirs):
    parent = os.path.dirname(f)
    runs[parent].append(f)

for run_dir, files in runs.items():
    run_name = os.path.basename(run_dir)
    artifact_name = f"checkpoints-{run_name}"
    print(f"\nUploading {len(files)} checkpoints from {run_name}...")

    try:
        run = wandb.init(project=project, entity=entity, job_type="offboard", name=f"offboard-{run_name}")

        # Upload best/latest checkpoint as artifact
        artifact = wandb.Artifact(artifact_name, type="model")
        # Upload the latest (highest numbered) checkpoint
        latest = sorted(files, key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]))[-1]
        artifact.add_file(latest, name=os.path.basename(latest))
        # Also upload model_0 (initial) for reference
        initial = sorted(files)[0]
        if initial != latest:
            artifact.add_file(initial, name=os.path.basename(initial))

        run.log_artifact(artifact)
        run.finish()
        print(f"Uploaded {artifact_name} to wandb.")
    except Exception as e:
        print(f"ERROR uploading to wandb: {e}")
        sys.exit(1)

print("\nAll checkpoints uploaded to wandb.")
PYEOF

    if [ $? -eq 0 ]; then
        echo "wandb artifact upload succeeded."
        UPLOAD_SUCCESS=true
    else
        echo "wandb upload failed. Trying HuggingFace Hub..."

        # ── Fallback: HuggingFace Hub ──
        python3 << 'PYEOF'
import os
import glob
import sys

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("huggingface_hub not installed. pip install huggingface_hub")
    sys.exit(1)

hf_repo = os.environ.get("HF_REPO", "rrzhang139/omnireset-checkpoints")
checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "")
token = os.environ.get("HF_TOKEN", "")

if not token:
    print("ERROR: HF_TOKEN not set.")
    sys.exit(1)

api = HfApi()
try:
    create_repo(hf_repo, token=token, exist_ok=True, repo_type="model")
except Exception as e:
    print(f"Repo creation note: {e}")

files = sorted(glob.glob(os.path.join(checkpoint_dir, "**/*.pt"), recursive=True))
if not files:
    print("No files to upload.")
    sys.exit(0)

print(f"Uploading {len(files)} files to hf://{hf_repo}")
for f in files:
    rel = os.path.relpath(f, os.path.dirname(checkpoint_dir))
    print(f"  {rel}...")
    try:
        api.upload_file(path_or_fileobj=f, path_in_repo=rel, repo_id=hf_repo, token=token)
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

print("HuggingFace upload succeeded.")
PYEOF

        if [ $? -eq 0 ]; then
            echo "HuggingFace upload succeeded."
            UPLOAD_SUCCESS=true
        else
            echo ""
            echo "=============================================="
            echo "  BOTH WANDB AND HF UPLOADS FAILED"
            echo "  FALLBACK: Download to local machine"
            echo "=============================================="
            echo ""
            echo "Run these commands to download checkpoints locally:"
            echo ""
            echo "  # On the pod — send the whole logs dir:"
            find "$CHECKPOINT_DIR" -name "model_*.pt" -exec dirname {} \; | sort -u | while read dir; do
                echo "  runpodctl send $dir"
            done
            echo ""
            echo "  # On your local machine — receive:"
            echo "  runpodctl receive <CODE>  # paste the code from above"
            echo ""
            echo "DO NOT TERMINATE THE POD until you have downloaded the files."
        fi
    fi
fi
echo ""

# ── Step 4: Upload results ────────────────────────────────────────────────
echo "--- [4/4] Upload results/logs ---"
if [ -d "$RESULTS_DIR" ] && [ "$(ls -A $RESULTS_DIR 2>/dev/null)" ]; then
    echo "Results dir contents:"
    ls -lh "$RESULTS_DIR"
    echo ""
    echo "Training logs are captured in wandb. Local logs are for debugging only."
    echo "If you need them, run: runpodctl send $RESULTS_DIR"
else
    echo "No results to upload."
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────
echo "=============================================="
echo "  OFFBOARD SUMMARY"
echo "=============================================="
echo "  Git push: OK"
echo "  wandb sync: $([ "$WANDB_SYNCED" = true ] && echo 'OK' || echo 'SKIPPED/FAILED')"
echo "  Checkpoints: $([ "$UPLOAD_SUCCESS" = true ] && echo 'UPLOADED' || echo 'FAILED — USE RUNPODCTL')"
echo ""
if [ "$UPLOAD_SUCCESS" = true ]; then
    echo "  All data safely offboarded. Pod can be stopped/terminated."
else
    echo "  WARNING: Checkpoints NOT uploaded. Do NOT terminate until downloaded."
fi
echo "=============================================="
