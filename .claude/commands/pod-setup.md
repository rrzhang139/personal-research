Create a brand new RunPod pod and set up everything from scratch.

Arguments: $ARGUMENTS (optional: gpu count, e.g. "2" for 2x4090. Default: 4)

## Timing Reference (plan your sleeps/timeouts around these)

| Phase | Duration | Timeout to use |
|-------|----------|----------------|
| Pod creation API call | instant | 10s |
| SSH becomes reachable | 1-2 min | Poll every 15s, max 3 min |
| Step 1 SSH (upload .env + setup.sh + clone fork + launch setup_env.sh) | ~3 min | 300s (5 min) |
| setup_env.sh running in tmux (Isaac Sim download is bottleneck) | ~8-10 min | sleep 480 before checking |
| Step 2 SSH (finish: download checkpoint + patch + verify) | ~1 min | 120s (2 min) |
| **Total end-to-end** | **~15 min** | |

## Step 0: Create pod via API + get SSH address

Run these locally (no SSH needed):
```bash
RUNPOD_API_KEY="$(grep apikey ~/.runpod/config.toml | cut -d'"' -f2)"
GPU_COUNT=4  # or from $ARGUMENTS

# Create pod
curl -s -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation { podFindAndDeployOnDemand(input: { name: \\\"uwlab-${GPU_COUNT}x4090\\\", gpuTypeId: \\\"NVIDIA GeForce RTX 4090\\\", gpuCount: $GPU_COUNT, cloudType: ALL, volumeInGb: 100, containerDiskInGb: 20, imageName: \\\"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04\\\", volumeMountPath: \\\"/workspace\\\", ports: \\\"22/tcp,8888/http\\\" }) { id desiredStatus } }\"}" \
  "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"

# Get SSH address (use pod ID from above)
curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"query":"{ pod(input: {podId: \"<POD_ID>\"}) { machine { podHostId } } }"}' | python3 -m json.tool
# SSH_ADDRESS = <podHostId>@ssh.runpod.io
```

Update `providers/runpod.md` (Pod ID, SSH) AND `CLAUDE.md` Active Instances table immediately.

Then poll for SSH readiness (timeout 15s per attempt, try up to 6 times):
```bash
ssh -tt -o ConnectTimeout=15 -o StrictHostKeyChecking=no -i ~/.ssh/runpod <SSH_ADDRESS> 'echo SSH_OK && exit' 2>/dev/null
```
If it fails, `sleep 15` and retry. If still failing after 3 min total, the pod may be stuck — check API status.

## Step 1: Upload .env + system setup + clone fork + launch Isaac Sim install (ONE SSH call)

Read `runpod/.env` locally and embed contents. This single SSH call does everything up to launching the long Isaac Sim install:
```bash
# Timeout: 300s (5 min). setup.sh takes ~2-3 min, clone takes ~30s.
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << SSHEOF
mkdir -p /workspace /workspace/results
cat > /workspace/.env << 'ENVEOF'
<contents of runpod/.env>
ENVEOF

# Clone repo + system setup (~2-3 min)
cd /workspace
git clone https://github.com/rrzhang139/personal-research.git code/personal-research 2>/dev/null || (cd code/personal-research && git pull)
bash /workspace/code/personal-research/runpod/setup.sh
source /workspace/.bashrc_pod

# Clone UWLab fork BEFORE setup_env.sh so it uses the fork (setup_env.sh skips clone if dir exists)
cd /workspace/code/personal-research/uwlab
git clone https://github.com/rrzhang139/UWLab.git 2>/dev/null || echo "UWLab already cloned"
cd UWLab && git remote add upstream https://github.com/uw-lab/UWLab.git 2>/dev/null || true
cd ..

# Launch Isaac Sim install in tmux (takes ~8-10 min, SSH would timeout)
tmux new-session -d -s setup 'source /workspace/.bashrc_pod && cd /workspace/code/personal-research/uwlab && bash setup_env.sh 2>&1 | tee /workspace/results/setup.log'
echo "TMUX_STARTED: setup_env.sh running"
tmux ls
exit
SSHEOF
```

CRITICAL: Read `runpod/.env` locally and embed its contents. Do NOT hardcode tokens.

## Wait for Isaac Sim install

`sleep 480` (8 min). Then check:
```bash
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
tail -5 /workspace/results/setup.log
tmux ls 2>/dev/null || echo "tmux exited (setup done)"
exit
SSHEOF
```
If still running, `sleep 120` and check again. Look for "Setup complete" in the log.
If tmux session is gone, setup finished.

## Step 2: Finish setup — download checkpoint + apply patches + verify (ONE SSH call)

```bash
# Timeout: 120s (2 min). All fast operations.
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null

# Pull latest code (in case new scripts were pushed after pod creation)
cd /workspace/code/personal-research && git pull

# Download pretrained checkpoints (~10s)
mkdir -p /workspace/checkpoints
wget -q -O /workspace/checkpoints/cube_state_rl_expert.pt \
  "https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/cube_state_rl_expert.pt"

# Apply any patches needed (e.g., adaptive resets)
cd /workspace/code/personal-research/uwlab
source .venv/bin/activate
python scripts/patch_adaptive.py 2>/dev/null || echo "No patch to apply"

# Smoke test
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python3 -c "import os; print(f'WANDB: {os.environ.get(\"WANDB_API_KEY\", \"NOT SET\")[:8]}...')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ls -la /workspace/checkpoints/

echo "=== POD SETUP COMPLETE ==="
exit
SSHEOF
```

## Summary: Only 2 SSH calls needed after pod creation

1. **SSH call 1** (timeout 5 min): .env + setup.sh + clone fork + launch tmux
2. **sleep 480** (8 min wait for Isaac Sim)
3. **SSH call 2** (timeout 2 min): git pull + checkpoint + patch + verify

Total wall-clock: ~15 min. Total SSH sessions: 2 (plus polling for readiness).

## IMPORTANT
- setup_env.sh clones UWLab via `git clone ... 2>/dev/null || echo "already cloned"` — by cloning the fork FIRST in Step 1, setup_env.sh will skip the clone and use the fork.
- Auth tokens live at `runpod/.env` locally (NOT committed to git). They get uploaded to `/workspace/.env` on the pod.
- `.bashrc_pod` uses `set -a; source /workspace/.env; set +a` to auto-export tokens for Python/wandb/HF.
