Create a brand new RunPod pod and set up everything from scratch.

Arguments: $ARGUMENTS (optional: gpu count, e.g. "2" for 2x4090. Default: 4)

## Step 0: Create the pod via API

If no pod exists (check `providers/runpod.md`), create one:
```
RUNPOD_API_KEY="$(grep apikey ~/.runpod/config.toml | cut -d'"' -f2)"
GPU_COUNT="${1:-4}"
curl -s -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation { podFindAndDeployOnDemand(input: { name: \\\"uwlab-${GPU_COUNT}x4090\\\", gpuTypeId: \\\"NVIDIA GeForce RTX 4090\\\", gpuCount: $GPU_COUNT, cloudType: ALL, volumeInGb: 100, containerDiskInGb: 20, imageName: \\\"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04\\\", volumeMountPath: \\\"/workspace\\\", ports: \\\"22/tcp,8888/http\\\" }) { id desiredStatus } }\"}" \
  "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"
```

Then get the SSH address:
```
curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"query":"{ pod(input: {podId: \"<POD_ID>\"}) { machine { podHostId } } }"}' | python3 -m json.tool
# SSH address: <podHostId>@ssh.runpod.io
```

Update `providers/runpod.md` (Pod ID, SSH) AND `CLAUDE.md` Active Instances table.

Wait for SSH to be reachable (may take 1-3 min):
```
ssh -tt -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/runpod <SSH_ADDRESS> 'echo SSH_OK && exit' 2>/dev/null
```

## Step 1: Upload .env and run system setup

The `.env` file with auth tokens lives at `runpod/.env` in this repo. Upload it first, then run setup:
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
mkdir -p /workspace
cat > /workspace/.env << 'ENVEOF'
<paste contents of runpod/.env here>
ENVEOF
cd /workspace
git clone https://github.com/rrzhang139/personal-research.git code/personal-research 2>/dev/null || (cd code/personal-research && git pull)
bash /workspace/code/personal-research/runpod/setup.sh
exit
SSHEOF
```

CRITICAL: Read `runpod/.env` locally and embed its contents in the heredoc. Do NOT commit tokens to git.

## Step 2: Set up UWLab project (MUST run in tmux — takes ~10 min for Isaac Sim)
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
tmux new-session -d -s setup 'source /workspace/.bashrc_pod && cd /workspace/code/personal-research/uwlab && bash setup_env.sh 2>&1 | tee /workspace/results/setup.log'
echo "Setup running in tmux session: setup"
exit
SSHEOF
```

## Step 3: Check setup progress periodically
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
tail -20 /workspace/results/setup.log
tmux ls
exit
SSHEOF
```

## Step 4: After setup completes, clone UWLab fork and install
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research/uwlab
# Clone fork (not upstream) so we can push changes
git clone https://github.com/rrzhang139/UWLab.git
cd UWLab && git remote add upstream https://github.com/uw-lab/UWLab.git
# Install UWLab + rsl_rl into the venv
source /workspace/code/personal-research/uwlab/.venv/bin/activate
./uwlab.sh --install
./uwlab.sh --install rsl_rl
exit
SSHEOF
```

## Step 5: Download pretrained checkpoints
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
mkdir -p /workspace/checkpoints
wget -O /workspace/checkpoints/cube_state_rl_expert.pt "https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/cube_state_rl_expert.pt"
exit
SSHEOF
```

## Step 6: Verify everything works
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research/uwlab
source .venv/bin/activate
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import os; print(f'WANDB: {os.environ.get(\"WANDB_API_KEY\", \"NOT SET\")[:8]}...')"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
exit
SSHEOF
```

## IMPORTANT
- Isaac Sim install is ~12GB and takes ~10 min with uv. MUST run in tmux (SSH timeout kills it otherwise).
- `.bashrc_pod` uses `set -a; source /workspace/.env; set +a` to auto-export tokens.
- New pods take 1-3 min for SSH to become reachable after creation.
- After setup, update providers/runpod.md AND CLAUDE.md Active Instances table with new pod info.
