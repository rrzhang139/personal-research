# RunPod Cheap GPU — Testing & Validation

## When to Use
- **Smoke testing**: Verify code compiles and training loop starts (~5-10 min, costs $0.01-0.03)
- **Small-scale validation**: Train on a small dataset to check convergence (~1-2 hrs, costs $0.07-0.22)
- **Debugging**: Interactive SSH into a GPU machine to debug CUDA/rendering issues
- **Isaac Lab on a budget**: Single RTX GPU with Vulkan support for simulation testing

**Do NOT use for**: Full-scale multi-GPU production training (use the main RunPod multi-GPU pod instead — see `providers/runpod.md`).

## Pricing — Cheapest RunPod Options
RunPod community cloud has cheap single-GPU options. All billed per-second.

| GPU | VRAM | On-Demand $/hr | Spot $/hr | Isaac Lab? |
|-----|------|---------------|-----------|------------|
| A2 | 16 GB | $0.12 | **$0.06** | Possibly (limited Vulkan) |
| RTX 3070 | 8 GB | $0.13 | **$0.07** | Yes (basic envs) |
| RTX 3080 | 10 GB | $0.17 | **$0.09** | Yes |
| RTX A4000 | 16 GB | $0.17 | **$0.09** | Yes |
| RTX 3080 Ti | 12 GB | $0.18 | **$0.09** | Yes |
| RTX A5000 | 24 GB | $0.16 | **$0.11** | Yes |
| RTX 4070 Ti | 12 GB | $0.19 | **$0.10** | Yes |
| RTX 3090 | 24 GB | $0.22 | **$0.11** | Yes (recommended) |

**Recommended picks:**
- **Cheapest for Isaac Lab**: RTX 3070 spot at **$0.07/hr** (8GB VRAM, enough for small envs)
- **Best value**: RTX 3090 spot at **$0.11/hr** (24GB VRAM, runs any workload)
- **Pure PyTorch RL (no sim)**: A2 spot at **$0.06/hr** (cheapest CUDA GPU)

**Cost examples with per-second billing:**
| Task | RTX 3090 spot | RTX 3070 spot |
|------|--------------|--------------|
| 10-min smoke test | $0.018 | $0.012 |
| 1-hr validation | $0.11 | $0.07 |
| 4-hr small training | $0.44 | $0.28 |

## Tested & Verified (Feb 2026)

RTX A4000 community cloud pod — **confirmed working**:
- GPU: NVIDIA RTX A4000, 16GB VRAM, driver 550.144.03
- PyTorch 2.4.1+cu124, CUDA working
- System: 112 CPU cores, 503GB RAM (shared host machine)
- 20GB volume at /workspace
- Cost: $0.17/hr on-demand
- Boot time: ~90 seconds from API call to SSH-ready
- Vulkan: NOT pre-installed (need `apt-get install` for Isaac Lab)

### Availability Notes (from testing)
- **RTX 3090**: Unavailable ("not enough resources") — popular/scarce on community cloud
- **A2**: Unavailable ("no instances available") — very limited supply
- **RTX 3070**: Created successfully but **failed to boot** (stuck with runtime=null for 4+ min) — community host may have been offline
- **RTX 3080**: Created and booted successfully (~60s)
- **RTX A4000**: Created and booted successfully (~90s) — **most reliable in our test**
- **RTX 4000 Ada**: Created and booted successfully (~60s) but $0.20/hr

**Takeaway: Try multiple GPU types if your first pick is unavailable. RTX A4000 and RTX 3080 were the most reliably available.**

## GPU Driver Compatibility Matrix

**CRITICAL:** RunPod community cloud machines have mixed driver versions. The CUDA toolkit
in the container image is NOT the same as the driver version on the host.
Common failure: pip installs latest torch (cu130, requires driver 570+) → CUDA unavailable → silent CPU fallback.

| GPU (RunPod community) | Typical driver | Max CUDA | Recommended torch |
|------------------------|---------------|----------|-------------------|
| RTX A4000 | 550.144 | 12.4 | `cu121` or `cu124` |
| RTX 3090 | 520.xx | 11.8 | **`cu121`** (forward compat works) |
| RTX 3080 | 520–535 | 11.8–12.1 | `cu121` |
| RTX A5000 | 520–550 | 11.8–12.4 | `cu121` |
| A40 (SECURE) | 520.xx | 11.8 | **`cu121`** |
| A100 (SECURE) | 535+ | 12.1+ | `cu121` or `cu124` |

**Always install torch FIRST with explicit index, THEN verify CUDA before proceeding:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python3 -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:', ok, torch.cuda.get_device_name(0) if ok else 'FAIL'); sys.exit(0 if ok else 1)"
```

**The pod API query trick:** to get `podHostId` for SSH, you MUST include at least one non-runtime field in the query (e.g. `lastStatusChange`) — querying only `runtime { uptimeInSeconds }` returns null until the pod is fully up:
```bash
curl -s "https://api.runpod.io/graphql?api_key=$KEY" -H "Content-Type: application/json" \
  -d '{"query":"{ pod(input:{podId:\"POD_ID\"}) { id lastStatusChange machine { podHostId } } }"}'
```

**Community cloud pods can be preempted within 30 seconds.** For overnight training, always use `cloudType: SECURE` or `cloudType: ALL` (prefers secure). Community is only safe for <1hr smoke tests.

## Quick Start — Creating a Cheap Test Pod

### Via API (recommended for automation)
```bash
RUNPOD_API_KEY="$(grep apikey ~/.runpod/config.toml | cut -d'"' -f2)"

# Create a cheap 1x RTX A4000 community cloud pod (most reliable availability)
curl -s -H "Content-Type: application/json" \
  -d '{"query":"mutation { podFindAndDeployOnDemand(input: { name: \"test-cheap\", gpuTypeId: \"NVIDIA RTX A4000\", gpuCount: 1, cloudType: COMMUNITY, volumeInGb: 20, containerDiskInGb: 10, imageName: \"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04\", volumeMountPath: \"/workspace\", ports: \"22/tcp,8888/http\" }) { id desiredStatus } }"}' \
  "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"
```

If unavailable, try these GPU type IDs in order:
1. `NVIDIA RTX A4000` ($0.17/hr, 16GB) — most reliable
2. `NVIDIA GeForce RTX 3080` ($0.17/hr, 10GB)
3. `NVIDIA GeForce RTX 3090` ($0.22/hr, 24GB)
4. `NVIDIA RTX A5000` ($0.16/hr, 24GB)

### Getting the SSH Address (CRITICAL)
The RunPod API does NOT have a dedicated SSH address field. You MUST query `machine.podHostId`:

```bash
# After pod is created, get the SSH address:
curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"query":"{ pod(input: {podId: \"POD_ID\"}) { id machine { podHostId gpuDisplayName } runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }"}'

# The podHostId field IS the SSH username for the gateway:
# ssh -tt -i ~/.ssh/runpod <podHostId>@ssh.runpod.io
```

**Wait for `runtime` to be non-null** before attempting SSH (~60-120s after creation).

The SSH address format is: `<podHostId>@ssh.runpod.io`
Example: `n4wieselci8dmp-644119af@ssh.runpod.io`

### Via Dashboard
1. Go to https://www.runpod.io/console/pods
2. Click "Deploy" → Community Cloud
3. Select cheapest available GPU (sort by price)
4. Set volume to 20 GB (minimum needed)
5. Use image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
6. SSH address visible in pod details after boot

### Setup (Minimal — For Testing Only)
Unlike the full production pod, a cheap test pod needs minimal setup:
```bash
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS>@ssh.runpod.io << 'SSHEOF'
# Quick setup — no Claude Code, no dev user, just essentials
apt-get update && apt-get install -y tmux libglu1-mesa libgl1-mesa-glx libegl1-mesa build-essential cmake

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone repo
cd /workspace
git clone https://github.com/rrzhang139/personal-research.git code/personal-research

# Set env vars
cat > /workspace/.env << 'ENVEOF'
export WANDB_API_KEY=<key>
export HF_TOKEN=<token>
export GITHUB_TOKEN=<token>
export HDF5_USE_FILE_LOCKING=FALSE
ENVEOF
source /workspace/.env

# Set up the project you want to test
cd /workspace/code/personal-research/<project>
bash setup_env.sh
exit
SSHEOF
```

## Spot Instances — Important Caveats
Spot (community cloud) instances are cheap but can be interrupted:
- **5-second SIGTERM warning** before termination — not enough time to save a checkpoint
- Your instance can be taken at any time if someone outbids you
- **Mitigation**: Use checkpointing every N steps. For smoke tests (<30 min), interruption risk is low.
- **Data**: Volume persists through spot interruptions (pod is paused, not terminated)

For smoke testing, spot risk is acceptable — you're only running for 5-30 minutes.

## Destroy When Done
```bash
RUNPOD_API_KEY="$(grep apikey ~/.runpod/config.toml | cut -d'"' -f2)"

# Stop the pod (keeps volume, ~$0.005/hr idle)
curl -s -H "Content-Type: application/json" -d '{"query":"mutation { podStop(input: {podId: \"POD_ID\"}) { id desiredStatus } }"}' "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"

# Or terminate completely (destroys volume, no ongoing cost)
curl -s -H "Content-Type: application/json" -d '{"query":"mutation { podTerminate(input: {podId: \"POD_ID\"}) }"}' "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"
```

**For test pods: prefer terminate over stop.** The volume is disposable (no persistent data worth keeping). Terminating avoids the ~$5/month idle storage cost.

## Charging: When Does Billing Start?
- **Billing starts immediately** when the pod status becomes `RUNNING` — even before SSH is ready
- The pod boots in ~60-120s; you're billed for boot time
- **Billing stops** when you stop or terminate the pod
- Stopped pods: ~$0.005/hr for volume storage (20GB = ~$3.60/month)
- Terminated pods: $0.00 (volume destroyed)
- **Per-second billing**: You pay only for actual runtime. A 7-minute test on a $0.17/hr A4000 costs $0.020.

## vs Main RunPod Pod (providers/runpod.md)

| | Cheap Test Pod | Main Production Pod |
|-|---------------|---------------------|
| **GPUs** | 1x (RTX A4000, 3080, etc.) | 4x RTX 4090 |
| **Cost** | $0.07-0.22/hr | $1.36-1.92/hr |
| **Volume** | 20 GB (disposable) | 100 GB (persistent) |
| **Setup** | Minimal (no Claude Code) | Full (Claude Code, dev user, etc.) |
| **Use case** | Smoke tests, validation | Production training |
| **Lifecycle** | Create → test → terminate | Stop/resume, persistent |
| **Data** | Upload to wandb, then terminate | Persists on volume |
