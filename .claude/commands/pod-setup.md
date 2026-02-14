Brand new pod setup from scratch. The user has created a fresh RunPod instance and needs everything installed.

IMPORTANT: Get the new SSH address from the user first if not provided. Update the SSH Address in the root CLAUDE.md.

Steps:
1. SSH in and run the system-level setup:
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
# Clone the repo first
cd /workspace
git clone https://github.com/rrzhang139/personal-research.git code/personal-research 2>/dev/null || (cd code/personal-research && git pull)
bash /workspace/code/personal-research/runpod/setup.sh
exit
SSHEOF
```

2. Set up the UWLab project environment (MUST run in tmux — takes ~10 min for Isaac Sim):
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
tmux new-session -d -s setup 'source /workspace/.bashrc_pod && cd /workspace/code/personal-research/uwlab && bash setup_env.sh 2>&1 | tee /workspace/results/setup.log'
echo "Setup running in tmux 'setup'"
exit
SSHEOF
```

3. Check setup progress periodically:
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
tail -20 /workspace/results/setup.log
tmux ls
exit
SSHEOF
```

4. After setup completes, clone the UWLab fork:
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
source /workspace/.bashrc_pod 2>/dev/null
cd /workspace/code/personal-research/uwlab
git clone https://github.com/rrzhang139/UWLab.git
cd UWLab && git remote add upstream https://github.com/uw-lab/UWLab.git
exit
SSHEOF
```

5. Download pretrained checkpoints:
```
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
mkdir -p /workspace/checkpoints
wget -O /workspace/checkpoints/cube_state_rl_expert.pt "https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/cube_state_rl_expert.pt"
exit
SSHEOF
```

6. Verify everything works (quick smoke test):
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

IMPORTANT:
- Isaac Sim install is ~12GB and takes ~10 min with uv. MUST run in tmux.
- Ensure `.bashrc_pod` has `set -a; source /workspace/.env 2>/dev/null; set +a` (not just `source`)
- Update SSH address in root CLAUDE.md after setup
