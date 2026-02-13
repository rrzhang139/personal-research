# Residual-RL Project (Policy Decorator)

## Architecture & Code
- **Original codebase**: ManiSkill2 (`PegInsertionSide-v2`). We ported to MS3 (`PegInsertionSide-v1`)
- **MS2 obs_dim=50, MS3 obs_dim=43**: The 7-dim difference is `base_pose` (robot base pos `[-0.615,0,0]` + identity quat `[1,0,0,0]`) at indices 18-24. MS3 removed it because it's constant for stationary robots
- **To use original MS2 pretrained checkpoint on MS3**: Pad observations with the constant `base_pose=[-0.615, 0, 0, 1, 0, 0, 0]` at index 18 using `--pad-obs-to 50`. Do NOT pad with zeros
- **Pretrained checkpoints**: Download from [Google Drive](https://drive.google.com/drive/folders/1W0jwgVP9W1odt_F6vrwtYcnksdbwF1gi?usp=sharing) or W&B artifact `rzhang139/policy_decorator/diffusion_base_policy:latest`. Put under `./checkpoints/`
- **PYTHONPATH required**: Running `python offline/script.py` adds `offline/` not project root to sys.path. Always `export PYTHONPATH=/workspace/code/personal-research/residual-rl:$PYTHONPATH`
- **`proj` command unavailable in tmux**: Use explicit `source .venv/bin/activate` in tmux scripts instead

## Training Speed & Batch Size
- **~950 iters/min with bs=1024** on RTX 4090 is **expected and correct**. 1M iters ~ 17.5 hours
- **Larger batch size is SLOWER in wall-clock**: bs=4096 -> ~420 iters/min (40 hrs for 1M). More samples/min but fewer iters/min, and convergence doesn't improve per-iter
- **Keep bs=1024** (the paper default). The paper notes "2048 does not further improve"
- **Offline training does NOT benefit from GPU-parallelized envs**: Data is pre-loaded to GPU memory. Bottleneck is purely neural net forward/backward
- **Eval is expensive**: Each eval runs 100 episodes. Use `--eval-freq 100000` for long runs, not 1000

## Potential Speedups
- **Reduce total iterations**: Loss plateaus well before 1M. Try 300k-500k, check loss curve
- **`torch.compile()`**: Could speed up UNet forward/backward by 20-40% (untested)
- **Mixed precision (AMP)**: Marginal gains for this small model (4.5M params)
- **Reduce eval frequency**: `--eval-freq 100000` instead of 1000 saves hours
- **Run one job at a time**: Two GPU jobs share compute and both slow down ~50%

## MS3 Compatibility Fixes Applied
- `CPUNumpyWrapper`: Converts MS3 torch tensor obs/rewards to numpy for AsyncVectorEnv compatibility
- `PadObsWrapper`: Pads MS3 43-dim obs to MS2 50-dim with constant base_pose for using pretrained MS2 checkpoints
- Action normalization (`action_mean`/`action_std`): Added in commit `0601e38`. Original pretrained checkpoints do NOT use normalization
- `envs/maniskill_fixed.py`: Registers fixed MS3 environments for policy decorator

## Datasets & Checkpoints

### ManiSkill3 Demos
```bash
python -m mani_skill.utils.download_demo "PegInsertionSide-v1"
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
  --save-traj -c pd_ee_delta_pose -o state
```

### W&B Artifacts
```bash
wandb artifact get rzhang139/policy_decorator/diffusion_base_policy:latest \
  --root /workspace/checkpoints
```

## Key File Locations
| File | Description |
|------|-------------|
| `offline/diffusion_policy_unet_maniskill2.py` | Base diffusion policy training (offline IL) |
| `online/pi_dec_diffusion_maniskill2.py` | Policy Decorator residual RL (online) |
| `/workspace/checkpoints/best.pt` | Pretrained MS2 diffusion policy (from W&B) |
| `/workspace/checkpoints/best_adapted_ms3.pt` | Adapted checkpoint (MS2->MS3, obs weights reinitialized) |
| `/workspace/datasets/maniskill_demos/PegInsertionSide-v1/` | MS3 converted demos |
| `scripts/run_*.sh` | Launch scripts for various experiments |
