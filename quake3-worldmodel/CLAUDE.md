# Quake III World Model — Project Instructions

## Overview

Building the first world model for Quake III Arena. Fork DIAMOND (diffusion U-Net), collect data via DeepMind Lab, train on RunPod.

## Key Architecture Decisions

- **Data source**: DeepMind Lab (built on ioquake3, Quake III engine). Gymnasium wrappers via Shimmy.
- **Model**: Fork [DIAMOND](https://github.com/eloialonso/diamond) (MIT license). EDM diffusion, U-Net backbone, ~4-50M params.
- **Action conditioning**: Adaptive group normalization (DIAMOND's approach)
- **Resolution**: Start at 84x84, scale up later
- **Training hardware**: 1x RTX 4090 on RunPod ($0.20-0.39/hr)

## Project Structure

```
quake3-worldmodel/
├── CLAUDE.md           # This file
├── README.md           # Project overview
├── SCOPE.md            # Full scope document
├── setup_env.sh        # Environment setup
├── requirements.txt    # Python dependencies
├── data/               # Collected frame+action data (.npz)
├── src/                # Source code
│   ├── collect.py      # Data collection from DeepMind Lab
│   ├── dataset.py      # PyTorch dataset/dataloader
│   ├── model.py        # DIAMOND-adapted U-Net
│   ├── train.py        # Training loop
│   └── eval.py         # Evaluation pipeline
├── configs/            # Hydra configs (DIAMOND-style)
├── experiments/        # Timestamped experiment logs
└── scripts/            # Utility scripts
```

## Dependencies

- `dm_env`, `deepmind-lab` or `shimmy[dm-lab]` — DeepMind Lab gym wrapper
- `torch`, `torchvision` — PyTorch
- `wandb` — experiment tracking
- `lpips` — perceptual similarity metric
- `hydra-core` — configuration (DIAMOND uses this)
- `numpy` — data storage

## DeepMind Lab Setup Notes

DeepMind Lab requires building from source (Bazel build system). This is the most complex dependency.
- Repo: https://github.com/google-deepmind/lab
- Requires Bazel, Python dev headers, various system libs
- Alternative: use pre-built Docker images if available
- Shimmy provides Gymnasium wrappers: `shimmy.make("dm_lab/...")`

## W&B

- **Project**: `rzhang139/quake3-worldmodel`
- **Entity**: `rzhang139`
- Log: training loss, PSNR, LPIPS, FVD, rollout videos, side-by-side comparisons

## Development Philosophy

**Breadth-first, not depth-first.** Get each part working end-to-end at minimum viable quality, then iterate. Don't perfect the dataloader before seeing a single generated frame. Always find the bottleneck.

## Key References

- DIAMOND code: https://github.com/eloialonso/diamond
- DeepMind Lab: https://github.com/google-deepmind/lab
- Shimmy DMLab: https://shimmy.farama.org/environments/dm_lab/
- OpenArena: https://openarena.fandom.com/wiki/Main_Page
- GameNGen noise augmentation: https://arxiv.org/abs/2408.14837
- WHAMM (Quake II): https://www.microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments/

## Cost Tracking

| Date | Provider | GPU | Hours | Cost | Purpose |
|------|----------|-----|-------|------|---------|
| — | — | — | — | — | — |

## Gotchas / Lessons Learned

- DeepMind Lab build is finicky — may need specific Bazel version
- DMLab environments use simplified procedural mazes by default — need to configure for actual Q3-style maps or use OpenArena
- DIAMOND uses Hydra configs — study their config structure before modifying
- EDM diffusion (not DDPM) is critical for stable long rollouts
- GameNGen's noise augmentation on context frames prevents autoregressive drift — implement this
- Data collection is CPU-bound, no GPU needed — collect locally on Mac
- Always use `python -u` for unbuffered output when redirecting to log files on RunPod
