# Auto Memory — Personal Research

## Project: alphago (AlphaZero from Scratch)
- **W&B project**: https://wandb.ai/rzhang139/alphazero
- **W&B entity**: `rzhang139`
- **Weight storage**: W&B Artifacts (preferred) + Git LFS (backup for <10MB)
  - Upload: `wandb.Artifact(name, type='model').add_file(path)`
  - Download: `wandb.Api().artifact('rzhang139/alphazero/<name>:latest').download(dir)`
- **Artifacts uploaded**: tictactoe-mlp-baseline, connect4-mlp-baseline, othello6-mlp-baseline, othello6-mlp-experiment, othello10-mlp-experiment, othello10-cnn-experiment
- **Git LFS**: set up for `*.pt` files. Use `GIT_LFS_SKIP_SMUDGE=1 git clone` on pods to skip downloading existing weights
- **Pod setup**: always `rm -rf .venv && bash setup_env.sh` (not `uv pip install -e .` alone — misses torch/numpy)
- **FIFO vs window buffer**: window buffer OOMs on long runs (memory grows each iter). Use FIFO for stability.
- **Always commit ALL files** before ending session — don't skip unstaged changes
- **Active GPU run (2026-03-11)**: Pod `rwd3neocf5azky` (A4000, $0.17/hr), W&B `absurd-water-6`, 50 iters × 100 games, ~700s/iter. SSH: `rwd3neocf5azky-644110b6@ssh.runpod.io`
- **MCTS optimizations applied**: lazy expansion (95% fewer get_next_state), suicide fast-path (+19%), FPU reduction (depth 2x), eval() caching
- **CUDA batch=64 is 1.66x faster** than batch=8 on A4000. Use for next run.
- **KataGo params (from research)**: fpuReduction=0.2 (root=0.1), cpuct=1.0+log, dirichlet_alpha=0.12 for 9x9, temp halflife=19
- **v2 experiment ready**: `experiments/20260311_go9_v2/run.py` — SE blocks, no BN, global pool value, batch=64, cosine LR, all KataGo params
- **Go 9x9 needs ~10K-600K games** to reach amateur strength (MiniZero ref: 600K with 200 sims)
- **Key gap vs KataGo**: no auxiliary targets (ownership, score prediction). This is the #1 remaining improvement.

## Project: residual-rl (Policy Decorator)
- See `residual-rl-notes.md` for detailed notes
- SSH: `2djfma2zu7g1oh-644112fd@ssh.runpod.io` (changes on pod recreate)
- Always use launch scripts in `scripts/` dir, not raw python commands
- Always set `PYTHONPATH` and `HDF5_USE_FILE_LOCKING=FALSE`
- bs=1024 is optimal for wall-clock time (bs=4096 is slower per iteration)
- Eval is expensive: use --eval-freq 100000 for long runs
- MS2 obs_dim=50 vs MS3 obs_dim=43: diff is constant base_pose at idx 18-24
