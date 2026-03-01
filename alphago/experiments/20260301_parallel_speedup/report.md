# Experiment: Parallel Self-Play Speedup

## Hypothesis

Parallel self-play using multiprocessing should significantly reduce wall time for games where per-game cost is non-trivial. Othello 6x6 at 200 MCTS simulations (~0.8s/game) should amortize the pool creation overhead that made parallelism counterproductive on tic-tac-toe (~0.03s/game).

## Setup

- **Game**: Othello 6x6
- **Network**: 4x128 MLP
- **MCTS**: 200 simulations, c_puct=1.0
- **Per iteration**: 50 self-play games, 20 arena games, 30 eval games
- **Iterations**: 3 per worker config
- **Workers tested**: 1, 2, 4, 8
- **Platform**: macOS (14 CPUs), spawn context (slower than fork on Linux)

### Config diff from baseline

| Parameter | Baseline | This experiment |
|-----------|----------|-----------------|
| game | tictactoe | othello |
| num_simulations | 25 | 200 |
| games_per_iteration | 100 | 50 |
| arena_games | 40 | 20 |
| num_iterations | 25 | 3 |

## Results

### Speedup Summary

| Workers | Self-Play | Arena | Eval | Total | Speedup |
|---------|-----------|-------|------|-------|---------|
| 1 | 40.0s | 15.8s | 11.9s | 68.6s | 1.00x |
| 2 | 25.1s | 10.7s | 8.2s | 44.8s | **1.53x** |
| 4 | 15.2s | 8.0s | 6.2s | 30.2s | **2.27x** |
| 8 | 12.9s | 8.3s | 7.3s | 29.4s | **2.33x** |

### Self-Play Phase Speedup

| Workers | Self-Play Time | Speedup | Efficiency |
|---------|---------------|---------|------------|
| 1 | 40.0s | 1.00x | 100% |
| 2 | 25.1s | 1.59x | 80% |
| 4 | 15.2s | 2.63x | 66% |
| 8 | 12.9s | 3.10x | 39% |

See `figures/speedup.png` for the full speedup curves and phase breakdown.
See `figures/per_iteration.png` for per-iteration timing stability.

## Analysis

### What worked

1. **Self-play scales well up to 4 workers** — 2.63x speedup at 4 workers (66% efficiency). Each game is independent with its own MCTS tree and read-only model, making it embarrassingly parallel.

2. **Total iteration speedup is meaningful** — 68.6s → 30.2s at 4 workers. For the Othello MLP baseline (16.7 min at 25 iters), this would cut to ~7-8 min.

3. **Consistent across iterations** — standard deviation of self-play time is <0.5s across all configs, showing stable performance.

### Diminishing returns at 8 workers

4→8 workers gives negligible additional speedup (30.2s → 29.4s). Three factors:

1. **Pool creation overhead (spawn on macOS)**: Creating a spawn-context pool takes ~2-3s. With 3 pool creations per iteration (self-play, arena, eval), that's ~9s of fixed overhead. This is the dominant cost at 8 workers.

2. **Amdahl's law**: Training (0.9s) is serial. At 8 workers, serial phases are ~30% of total time, capping max speedup at ~3x.

3. **Arena/eval have fewer games**: 20 arena games and 30 eval games don't have enough work to benefit from 8 workers (only 2.5-3.75 games/worker).

### Linux will be faster

On Linux (RunPod), the `fork` context avoids the spawn overhead entirely — no re-importing modules, no re-serialization. Expect:
- Pool creation: ~0.1s instead of ~2-3s
- Self-play speedup closer to linear (3.5x+ at 4 workers)
- Total speedup of 3x+ at 4 workers

### Optimal worker count

For this workload (50 games/iter, 200 sims on Othello):
- **macOS**: 4 workers is the sweet spot (2.27x total, diminishing beyond)
- **Linux (expected)**: 4-8 workers should scale nearly linearly for self-play

## Next Steps

1. **Re-run on Linux (RunPod)** to measure fork-context speedup — expect ~3-4x at 4 workers
2. **Pool reuse** — create one pool per iteration instead of per-phase (3x fewer pool creates, ~6s savings on macOS)
3. **CNN network test** — CNNs have heavier forward passes; parallelism may help more
4. **Othello CNN + parallel** — combine the CNN experiment with 4 workers to bring Othello training time under control
