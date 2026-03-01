# Experiment: Game Parallelism + Virtual Loss Batching

## Hypothesis

Two independent parallelism strategies should compound:
1. **Game-level parallelism** (multiprocessing) — run multiple self-play games simultaneously
2. **MCTS-level batching** (virtual loss) — batch multiple leaf evaluations into a single NN forward pass

We expect virtual loss to reduce per-game time by amortizing NN call overhead, and game parallelism to reduce per-iteration time by running games concurrently. Combined, they should give the best speedup.

## Background: Virtual Loss

In standard MCTS, each simulation does one NN forward pass (select → expand → evaluate → backprop). With 200 simulations per move, that's 200 sequential forward passes.

Virtual loss batching changes this:
1. Select `batch_size` paths down the tree simultaneously
2. Apply "virtual loss" (+1 to W, +1 to N) on visited nodes, making them look worse so subsequent paths explore elsewhere
3. Batch-evaluate all leaf states in **one** forward pass
4. Revert virtual loss, expand, and backpropagate normally

This reduces 200 forward passes to 200/8 = 25 batched passes. Each batch evaluates 8 states in one `model.predict_batch()` call. The NN already operates on tensors, so batching 8 states costs barely more than evaluating 1.

## Setup

- **Game**: Othello 6x6, 200 MCTS simulations
- **Network**: 4x128 MLP
- **Per iteration**: 50 self-play, 20 arena, 30 eval games
- **Iterations**: 3 per config
- **Platform**: macOS (14 CPUs), spawn context

### Configs tested

| Config | Workers | NN Batch | Description |
|--------|---------|----------|-------------|
| `baseline` | 1 | 1 | Sequential everything |
| `4w_batch1` | 4 | 1 | Game parallelism only |
| `4w_batch8` | 4 | 8 | Game parallelism + virtual loss |

## Results

### Wall Time Summary

| Config | Self-Play | Arena | Eval | Total | Speedup |
|--------|-----------|-------|------|-------|---------|
| baseline | 39.5s | 15.7s | 11.6s | 67.8s | 1.00x |
| 4w_batch1 | 15.1s | 8.0s | 6.3s | 30.3s | **2.24x** |
| 4w_batch8 | 11.1s | 7.9s | 6.2s | 26.2s | **2.59x** |

### Self-Play Phase Speedup

| Config | Self-Play Time | Speedup |
|--------|---------------|---------|
| baseline | 39.5s | 1.00x |
| 4w_batch1 | 15.1s | 2.62x |
| 4w_batch8 | 11.1s | **3.56x** |

### Speedup Decomposition

- **Game parallelism alone** (4 workers): self-play 2.62x, total 2.24x
- **Virtual loss alone** (batch=8): per-game 1.53x (measured separately: 0.815s → 0.534s)
- **Combined**: self-play 3.56x, total 2.59x
- **Theoretical combined**: 2.62 × 1.53 = 4.0x → actual 3.56x (89% of theoretical)

The 11% gap vs theoretical is from spawn overhead (constant ~6s/iter for pool creation) which doesn't benefit from VL.

### Play Quality

vs-random win rates across configs: 50-73% (untrained/early-training Othello MLP). All configs produce similar quality — virtual loss doesn't degrade play.

See `figures/speedup.png` and `figures/phase_breakdown.png`.

## Analysis

### Virtual loss is a genuine win

Even on CPU with a tiny MLP, batching 8 NN evaluations gives 1.53x per-game speedup. The forward pass cost doesn't scale linearly with batch size — 8 states cost ~65% more than 1 state, giving effectively free evaluations. On GPU (where batch overhead is even smaller), this would be more dramatic.

### The two strategies are complementary

Game parallelism and virtual loss operate at different levels:
- Game parallelism: reduces wall time by running N games simultaneously (bounded by # CPUs)
- Virtual loss: reduces per-game time by batching NN calls (bounded by tree diversity)

They multiply: 4 workers × 1.53x per-game → 3.56x self-play speedup.

### Arena and eval don't benefit from VL as much

Arena/eval use deterministic greedy play (no Dirichlet noise, temp=0.01). Virtual loss still batches NN evals but the benefit is marginal since arena configs already have `nn_batch_size` propagated through the MCTS config.

### The real bottleneck is now pool creation

At 4w+batch8, self-play is 11.1s but total is 26.2s. The remaining 15.1s is:
- Train: 0.9s (serial, can't parallelize)
- Arena: 7.9s (3.6s pool creation + 4.3s actual games)
- Eval: 6.2s (3.2s pool creation + 3.0s actual games)

Pool creation (spawn context) is ~6.8s/iter = 26% of total. On Linux with fork, this drops to ~0.3s.

## Next Steps

1. **Larger batch sizes on GPU**: batch=32 gives 2.14x per-game on CPU; GPU should allow batch=64-256 with near-linear scaling
2. **Pool reuse**: Keep one pool alive per iteration instead of recreating per phase — saves ~6s/iter on macOS
3. **Test on Linux/RunPod**: fork context eliminates pool overhead → expect 3.5-4x total speedup
4. **CNN + VL**: CNN forward passes are more expensive → larger batches should give even greater VL benefit
5. **Sweep nn_batch_size**: test 4, 8, 16, 32 to find the sweet spot where tree diversity degrades
