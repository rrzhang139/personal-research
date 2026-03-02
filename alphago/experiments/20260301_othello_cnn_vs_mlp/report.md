# Experiment: CNN vs MLP on Othello (6x6)

## Hypothesis

The MLP baseline struggles on Othello (60% vs random, 5/25 models accepted in original baseline). Othello has strong spatial structure — flanking, corners, edges — that convolutions should exploit. A ResNet CNN should:
1. Learn faster (more models accepted in arena)
2. Reach higher win rates vs random
3. Produce lower loss (better understanding of move quality)

## Setup

- **Game**: Othello 6x6
- **MLP**: 4x128 (4 hidden layers, 128 units)
- **CNN**: 4 residual blocks, 64 filters
- **Shared config**: 50 sims, 25 iters, 100 games/iter, 10 epochs, lr=0.001, arena=40 games
- **MCTS**: nn_batch_size=32 (virtual loss batching for GPU utilization)
- **Platform**: RTX 3080 Ti (RunPod community cloud, $0.18/hr), sequential (num_workers=1)
- **Seed**: 42

### Why sequential + GPU

GPU single-sample inference has high transfer overhead (3-4x slower than CPU for batch=1). But with virtual loss batching (batch=32), the GPU amortizes transfer cost across 32 states per forward pass, making it faster. Cloud vCPUs are ~6x slower than Mac for MCTS, so GPU batch inference + fast GPU training is the right tradeoff.

## Results

| Metric | MLP (4x128) | CNN (4res/64f) |
|--------|-------------|----------------|
| **Final vs Random** | 64% | 62% |
| **Best vs Random** | 73% | **73%** |
| **Final Loss** | 1.83 | **1.37** |
| **Final Policy Loss** | 1.28 | **1.03** |
| **Final Value Loss** | 0.54 | **0.33** |
| **Models Accepted** | 6/25 | **9/25** |
| **Wall Time** | 27.8m | 65.2m |

### Key observations

1. **CNN achieves dramatically lower loss**: Final total loss 1.37 vs 1.83 (25% lower). Policy loss 1.03 vs 1.28 (20% lower). Value loss 0.33 vs 0.54 (39% lower). The CNN is learning a much better internal model of Othello.

2. **CNN accepts more models**: 9/25 vs 6/25. The CNN generates genuinely different (better) models that beat the previous version, indicating active learning rather than plateau.

3. **Win rates are similar**: Both peak at 73% vs random. This is surprising — the CNN's much lower loss doesn't translate to proportionally better play. Possible reasons:
   - 50 MCTS simulations may be enough to compensate for the MLP's weaker priors
   - The random opponent is too weak to distinguish between 73% and higher quality play
   - 25 iterations may not be enough for the CNN's advantage to fully manifest

4. **CNN learns faster initially**: By iter 7, CNN hits 64% vs random (MLP doesn't reach consistent 60%+ until iter 14). But CNN is noisier — win rate fluctuates between 47-73%.

5. **Self-play dynamics differ**: CNN produces more P1 wins (33-46 per iter vs MLP's 18-31), suggesting CNN develops more aggressive/decisive play.

## Analysis

The CNN is learning a genuinely better representation of Othello — 25% lower loss, 39% better value predictions, 50% more models accepted. But play strength at 50 sims doesn't reflect this gap because MCTS compensates for the MLP's weaker priors.

This suggests two follow-ups:
1. **Lower sim counts** (10-25) where network quality matters more → CNN advantage should widen
2. **Stronger opponents** (trained models instead of random) where position understanding matters

The CNN's higher model acceptance rate (9/25 vs 6/25) is the strongest signal — the CNN is actively improving throughout training while the MLP plateaus after iter 4.

## Infrastructure Learning

- **GPU single-sample MCTS is slow** — batch=1 on GPU is 3-4x slower than CPU due to transfer overhead. Batch=32 with virtual loss fixes this.
- **Cloud vCPUs are ~6x slower than Mac** for single-threaded Python MCTS. Sequential + GPU batch is the right pattern for cloud.
- **CNN training is 70x slower on CPU** (55ms vs 0.8ms per step). GPU is essential for CNN training (reduces to ~1-2ms per step).
- **RunPod 3090 unavailable** — 3080 Ti worked fine as fallback.

## Next Steps

1. **Lower sim sweep**: Compare CNN vs MLP at 10, 25, 50 sims — expect CNN advantage to grow at lower sims
2. **More iterations**: Run 50+ iters to see if CNN continues improving past MLP's plateau
3. **Head-to-head**: Pit the best CNN model directly against the best MLP model (not just vs random)
4. **Larger board**: Try 8x8 Othello where spatial patterns are even more important
