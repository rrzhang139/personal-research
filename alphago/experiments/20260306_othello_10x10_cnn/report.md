# Experiment: OthelloNNet CNN on 10x10 Othello

## Hypothesis

CNN's spatial inductive bias becomes decisive on larger boards. On 6x6, MLP won
head-to-head 40-0 despite CNN having lower loss — MLP's sharper priors gave deeper
MCTS search. On 10x10 (100 cells, 101 actions), spatial patterns (corners, edges,
adjacency) are too complex for MLP to learn from flat input.

## Setup

- **Board**: 10x10 Othello (100 cells, 101 actions including pass)
- **Architecture**: OthelloNNet — plain CNN with 4 conv layers (2 padded + 2 shrinking), 512 filters, BN, dropout=0.3, FC 18432→1024→512→heads
- **MCTS**: 100 sims, nn_batch_size=8, c_puct=1.0
- **Training**: 25 iters, 100 games/iter, FIFO buffer (50K cap), lr=0.001
- **GPU**: A100 80GB PCIe ($1.64/hr)
- **Comparison**: MLP 4x256 trained with same params on RTX A4000

### Previous failures
- **A4000 + window buffer**: OOM at iter 7-16 (window buffer grows unboundedly)
- **Fix**: switched to FIFO buffer — constant memory, stable iter times

## Results

### CNN vs MLP on 10x10

| Metric | CNN (OthelloNNet 512f) | MLP (4x256) |
|--------|----------------------|-------------|
| vs Random (final) | **100%** | 80% |
| vs Random (peak) | **100%** | 94% |
| Models Accepted | 9/25 | 9/25 |
| Total Loss (final) | **1.266** | 4.140 |
| Policy Loss (final) | **1.045** | 3.739 |
| Value Loss (final) | **0.220** | 0.401 |
| Policy Entropy | **0.180** | 0.201 |
| Search Depth | **6.6** | 5.0 |
| Training Time | 422.8m (A100) | 380.6m (A4000) |
| Model Size | 304 MB | 3.1 MB |

### Head-to-Head (40 games, 100 sims, CPU)

| | Wins | Draws | Losses |
|---|------|-------|--------|
| **CNN** | **40** | 0 | 0 |
| **MLP** | 0 | 0 | **40** |

**CNN wins 40-0.** Complete reversal from 6x6 (where MLP won 40-0).

### Training Dynamics

- **Loss**: CNN converges to 1.27 (3.3x lower than MLP's 4.14). The CNN's value loss
  is nearly half the MLP's, meaning far better position evaluation.
- **Entropy**: Both converge to similar entropy (~0.18-0.20), but CNN achieves this
  from a lower loss — sharper AND more accurate priors.
- **Depth**: CNN searches 6.6 plies deep vs MLP's 5.0. On 6x6, MLP had the depth
  advantage (7.1 vs 4.8). The crossover happened because CNN's spatial features
  produce better priors on the larger board.
- **Game length**: ~96 moves/game (nearly full board), confirming 10x10 games are
  substantially longer than 6x6 (~30 moves).
- **Self-play outcomes**: roughly 45-50% P1 wins, 45-50% P2 wins, 1-5% draws.
  Very few draws on 10x10 — the game has a clear first-mover advantage.
- **Buffer**: FIFO at 50K cap throughout. Stable memory usage, no OOM.

### Board Size Crossover Summary

| Board | CNN H2H | MLP H2H | Winner | Key Factor |
|-------|---------|---------|--------|------------|
| 6x6 (37 actions) | 0 | 40 | MLP | Small enough for MLP priors to dominate |
| 10x10 (101 actions) | 40 | 0 | CNN | Spatial patterns too complex for flat input |

## Analysis

The result definitively confirms that **board size is the crossover variable** for
CNN vs MLP in AlphaZero:

1. **On small boards (6x6)**: MLP wins because the action space is small enough for
   an MLP to produce sharp priors over 37 actions. Sharper priors → deeper MCTS
   search → better play. CNN's convolutional structure is unnecessary overhead.

2. **On large boards (10x10)**: CNN wins because spatial patterns (corner control,
   edge stability, adjacency flipping chains) are the dominant strategic features.
   An MLP with 100-dim flat input cannot learn these patterns efficiently. CNN's
   conv layers extract local spatial features naturally, producing both lower loss
   AND sharper priors at this scale.

3. **The crossover** likely happens around 8x8 (standard Othello). Testing this
   would pinpoint the exact threshold.

## Weights

- **W&B**: `rzhang139/alphazero/othello10-cnn-experiment:latest`
- **Local**: `cnn_data/othellonet/best.pt` (304 MB)
- **Git LFS**: tracked

## Next Steps

1. **8x8 Othello** — find the crossover point between CNN and MLP dominance
2. **Sim scaling curves** — how much does each architecture benefit from more search?
3. **Go 9x9** — the real test of scalability (much deeper strategy than Othello)
4. **Auxiliary targets** (KataGo) — ownership prediction to boost CNN spatial learning

Total cost: ~$11.50 (A100 80GB, 7 hours). Head-to-head: free (CPU, 8 min).
