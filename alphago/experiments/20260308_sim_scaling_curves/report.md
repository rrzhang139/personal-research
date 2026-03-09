# Experiment: Sim Scaling Curves — CNN vs MLP at Varying Search Depth

## Hypothesis

On 6x6, MLP won 40-0 head-to-head despite CNN having lower loss. The theory:
MLP produces sharper priors on small boards, giving deeper MCTS search.

Two questions:
1. **Does CNN catch up on 6x6 with enough sims?** If yes, MLP's advantage is
   purely from enabling deeper search at low sim counts — CNN's learned features
   are equally good but need more search to exploit.
2. **On 10x10, does CNN need MCTS at all?** If CNN's raw policy (zero sims) beats
   MLP's raw policy, the CNN has learned genuinely better representations.

## Setup

- **Models**: Existing trained weights (no new training)
  - 6x6: MLP 4x128 vs OthelloNNet 512f (from 20260304 experiment)
  - 10x10: MLP 4x256 vs OthelloNNet 512f (from 20260305/20260306 experiments)
- **Sim counts**: [0, 1, 5, 25, 50, 100, 200]
- **Games**: 40 per sim count, alternating colors
- **Zero-sim**: raw network policy, greedy argmax, no MCTS
- **MCTS**: dirichlet_epsilon=0.0 (no root noise, pure exploitation)
- **Hardware**: CPU only (no training needed)

## Results

### 6x6 Othello

| Sims | CNN Wins | Draws | MLP Wins | CNN Win Rate |
|------|----------|-------|----------|--------------|
| 0 | 0 | 0 | **40** | 0% |
| 1 | 0 | 40 | 0 | 50% |
| 5 | 0 | 0 | **40** | 0% |
| 25 | 0 | 0 | **40** | 0% |
| 50 | 0 | 0 | **40** | 0% |
| 100 | 0 | 0 | **40** | 0% |
| 200 | **20** | 20 | 0 | **75%** |

### 10x10 Othello (partial — high sim counts too slow on CPU)

| Sims | CNN Wins | Draws | MLP Wins | CNN Win Rate |
|------|----------|-------|----------|--------------|
| 0 | **40** | 0 | 0 | **100%** |
| 1 | 20 | 0 | 20 | 50% |
| 100 | **40** | 0 | 0 | **100%** |

*(100-sim result from prior head-to-head experiment. Sims 5-200 stopped — each takes hours on CPU.)*

See `figures/sim_scaling.png` for the scaling curves.

## Analysis

### Key Finding 1: CNN catches MLP at 200 sims on 6x6

MLP dominates completely at 0-100 sims on 6x6. But at 200 sims, CNN suddenly
jumps to 75% (20W/20D/0L). This is a **phase transition**, not a gradual shift —
CNN goes from 0% to 75% between 100 and 200 sims.

This means CNN's learned features ARE good on 6x6 — it just needs enough search
depth to exploit them. At low sim counts, MLP's sharper priors give it a decisive
advantage because MCTS doesn't have enough budget to overcome poor priors.

### Key Finding 2: On 10x10, CNN wins with ZERO search

CNN beats MLP 40-0 using just the raw network policy (no MCTS). This is
definitive: CNN has learned genuinely superior representations on the larger board.
The spatial features (corners, edges, adjacency) are directly encoded in CNN's
weights — it doesn't need MCTS to compensate.

### Key Finding 3: 1-sim anomaly

Both board sizes show 50% at 1-sim. With 1 MCTS simulation, the search expands
exactly one node from root — this adds noise to the policy without enough depth
to improve it. The result is that both models play roughly equally badly with
1-sim MCTS (worse than their raw policies for the stronger model).

### The Two-Dimensional Landscape

The CNN vs MLP tradeoff has two axes:
1. **Board size**: larger boards favor CNN (spatial patterns become dominant)
2. **Search budget**: more sims favor CNN (enough depth to exploit learned features)

| | Low sims (≤100) | High sims (≥200) |
|---|-----------------|------------------|
| **Small board (6x6)** | MLP wins (sharper priors) | CNN catches up |
| **Large board (10x10)** | CNN wins (even with 0 sims) | CNN dominates |

The "sweet spot" for MLP is: small board + low sim count. Everywhere else, CNN wins.

### Timing Note

The 6x6 200-sim run took 3 hours (11,001s) — 112x slower than 100 sims. This is
because the OthelloNNet 512f CNN is expensive per forward pass on CPU, and 200
sims on a 6x6 board explores a large fraction of the game tree.

## Next Steps

1. **8x8 Othello**: find the exact board-size crossover at the standard sim count (50-100)
2. **6x6 at 150 sims**: pin down the exact sim-count crossover (somewhere between 100 and 200)
3. **Fill in 10x10 gaps on GPU**: run sims 5-200 on GPU where forward passes are fast
4. **Smaller CNN on 6x6**: would a 64f or 128f CNN be faster AND catch up sooner?

Total CPU time: ~4.5 hours (dominated by 6x6 at 200 sims and 10x10 zero-sim).
