# Experiment: CNN vs MLP at Scale — Sim Sweep + 10x10 Othello

## Hypothesis

On 6x6, MLP won head-to-head 40-0 despite CNN having lower loss. Two possible
explanations: (1) MLP produces sharper priors at low sim counts, giving deeper
MCTS search. (2) 6x6 is too small for spatial patterns to matter.

This experiment tests both by:
1. **Sim sweep on 6x6**: head-to-head at 50/100/200 sims (existing weights)
2. **10x10 Othello**: train both architectures on a larger board where spatial
   patterns (corners, edges) should matter more

## Part 1: Sim Sweep on 6x6

Using weights from 20260304_othello_cnn_vs_mlp experiment.

| Sims | CNN Wins | Draws | MLP Wins | CNN Win Rate |
|------|----------|-------|----------|--------------|
| 25 | 0 | 0 | 10 | 0% |
| 50 | 0 | 0 | 10 | 0% |


## Analysis

TODO: fill in after results are collected.

Total experiment time: 0.3 minutes.
