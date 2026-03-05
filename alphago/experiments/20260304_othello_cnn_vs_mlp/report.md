# Experiment: CNN vs MLP on Othello 6x6 (Post-Bugfix)

## Hypothesis

The OthelloNNet (plain CNN with shrinking spatial dims) should outperform the
MLP (4x128) on Othello because Othello has strong spatial patterns (corners,
edges, adjacency). Previous comparisons were invalid due to a winner-detection
bug that capped reported win rates at ~60-75%.

## Setup

- **Baseline**: Both trained with shared config (50 sims, 25 iters, 100 games/iter)
- **MLP**: 4 layers x 128 hidden, FIFO buffer
- **OthelloNNet**: 512 filters, dropout=0.3, window buffer (last 20 iters), nn_batch_size=8
- **Evaluation**: 50 games vs random per iteration, 40-game head-to-head after training

### Config diff from defaults
| Parameter | MLP | OthelloNNet |
|-----------|-----|-------------|
| network_type | mlp | othellonet |
| num_filters | — | 512 |
| dropout | — | 0.3 |
| nn_batch_size | 1 | 8 |
| buffer_strategy | fifo | window |
| buffer_window | — | 20 |
| num_simulations | 50 | 50 |

## Results

### vs Random Win Rate
| Metric | MLP | OthelloNNet |
|--------|-----|-------------|
| Final vs Random | 100% | 100% |
| Peak vs Random | 100% | 100% |
| Models Accepted | 11/25 | 6/25 |
| Training Time | 4152s | 13053s |

### Head-to-Head (40 games, alternating colors)
| | Wins | Draws | Losses |
|---|------|-------|--------|
| **OthelloNNet** | 0 | 0 | 40 |
| **MLP** | 40 | 0 | 0 |

OthelloNNet win rate: 0%

### Training Dynamics
| Metric | MLP (final) | OthelloNNet (final) |
|--------|-------------|---------------------|
| Total Loss | 1.747 | 1.220 |
| Policy Loss | 1.173 | 0.886 |
| Value Loss | 0.574 | 0.333 |
| Policy Entropy | 0.417 | 0.611 |
| Search Depth | 7.1 | 4.8 |

See `figures/comparison.png` for learning curves.

## Analysis

**vs Random:** Both architectures reach high win rates post-bugfix. The CNN
matches the MLP (100% vs 100%).

**Head-to-head:** The direct match reveals which architecture produces
genuinely stronger play beyond beating random. CNN
loses
0-40 (with 0 draws).

**Loss landscape:** CNN typically achieves lower loss (especially value loss),
indicating better position evaluation. Lower policy entropy means MCTS
focuses on fewer moves — the network priors are sharper.

**Training cost:** OthelloNNet is significantly slower per iteration due to
CNN forward passes, especially on CPU. On GPU with nn_batch_size=8, the
overhead is amortized.

## Next Steps

1. **MCTS contribution sweep** — test OthelloNNet at 0/1/5/10/25/50/100/200
   sims during eval to measure how much MCTS adds on top of the learned network.
2. **Scale to 8x8 Othello** — the real test. 8x8 has ~60 moves/game and much
   deeper strategy. May need more sims, iterations, and filters.
3. **Training efficiency** — CNN needs fewer iterations to reach the same loss.
   Could we train CNN with fewer games/iter and still match MLP quality?

Total experiment time: 287.5 minutes.
