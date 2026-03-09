# Research Ideas

Possible directions for experiments and extensions from the AlphaZero-from-scratch setup.

---

## MCTS & Search

- **Tree reuse across moves.** Right now we discard the MCTS tree after each move. The next root is a child of the previous root — reuse that subtree instead of rebuilding. Measure: sims-to-strength, wall-clock per move.
- **PUCT vs UCT.** Compare `sqrt(N_parent)` (current) vs `sqrt(ln(N_parent))` in the exploration term; sweep `c` / `c_puct` and report win rate vs sim count.
- **Virtual loss / batched MCTS.** Parallelize simulations: multiple threads run SELECT to different leaves, batch NN eval, then BACKPROP. Does strength scale with threads?
- **Gumbel AlphaZero / Sequential Halving.** Use Gumbel-based action selection for low-sim regimes; compare to PUCT at 25–100 sims.
- **KataGo-style ideas.** Payout cap randomization, auxiliary (e.g. ownership) targets, root uncertainty.

---

## Data & Training

- **Prior-only vs value-only vs both.** Ablate: train only policy head (imitate MCTS π), only value head (predict z), or both. How does sample efficiency and final strength change?
- **Visit-count distribution as target.** Compare cross-entropy vs MSE vs KL for policy loss; temperature on π (sharp vs soft targets).
- **Replay buffer content.** FIFO vs prioritized (e.g. by game outcome or position novelty); buffer size vs diversity vs overfitting.
- **Human data in the prior.** Add a small amount of (s, human_move) to the policy loss (AlphaGo-style) and see if it speeds early training or improves sample efficiency.

---

## Representation & Network

- **Residual CNN for board games.** Replace MLP with a small ResNet + policy/value heads for Connect4 or Othello; compare to MLP at same parameter count.
- **Board symmetry augmentation.** Use dihedral (rot/ref) to augment (s, π, z); re-index π for symmetries. Already noted for tic-tac-toe — extend to larger boards.
- **Separate vs shared policy/value.** Compare single trunk + two heads (current) vs two separate nets; same compute budget.

---

## Self-Play & Play

- **Cross-game information.** Currently parallel games share only the NN; no shared tree. Experiment: periodically inject “good” positions from one game into another’s buffer, or share a global prior cache (risky: distribution shift).
- **Simulations per move schedule.** Constant (e.g. 25) vs increasing over training vs adaptive (more sims in “critical” positions).
- **Temperature schedule.** Current: temp in [0, 1] for move sampling. Add annealing over iterations (high temp early, low late) and measure effect on diversity and strength.

---

## Games & Evaluation

- **Richer games.** Connect4, Othello, small Go — same pipeline, new `Game` impl; compare sample efficiency and sims-to-strength.
- **Arena and acceptance.** Sweep `update_threshold` and `arena_games`; plot win rate vs iteration and update frequency.

---

## High-Level / Open

- **Theory.** When is MCTS visit distribution a “policy improvement” over the network policy? Bounds or experiments on small trees.
- **Low-sim regime.** Make the system strong with 10–50 sims per move (deployment cost); combine tree reuse, Gumbel, and smaller/faster nets.
