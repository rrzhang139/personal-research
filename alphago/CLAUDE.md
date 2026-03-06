# AlphaZero from Scratch — Research Playbook

## Philosophy

You are the **research director**, not the implementer. Your workflow:
1. **Read code** to understand how things work
2. **Hypothesize** — "what if I increase MCTS simulations from 25 to 100?"
3. **Direct experiments** — tell Claude what to run, with what parameters
4. **Read results** — experiment reports tell you what happened and why
5. **Conclude and iterate** — each experiment informs the next

Claude handles all implementation. You accumulate a growing surface area of **research knobs** — techniques from papers, blog posts, and your own experiments — that you can mix, match, and tune.

## Quick Start

```bash
cd alphago && bash setup_env.sh
python scripts/train.py --game tictactoe            # train (~3 min CPU)
python scripts/play.py --game tictactoe              # play against it
python scripts/train.py --game tictactoe --wandb     # train with W&B logging
```

---

## Historical Background: How We Got Here

### The Problem: Go Was Unsolvable

Chess engines dominated humans by 1997 (Deep Blue). But Go — with its 250-wide branching factor (vs 35 in chess) and positions that defy handcrafted evaluation — resisted all traditional AI for decades. Minimax with alpha-beta pruning, the backbone of chess engines, was hopeless: you can't search deep enough, and you can't score positions reliably.

### Phase 1: Monte Carlo Methods (1993–2006)

**Key idea:** if you can't evaluate a position analytically, *estimate* its value by playing random games from it and averaging the outcomes.

Brugmann (1993) first tried this for Go. It was crude but worked: play thousands of random games from a position, count how often each side wins, use that as the value estimate. No tree, no search — just raw statistics.

**Coulom (2006)** combined this with incremental tree building, coining **"Monte Carlo Tree Search"**. The four-phase algorithm:

```
1. SELECT    — walk down the tree, picking "best" child at each node
2. EXPAND    — add a new leaf node
3. SIMULATE  — play a random game (rollout) from the leaf to completion
4. BACKPROP  — propagate the win/loss result back up the path
```

### Phase 2: UCT Makes It Principled (2006)

**Kocsis & Szepesvari (2006)** recognized that choosing which child to select is a multi-armed bandit problem. They applied the UCB1 formula:

```
Score(a) = Q(a) + c * sqrt(ln(N_parent) / N(a))
```

This is **UCT** (Upper Confidence bounds applied to Trees). The first term exploits (pick moves with high win rates), the second explores (pick under-visited moves). With enough simulations, UCT provably converges to the optimal move.

By 2012, UCT-based programs (MoGo, Crazy Stone, Pachi) reached low amateur-dan level in Go. Strong, but nowhere near professional.

**The bottleneck:** random rollouts are a terrible way to evaluate positions. Playing random moves to the end of a Go game is like estimating a book's quality by reading random sentences.

### Phase 3: AlphaGo — Neural Networks Meet MCTS (2016)

**Silver et al., Nature 2016.** Two breakthrough modifications:

1. **Policy network as prior.** Instead of exploring children equally, a deep CNN trained on human expert games provides `P(s,a)` — a prior probability for each move. The selection formula becomes **PUCT**:
   ```
   Score(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
   ```
   The `ln()` is gone, replaced by `sqrt()`. The prior `P(a)` multiplicatively scales the exploration bonus. Moves the network likes get explored first; moves it ignores are effectively pruned.

2. **Value network replaces (most) rollouts.** A separate CNN predicts the game outcome directly from the board position. Leaf evaluation becomes:
   ```
   V(leaf) = 0.5 * value_network(leaf) + 0.5 * random_rollout(leaf)
   ```
   The value network is far less noisy than random rollouts.

**Result:** AlphaGo defeated Lee Sedol 4-1 (March 2016). First superhuman Go AI.

### Phase 4: AlphaGo Zero — Pure Self-Play, No Rollouts (2017)

**Silver et al., Nature 2017.** Three simplifications that made it *stronger*:

1. **Rollouts completely eliminated.** Leaf evaluation uses *only* the value head of the network. The "Monte Carlo" simulation that gave MCTS its name is gone. In our code, this is `search.py` line 69: `policy, value = self.model.predict(canonical)` — one forward pass, no random games.

2. **Single dual-headed network.** One CNN with shared trunk, two output heads: policy (where to play) and value (who's winning). One forward pass gives both the prior for selection AND the value for leaf evaluation.

3. **No human data.** Training starts from random weights. MCTS acts as a "policy improvement operator": the visit-count distribution from search is *better* than the raw network policy. Train the network to match MCTS outputs → network improves → MCTS with improved network is even better → repeat.

**Result:** Surpassed the Lee Sedol version of AlphaGo in 3 days of training.

### Phase 5: AlphaZero — Same Algorithm, Any Game (2017)

**Silver et al., Science 2018.** The same algorithm — zero changes to MCTS — applied to chess, shogi, and Go. Defeated Stockfish (chess) and the 3-day AlphaGo Zero (Go) within 24 hours from random initialization.

**This is what our codebase implements.** The core algorithm is game-independent. Our `base_game.py` interface is the abstraction boundary: swap in any two-player zero-sum game and training just works.

### What Came After

| System | Year | Key Innovation | Changed in MCTS? |
|--------|------|----------------|-------------------|
| **KataGo** | 2019 | 50x more efficient training: playout cap randomization, auxiliary targets (ownership, score prediction) | No — training improvements only |
| **MuZero** | 2020 | Learned dynamics model — MCTS plans in latent space without knowing game rules | Yes — expansion uses learned model instead of true game rules |
| **Gumbel AlphaZero** | 2022 | Sequential Halving at root with Gumbel noise — provably improves policy even with 8 simulations | Yes — new selection/move-choice mechanism |

---

## Understanding the Code: What Each Piece Does

### The Training Loop (the big picture)

```
┌─────────────────────────────────────────────────┐
│  for each iteration:                            │
│                                                 │
│  1. SELF-PLAY: play N games using MCTS+model    │
│     → produces (board, mcts_policy, outcome)    │
│                                                 │
│  2. TRAIN: update a copy of the model on the    │
│     accumulated replay buffer                   │
│                                                 │
│  3. ARENA: new model vs old model               │
│     if new wins enough → accept, else reject    │
│                                                 │
│  4. EVALUATE: best model vs random player       │
└─────────────────────────────────────────────────┘
```

### MCTS Search (the core algorithm)

**File: `mcts/search.py`** — implements AlphaZero-style MCTS exactly.

```
For each simulation (default 25):

  1. SELECT: start at root, pick child with highest PUCT score
     PUCT(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
              ─────   ──────────────────────────────────────────
              exploit              explore

     Keep descending until we hit a leaf (unexpanded node).

  2. EXPAND: ask the neural network for P(a) for all actions
     Create child nodes with those priors.

  3. EVALUATE: the neural network also returns a value v.
     No rollout. Just the network's estimate.

  4. BACKPROP: propagate v back up the tree.
     Negate at each level (zero-sum: what's good for me is bad for you).

After all simulations:
  Return visit counts as the policy (more visits = better move).
```

**Why it works:** MCTS acts as a policy improvement operator. The network's raw policy `P(a)` is imperfect. But searching with PUCT concentrates visits on genuinely good moves (combining the network's intuition with lookahead). The visit distribution `pi(a)` after search is *better* than `P(a)`. Training the network to match `pi` makes it better, which makes the next round of search even better.

### The Neural Network

**File: `neural_net/simple_net.py`** — MLP with shared trunk and two heads.

```
Board State (9 floats for tic-tac-toe)
    │
    ▼
┌─────────────┐
│ Shared Trunk │  ← num_layers × hidden_size, ReLU activations
│  (MLP)       │
└─────┬───────┘
      │
   ┌──┴──┐
   │     │
   ▼     ▼
Policy  Value
 Head    Head
  │       │
  ▼       ▼
 P(a)    v(s)
softmax  tanh
```

- **Policy head:** "Where should I play?" → probability over all actions
- **Value head:** "Who's winning?" → scalar in [-1, +1]

---

## Parameter Guide: What Each Knob Does

### MCTS Parameters (`MCTSConfig`)

| Parameter | Default | What it controls | Turn it up | Turn it down |
|-----------|---------|------------------|------------|--------------|
| `num_simulations` | 25 | Search depth/breadth. More sims = stronger play but slower. | Stronger but slower. Training cost scales linearly. | Weaker but faster self-play. Below ~10 sims, MCTS barely improves on the raw network. |
| `c_puct` | 1.0 | Exploration vs exploitation tradeoff. | More exploration: tries unusual moves, slower convergence, but finds surprises. | More exploitation: focuses on what looks best, faster convergence, but might miss better lines. |
| `dirichlet_alpha` | 0.3 | Noise concentration. Scaled inversely with action space (0.3 for chess/9-action, 0.03 for Go/361-action). | Uniform noise: all moves get some exploration. | Concentrated noise: one random move gets boosted, others stay. |
| `dirichlet_epsilon` | 0.25 | How much noise vs network prior at root. | More random root exploration. Prevents the network from getting stuck in a rut. | Trust the network more. Less exploration at root. |
| `temp_threshold` | 15 | Move number where temperature drops to ~0 (greedy). | Explore longer in each game. More diverse training data but noisier play. | Exploit sooner. More consistent play but less diverse data. |

**The key insight about `num_simulations`:** This is your primary quality-vs-speed dial. At 25 sims, tic-tac-toe trains in 3 minutes. At 800 sims (AlphaZero's chess setting), each self-play game takes ~30x longer. For tic-tac-toe, 25 is plenty. For chess, you need 800. The value of more sims depends on how complex the game is.

**The key insight about `c_puct`:** If too high, the search wastes time on bad moves. If too low, it never discovers moves the network didn't initially predict. For a game like tic-tac-toe where the network quickly learns the right moves, c_puct matters less. For harder games, tuning this is important.

### Network Parameters (`NetworkConfig`)

| Parameter | Default | What it controls | Turn it up | Turn it down |
|-----------|---------|------------------|------------|--------------|
| `hidden_size` | 128 | Width of each layer. Capacity. | More representational power. Needed for complex games. Slower training. | Faster but less expressive. Fine for tic-tac-toe. |
| `num_layers` | 4 | Depth. How many transformations. | Can learn more abstract features. Risk of optimization difficulty. | Simpler model. Easier to train but limited abstraction. |

**For tic-tac-toe:** 4x128 is overkill. Even 2x32 works. The defaults are set to leave room for harder games.

### Training Parameters (`TrainingConfig`)

| Parameter | Default | What it controls | Turn it up | Turn it down |
|-----------|---------|------------------|------------|--------------|
| `lr` | 0.001 | Learning rate. Step size for gradient descent. | Faster learning, risk of instability/overshooting. | More stable but slower convergence. |
| `batch_size` | 64 | Minibatch size. | More stable gradients, higher throughput on GPU. | Noisier gradients, can act as regularizer. |
| `epochs_per_iteration` | 10 | Passes over the replay buffer per iteration. | More thorough training on current data. Risk of overfitting to buffer. | Less overfitting, but might under-train. |
| `num_iterations` | 25 | Total self-play → train → arena cycles. | More training. Diminishing returns once the game is "solved". | Faster total time. |
| `games_per_iteration` | 100 | Self-play games per iteration. | More training data per iter. Higher quality buffer. | Faster iterations but sparser data. |
| `max_buffer_size` | 50,000 | Replay buffer capacity. | More historical data, smoother training. | Focuses on recent games, adapts faster to new model. |

**The key training tradeoff:** `games_per_iteration × num_simulations` determines how much compute goes into self-play (the expensive part). `epochs_per_iteration` determines how much you squeeze out of that data. If you over-train (high epochs, small buffer), the network memorizes specific games. If you under-train, you waste the self-play compute.

### Arena Parameters (`ArenaConfig`)

| Parameter | Default | What it controls | Turn it up | Turn it down |
|-----------|---------|------------------|------------|--------------|
| `arena_games` | 40 | Games played to compare models. | More confident accept/reject decision. More compute. | Faster but noisier — might accept a worse model or reject a better one. |
| `update_threshold` | 0.55 | Win rate needed to accept new model. | Conservative: only accept clearly better models. Slower improvement but stable. | Aggressive: accept marginal improvements. Faster iteration but risk of accepting noise. |

---

## Diagnostic Metrics (W&B)

When running with `--wandb`, these metrics help you understand training dynamics:

| Metric | What it tells you | Healthy trend |
|--------|-------------------|---------------|
| `loss/total` | Overall training signal | Decreasing, eventually plateaus |
| `loss/policy` | Is the network learning what MCTS recommends? | Decreasing (network matches MCTS better) |
| `loss/value` | Is the network learning to evaluate positions? | Decreasing (value predictions get more accurate) |
| `vs_random` | Absolute strength | Increasing toward 95-100% |
| `arena_win_rate` | Is the new model better? | Hovering around 50-75% (draws increase as both models get strong) |
| `mcts/policy_entropy` | How confident is the search? | Decreasing (MCTS becomes more focused) |
| `mcts/mean_root_value` | How does the model evaluate starting positions? | Trending toward 0 for balanced games (draws) |
| `mcts/mean_search_depth` | How deep does MCTS look? | Increasing (sharper priors → deeper search) |
| `self_play/mean_game_length` | How long are games? | Stabilizes. Sudden shortening may indicate a forced-win strategy |
| `self_play/draw_rate` | Are games becoming draws? | For tic-tac-toe, should increase as the model learns optimal play (tic-tac-toe is a draw with perfect play) |

---

## Research Surface Areas

A living catalog of tunable knobs. Each row is something you can experiment with.

| Surface Area | Current State | Knobs | Source |
|---|---|---|---|
| **MCTS** | PUCT + virtual loss batching | `num_simulations`, `c_puct`, `dirichlet_alpha`, `dirichlet_epsilon`, `temp_schedule`, `nn_batch_size` | [AlphaZero](https://arxiv.org/abs/1712.01815) |
| **Network** | MLP + CNN | `hidden_size`, `num_layers`, `num_filters`, `num_res_blocks`, `network_type` | — |
| **Training** | Parallel self-play | `lr`, `batch_size`, `num_iterations`, `games_per_iteration`, `epochs_per_iteration`, `num_workers` | — |
| **Arena** | Win-rate threshold | `update_threshold`, `arena_games` | — |
| **Replay Buffer** | Fixed-size FIFO | `max_buffer_size` | — |

### Future Surface Areas (added through discourse)
- Board symmetry augmentation (already implemented — 8x free data for tic-tac-toe)
- Learning rate schedules (cosine, step decay)
- Weight decay / L2 regularization
- Temperature annealing across training (not just within games)
- Tree reuse between moves (reuse subtree after a move is played)
- Pool reuse (keep one pool per iteration instead of per-phase — saves ~6s/iter on macOS)
- Go game implementation
- KataGo innovations (playout cap randomization, auxiliary targets)
- Gumbel AlphaZero (Sequential Halving — provably good at low sim counts)

## Running on GPU (RunPod)

For experiments that take >10 min on CPU (Othello, larger games, CNN networks), use a cheap RunPod pod.

**Setup pattern:**
```bash
# 1. Push code to git first (local)
cd personal-research && git add alphago/ && git commit -m "..." && git push

# 2. Create pod via API (RTX A4000 = $0.17/hr, most reliable availability)
# See providers/runpod-cheap.md for the curl command

# 3. SSH and setup (one-time per pod)
ssh -tt -i ~/.ssh/runpod <podHostId>@ssh.runpod.io << 'SSHEOF'
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd /workspace
git clone https://github.com/rrzhang139/personal-research.git code/personal-research
cd code/personal-research/alphago && bash setup_env.sh
exit
SSHEOF

# 4. Launch experiment (MUST use python -u for unbuffered output to log file)
ssh -tt -i ~/.ssh/runpod <podHostId>@ssh.runpod.io << 'SSHEOF'
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/code/personal-research/alphago && source .venv/bin/activate
nohup python -u experiments/<exp>/run.py > /workspace/exp_output.log 2>&1 &
echo "PID: $!"
exit
SSHEOF

# 5. Monitor progress
ssh -tt -i ~/.ssh/runpod <podHostId>@ssh.runpod.io << 'SSHEOF'
tail -30 /workspace/exp_output.log
exit
SSHEOF

# 6. Pull results back — MUST include .pt weights! (from pod)
ssh -tt ... << 'SSHEOF'
cd /workspace/code/personal-research && git add -f alphago/experiments/<exp>/ && git commit -m "results" && git push
exit
SSHEOF

# 7. Pull locally (from local machine)
cd personal-research && git pull
# Verify weights are present:
ls -la alphago/experiments/<exp>/data/**/*.pt
```

**Key gotchas:**
- **ALWAYS pull weights locally before terminating a pod.** Weights are expensive to recreate. `git add -f` is needed because `.pt` files may be gitignored.
- **Large weights (>100MB)**: GitHub rejects files >100MB. OthelloNNet 512f on 10x10 = 304MB. Options: (1) compress with `gzip` before push, (2) use Git LFS, (3) reduce num_filters. **Set up git config + credentials on pod BEFORE training starts** so push works without scrambling: `git config user.email/name` + `git remote set-url origin https://<TOKEN>@github.com/...`. Reset URL after push.
- **GPU + virtual loss**: With `--nn-batch-size 8+`, GPU can now help via batched forward passes. Without batching (batch=1), GPU transfer overhead makes it 3-4x SLOWER than CPU. Always use `--nn-batch-size 8` or higher with GPU.
- `tmux` is NOT pre-installed on cheap pods — use `nohup` or install tmux first (`apt-get install -y tmux`)
- Python buffers stdout when redirected to file — always use `python -u` (unbuffered)
- RunPod SSH gateway requires heredoc (`<< 'SSHEOF'`) — passing commands as args is ignored. **SCP/SFTP do NOT work** through the gateway — use `git push` or pipe via `ssh -tt ... cat`.
- Terminate pod when done to avoid idle storage charges ($0.005/hr)
- Cloud vCPUs are much slower than Mac for single-threaded Python — MLP Othello baseline: 16.7m local vs 97m on pod

## Experiment Workflow

Use the `/experiment` slash command:

```
/experiment <description of what to try>
```

Creates a timestamped folder under `experiments/` with `config.json`, `data/`, `figures/`, and `report.md`.

**After every experiment or implementation finishes, you MUST do ALL of these before moving on:**

1. **Pull weights and results locally** — if the experiment ran on a remote machine (RunPod, etc.), always pull back the trained model weights (`.pt` files), `history.json`, and plots BEFORE terminating the pod. Weights are irreplaceable — retraining wastes hours and money. Use `git push` from the pod, then `git pull` locally, or `scp` the files directly.
2. **Update `PROGRESS.md`** — append a row to the progress log table. Include: date, what was done, baseline, measured result, and notes with file paths / report links. This is the breadcrumb trail for the next agent. No exceptions.
3. **Git commit and push** — stage the new/changed files and push so nothing is lost:
   ```bash
   git add -A alphago/ && git commit -m "<short description>" && git push
   ```

These steps apply to implementations, experiments, and hypothesis results alike. If you skip them, the next session starts blind and work may be lost. **NEVER terminate a pod without first saving weights locally.**

## Reading Guide

Read the source in this order:

1. **`utils/config.py`** — Your control panel. All parameters, documented.
2. **`games/base_game.py`** → **`games/tictactoe.py`** — The game interface and a concrete implementation.
3. **`mcts/node.py`** → **`mcts/search.py`** — The MCTS tree and search algorithm. This is the algorithmic core.
4. **`neural_net/base_net.py`** → **`neural_net/simple_net.py`** — The dual-headed network.
5. **`training/self_play.py`** → **`training/trainer.py`** → **`training/arena.py`** → **`training/pipeline.py`** — The training loop.
6. **`scripts/train.py`** — Entry point.

---

## Progress Log

**See [`PROGRESS.md`](PROGRESS.md)** for the full log of every implementation, experiment, and hypothesis with measured results. Read it first to understand where the project stands.

## Baselines

Reference models with consistent params for cross-game comparison. See `baselines/README.md` for full details.

```
baselines/<game>/best.pt           # weights
baselines/<game>/history.json      # metrics
baselines/<game>/training_curves.png
```

Shared config: 4x128 MLP, 50 sims, c_puct=1.0, lr=0.001, 25 iters, 100 games/iter.

| Game | vs Random | Time | Key Observation |
|------|-----------|------|-----------------|
| Tic-Tac-Toe | 95-100% | 3.5m | Converges iter 1. Arena = all draws. Game is "solved". |
| Connect Four | 100% | 7.8m | Takes ~6 iters. Arena stays competitive. Deeper search (5.6 vs 4.4). |
| Othello (6x6) | 60% (74% peak) | 16.7m | MLP struggles with spatial patterns. Only 5/25 models accepted. CNN needed. |

### Key Papers
- Coulom 2006 — [Efficient Selectivity and Backup Operators in MCTS](https://hal.science/hal-00116992/) — coined MCTS
- Kocsis & Szepesvari 2006 — [Bandit Based Monte-Carlo Planning](https://link.springer.com/chapter/10.1007/11871842_29) — UCT
- Silver et al. 2016 — [Mastering Go with Deep NN and Tree Search](https://www.nature.com/articles/nature16961) — AlphaGo
- Silver et al. 2017 — [Mastering Go without Human Knowledge](https://www.nature.com/articles/nature24270) — AlphaGo Zero
- Silver et al. 2018 — [A General RL Algorithm (chess, shogi, Go)](https://arxiv.org/abs/1712.01815) — AlphaZero
- Wu 2019 — [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565) — KataGo
- Schrittwieser et al. 2020 — [Planning with a Learned Model](https://www.nature.com/articles/s41586-020-03051-4) — MuZero
- Danihelka et al. 2022 — [Policy Improvement by Planning with Gumbel](https://arxiv.org/abs/2104.06303) — Gumbel AlphaZero
