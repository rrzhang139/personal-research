# Roadmap: From Tic-Tac-Toe to Go

A progression of increasingly complex games, each introducing new challenges that motivate specific algorithmic improvements. The goal: by the time you reach Go, every technique in the system was added because a simpler game *needed* it.

---

## Phase 1: Tic-Tac-Toe (DONE)

**Game**: 3x3 board, 9 actions, 5-9 moves per game, theoretically a draw.

**What it teaches**: The core AlphaZero loop works. MCTS + neural net + self-play converges. You understand what each parameter does.

**Experiments completed**:
- [x] Baseline training — 97% vs random in 3 min
- [x] `num_simulations` sweep — discovered the 1→5 sim cliff (61%→90%), knee at 10-25 sims
- [ ] `c_puct` sweep — exploration vs exploitation
- [ ] Dirichlet noise ablation — does root noise matter here?
- [ ] Network capacity sweep — how small can the net be?

**Graduation**: Model draws consistently in self-play. You can predict parameter effects before running.

---

## Phase 2: Connect Four (NEXT)

**Game**: 6x7 board, 7 actions (column drops), up to 42 moves, **first player wins** with perfect play.

**Why this game**: First game where the MLP network + 25 sims might not be enough. The game has real depth — there are forced wins 10+ moves out that require actual search to find. But the action space is small (7) so training is still fast.

**New challenges**:
| Challenge | Why it's new | What you'll add |
|-----------|-------------|-----------------|
| Deeper games (42 moves vs 9) | More positions to learn, longer credit assignment | May need more `games_per_iteration` |
| Vertical structure | Pieces drop — column 3 row 0 is different from column 3 row 5 | Network must learn gravity |
| First-player advantage | Unlike tic-tac-toe's draw, P1 wins with perfect play | Self-play stats should show P1 winning more |
| Harder evaluation | Positions that look even can be forced wins | Value head has a harder job, need more sims |

**Experiments to run**:
1. Baseline MLP training — does the existing 4x128 MLP work, or does it need more capacity?
2. `num_simulations` sweep on Connect Four — where's the knee now? (expect higher than tic-tac-toe)
3. Does training converge to first-player advantage? (P1 win rate in self-play should be >50%)
4. MLP vs CNN — does a 2D convolutional network help for the 6x7 board?

**Implementation needed**:
- `games/connect4.py` — drop pieces, check four-in-a-row, vertical/horizontal/diagonal

**Graduation**: Beats random >99%. Self-play shows clear P1 advantage. You've identified whether MLP or CNN is needed.

---

## Phase 3: Othello (6x6, then 8x8)

**Game**: Place pieces to flip opponent's pieces. Dramatic lead changes. 8x8 standard, ~60 moves per game.

**Why this game**: The 2D spatial structure *definitely* matters. Corner and edge control is the dominant strategy. An MLP treating the board as a flat vector will miss this entirely. This is where you'll be forced to add a CNN/ResNet.

**New challenges**:
| Challenge | Why it's new | What you'll add |
|-----------|-------------|-----------------|
| Spatial patterns matter | Corners are permanently captured, edges are stable | **Residual CNN** — the key architectural upgrade |
| Variable legal moves | Some positions have 1 legal move, others have 20 | Network must handle varying action masks |
| Long training | 60-move games, larger board, more data needed | **LR schedule** + **weight decay** |
| Lead reversals | You can be "winning" and lose everything in 3 moves | Value head must learn non-obvious evaluations |

**Technique unlocks**:
- **Residual CNN** (He et al. 2015) — shared convolutional trunk with skip connections
- **Weight decay / L2 regularization** — prevents overfitting on longer training
- **Learning rate schedule** — cosine or step decay

**Start with 6x6 Othello** as a stepping stone (36 squares vs 64, faster iteration). If the CNN works on 6x6, scale to 8x8.

**Graduation**: Beats random >99.5% on 8x8. ResNet architecture working. You can see the model learning corner/edge strategy.

---

## Phase 4: Go (5x5 → 9x9)

**Game**: Surround territory on a grid. Captures, ko rule, passing, complex scoring.

**Why this game**: This is *the* game. Every technique in the AlphaGo lineage was developed to solve Go. The branching factor (~250 on 19x19) means search alone can't work — the neural network must provide strong priors.

**Progressive board sizes**:

| Board | Legal moves | Rough complexity | Training time (est.) |
|-------|------------|------------------|---------------------|
| 5x5   | ~25/move   | Slightly harder than Othello | Hours on 1 GPU |
| 7x7   | ~49/move   | Non-trivial strategy emerges | Hours on 1 GPU |
| 9x9   | ~81/move   | Standard "small Go", serious game | ~1 day on 1 GPU |

**New challenges**:
| Challenge | Why it's new | What you'll add |
|-----------|-------------|-----------------|
| Ko rule | Position repetition is illegal — need game history | State includes recent board positions |
| Pass moves | Players can pass; two passes end the game | Action space = board_size² + 1 |
| Scoring | Area or territory counting at game end | Tromp-Taylor scoring (simplest) |
| Huge action space | 81+ moves per position on 9x9 | Network priors become critical |
| Training efficiency | Can't afford to waste compute | **KataGo tricks**: playout cap randomization, auxiliary targets |

**Technique unlocks**:
- **KataGo playout cap randomization** — vary sims per move, weight training inversely (Wu 2019)
- **Auxiliary training targets** — ownership prediction, score prediction (denser signal than win/loss)
- **Deeper ResNet** — 10-20 blocks for 9x9

**Graduation**: 9x9 agent beats GNU Go (~12 kyu amateur). You understand every component deeply enough to explain it.

---

## Phase 5: Full 19x19 Go (aspirational)

Requires multi-GPU, days of training. Same algorithm, scaled up:
- 20-40 residual blocks
- 800+ MCTS simulations
- Parallelized self-play (multiple workers)
- Batched neural network evaluation

The code change from Phase 4 is small. The compute change is large.

---

## Alternative Track: Chess (branch from Phase 3)

If you prefer chess over Go at any point, it's a valid alternative path. Different challenges:
- No board symmetries (asymmetric position of kings)
- Complex move encoding (piece × source × destination → ~4672 possible moves)
- Draw detection (50-move rule, repetition, stalemate)
- Well-studied benchmarks (ELO rating, standard test suites)

Can branch off after Othello — the ResNet architecture transfers directly.

---

## Technique Timeline

When each technique becomes necessary, mapped to the phase that motivates it:

```
Phase 1 (TicTacToe)     Phase 2 (Connect4)     Phase 3 (Othello)     Phase 4 (Go)
─────────────────────    ──────────────────     ─────────────────     ────────────
Core AlphaZero loop      Deeper search (sims)   Residual CNN          Ko handling
PUCT + Dirichlet noise   First-player bias      Weight decay          Pass moves
MLP dual-head net        CNN (maybe)             LR schedule           Scoring
Self-play + arena                                Board symmetry aug    KataGo tricks
Config system                                                          Playout cap rand
Diagnostics + W&B                                                      Auxiliary targets
                                                                       Parallel self-play
```

Each row was added because a game required it, not because it seemed like a good idea in advance.
