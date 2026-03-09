# From Othello to Go: What Changes and Why It's Hard

## Where We Are

We've validated the full AlphaZero pipeline on Othello at two scales:

| Board | Actions | Moves/game | Branching factor | Best architecture | Training time |
|-------|---------|------------|------------------|-------------------|---------------|
| 6x6 Othello | 37 | ~30 | ~5-8 | MLP (CNN catches up at 200 sims) | 1h (A4000) |
| 10x10 Othello | 101 | ~96 | ~10-15 | CNN (40-0 h2h, wins with 0 sims) | 7h (A100) |

Go is the game AlphaZero was built for. Here's why it's a different beast.

---

## The Game

Go is played on a grid of intersections (not squares). Two players place black and white stones. Stones with no empty adjacent intersections ("liberties") are captured and removed. Game ends when both players pass. Winner has more territory.

That's it. Four rules, infinite depth.

## Observation Space

**Othello**: single channel, 1 plane of {-1, 0, 1}.

**Go**: multi-channel input. AlphaGo Zero used **17 planes**:

```
Planes 1-8:   Current player's stones for the last 8 board states
Planes 9-16:  Opponent's stones for the last 8 board states
Plane 17:     Color to play (all 1s if black, all 0s if white)
```

Why 8 history planes? **Ko detection.** Go has a rule that you cannot recreate a previous board position (to prevent infinite loops). The network needs to see recent history to avoid illegal moves.

For our implementation, we can start simpler:

```
Minimum viable input (3 planes):
  Plane 1: Current player's stones
  Plane 2: Opponent's stones
  Plane 3: Ko point (single illegal intersection, if any)

Shape: (C, N, N) where C=3-17, N=board size
Flat size for MLP: C × N × N (e.g., 3 × 9 × 9 = 243)
```

**vs Othello**: Othello is `(1, N, N)` — one plane. Go needs at minimum 3 planes, practically 8-17. This means the CNN input channels change from 1 to C.

## Action Space

```
Actions: N × N + 1 (place on any intersection, or pass)

9x9:   82 actions
13x13: 170 actions
19x19: 362 actions
```

**vs Othello**: Same structure (board positions + pass). Othello restricts legal moves to flanking positions (~5-15 at any time). Go allows placement on ANY empty intersection with a liberty — typically **30-80 legal moves** at any point. This is the big difference.

## Computational Complexity Comparison

| Dimension | Othello 10x10 | Go 9x9 | Go 19x19 |
|-----------|--------------|--------|----------|
| Board positions | ~10^15 | ~10^38 | ~10^170 |
| Actions per turn | 10-15 | 30-50 | 50-200 |
| Game length | ~96 moves | ~50 moves | ~250 moves |
| Game tree size | ~10^30 | ~10^70 | ~10^360 |
| Legal positions | ~10^12 | ~10^38 | ~10^170 |

For reference: chess is ~10^120 game tree complexity. Go 19x19 at 10^360 is why brute-force search is impossible and why AlphaZero was a breakthrough.

**9x9 Go is the right starting point.** It has the full rule complexity but a manageable board. KataGo and most research starts here. The game tree is still ~10^40x larger than Othello 10x10, but our pipeline proved it works on spatial games with CNN.

## What Makes Go Hard for AlphaZero (vs Othello)

### 1. Higher branching factor → slower MCTS convergence

Othello: ~10 legal moves per turn → 100 MCTS sims explores each move ~10 times.
Go 9x9: ~40 legal moves per turn → 100 sims explores each move ~2.5 times.

**Impact**: Need more sims per move. AlphaZero used **800 sims** for chess/Go vs our 100 for Othello. At minimum, 200-400 sims for 9x9 Go.

### 2. Ko rule adds state complexity

Othello has no cycles — pieces only flip, never get removed. Go has ko: a move can recreate a board position from 1-2 moves ago. The ko rule forbids this.

**Implementation**: Must track at least the previous board state. Superko (the full rule) forbids recreating ANY prior position in the game — requires tracking the full game history.

**Impact on network**: The input must include history planes so the network can "see" what positions are forbidden. Without this, the network will repeatedly suggest illegal ko moves.

### 3. Game-end detection is non-trivial

Othello ends when neither player can move — clear and algorithmic.

Go ends when **both players agree** to pass. Then:
1. Remove dead stones (stones that would be captured if the game continued)
2. Count territory (empty intersections surrounded by one color)
3. Add captured stones
4. Apply komi (compensation for going second, typically 6.5 or 7.5 points)

**For self-play**: We can use Tromp-Taylor scoring (Chinese rules simplified) — count all stones + surrounded empty points. No need for manual dead stone agreement. This is what AlphaGo Zero used.

### 4. Suicide rule

Placing a stone that immediately has zero liberties is **illegal** in most rulesets (Chinese, Japanese, AGA). Some rulesets allow it (New Zealand). We should forbid it — it simplifies the legal move mask.

### 5. Value landscape is smoother but harder

Othello: piece count difference is a reasonable heuristic. Win/loss is clear.

Go: a 1-point win and a 50-point win are both "+1" to the value head, but the positions look very different. The value landscape has many local optima (capturing a group looks winning but may lose territory elsewhere). The network must learn whole-board evaluation — exactly what CNNs are good at.

## Implementation Plan

### New file: `games/go.py`

Must implement the `Game` interface:

```python
class Go(Game):
    def __init__(self, size=9, komi=7.5, history_planes=8):
        ...

    # State: flat array of size C * N * N (multi-channel)
    # Or: keep state as (N*N,) stones + separate ko tracking

    def get_initial_state(self) -> np.ndarray
    def get_next_state(self, state, action, player) -> np.ndarray
    def get_valid_moves(self, state, player) -> np.ndarray   # must check suicide + ko
    def check_terminal(self, state, action, player) -> tuple  # both pass → score
    def get_board_size(self) -> int       # C * N * N or N * N
    def get_board_shape(self) -> tuple    # (C, N, N) — multi-channel!
    def get_action_size(self) -> int      # N * N + 1
    def get_canonical_state(self, state, player) -> np.ndarray
    def get_symmetries(self, state, pi) -> list  # 8 symmetries (same as Othello)
```

### Key design decisions

1. **State representation**: Store as `(N*N,)` flat array for stones + track ko point separately in the game object? Or store as multi-channel `(C*N*N,)` with history baked in?
   - AlphaGo Zero bakes history into state → simpler for the network
   - But makes `get_next_state` more complex (must shift history planes)
   - Recommendation: keep state as stones `(N*N,)` + pass a history stack through the pipeline. Convert to multi-channel only for network input.

2. **Liberty counting**: Most expensive operation. Each `get_valid_moves` call must check every empty intersection for suicide. Use flood-fill or union-find for connected groups.
   - **Union-find** is O(α(n)) per operation — fast for incremental updates
   - Pre-compute liberties after each move, not on-demand

3. **Ko tracking**: Store the single forbidden point (simple ko). Superko can be added later if needed.

4. **Scoring**: Tromp-Taylor (Chinese area scoring). Count stones + territory via flood-fill. Add komi for white.

5. **Board shape for CNN**: `get_board_shape()` returns `(C, N, N)` instead of `(N, N)`. The CNN input layer needs `in_channels=C` instead of `in_channels=1`.

### Changes to existing code

| File | Change |
|------|--------|
| `neural_net/__init__.py` | Handle `board_shape` with >2 dims (C, N, N) |
| `neural_net/othello_net.py` | Change `nn.Conv2d(1, nf, ...)` → `nn.Conv2d(C, nf, ...)` |
| `neural_net/simple_net.py` | `board_size` becomes `C * N * N` for multi-channel input |
| `games/__init__.py` | Register `go9`, `go13`, `go19` |
| `mcts/search.py` | No changes (already game-agnostic) |
| `training/` | No changes (already game-agnostic) |

### Estimated complexity

| Component | Difficulty | Why |
|-----------|-----------|-----|
| Stone placement + capture | Medium | Flood-fill for group detection, liberty counting |
| Ko detection | Easy | Track previous board hash, forbid repeat |
| Superko | Medium | Track full game history, hash comparison |
| Scoring (Tromp-Taylor) | Easy | Flood-fill for territory |
| Valid move generation | Medium | Must check suicide for every empty point |
| Multi-channel state | Easy | Reshape + stack history planes |
| Tests | Medium | Many edge cases (snapback, ko fights, seki, large captures) |

### What to watch for during training

1. **Pass timing**: The network must learn when to pass. Too early = loses territory. Too late = wastes moves. This is hard — Othello forces pass only when no moves exist.
2. **Ko fights**: Complex tactical sequences that require reading ahead. The network needs enough history planes to understand ko.
3. **Corner play**: First moves in corners are critical in Go. The network should learn standard openings quickly — if it doesn't, the priors are wrong.
4. **Training time**: Expect 10-50x more training than Othello 10x10 for 9x9 Go. Budget for multi-day runs.
5. **Evaluation**: vs random is even less meaningful in Go (random play is terrible). Need a fixed reference opponent or ELO tracking.
