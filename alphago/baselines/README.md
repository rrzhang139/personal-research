# Baselines

Reference models for each game, trained with consistent parameters for fair comparison.

## Shared Parameters

All baselines use the same core config so results are directly comparable:

```
Network:          4x128 MLP (4 hidden layers, 128 units each)
MCTS sims:        50
c_puct:           1.0
Dirichlet:        alpha=0.3, epsilon=0.25
LR:               0.001 (Adam)
Batch size:       64
Epochs/iter:      10
Iterations:       25
Games/iter:       100
Arena games:      40
Update threshold: 0.55
Buffer size:      50,000
Seed:             42
```

## Results

| Game | vs Random | Best vs Random | Final Loss | Time | Notes |
|------|-----------|----------------|------------|------|-------|
| **Tic-Tac-Toe** | 95% | 100% | 1.20 | 3.5m | Converges iter 1. Arena = all draws after iter 1 (model is already strong). |
| **Connect Four** | 100% | 100% | 1.87 | 7.8m | Takes ~6 iters to stabilize. Arena stays competitive throughout (harder game). |

## How to Reproduce

```bash
# Tic-tac-toe
python scripts/train.py --game tictactoe --num-simulations 50

# Connect Four
python scripts/train.py --game connect4 --num-simulations 50
```

## How to Load a Baseline

```python
from alpha_go.games import get_game
from alpha_go.neural_net.simple_net import SimpleNetWrapper
from alpha_go.utils.config import NetworkConfig

game = get_game('connect4')
model = SimpleNetWrapper(
    board_size=game.get_board_size(),
    action_size=game.get_action_size(),
    config=NetworkConfig(hidden_size=128, num_layers=4),
)
model.load('baselines/connect4/best.pt')
```

## Files per Game

```
baselines/<game>/
├── best.pt              # Model weights (best accepted model)
├── history.json         # Full training metrics per iteration
└── training_curves.png  # 6-panel training plot
```
