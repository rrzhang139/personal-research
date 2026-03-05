# Baselines

Reference models for each game, trained with consistent parameters for fair comparison.

## Shared Parameters

All baselines use the same core training config so results are directly comparable:

```
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

| Game | Architecture | vs Random | Best vs Random | Time | Notes |
|------|-------------|-----------|----------------|------|-------|
| **Tic-Tac-Toe** | 4x128 MLP | 95% | 100% | 3.5m | Converges iter 1. Arena = all draws. |
| **Connect Four** | 4x128 MLP | 100% | 100% | 7.8m | ~6 iters to stabilize. |
| **Othello (6x6)** | 4x128 MLP | *pending re-run* | — | — | Re-run needed after winner-detection bugfix. |
| **Othello (6x6)** | OthelloNNet 512f | *pending* | — | — | Plain CNN + window buffer. |

> **Note:** Previous Othello MLP baseline (60% vs random) used broken winner detection.
> All Othello baselines are being re-generated with the bugfix. Run `baselines/othello/generate.py` to produce both.

## Othello: Multi-Architecture Baselines

Othello has spatial patterns that benefit from convolutional architectures. We maintain two baselines:

| Architecture | Description | Extra Config |
|-------------|-------------|-------------|
| **MLP** (4x128) | Shared baseline config, same as other games | — |
| **OthelloNNet** (512 filters) | Plain CNN with shrinking spatial dims, BN, dropout | `nn_batch_size=8`, `buffer_strategy=window`, `buffer_window=20`, `dropout=0.3` |

Generate both with:
```bash
python baselines/othello/generate.py                    # full run (25 iters each)
python baselines/othello/generate.py --num-iterations 2 # smoke test
python baselines/othello/generate.py --only othellonet  # run one architecture
```

## How to Reproduce

```bash
# Tic-tac-toe
python scripts/train.py --game tictactoe --num-simulations 50

# Connect Four
python scripts/train.py --game connect4 --num-simulations 50

# Othello (both architectures)
python baselines/othello/generate.py
```

## How to Load a Baseline

```python
from alpha_go.games import get_game
from alpha_go.neural_net import create_model
from alpha_go.utils.config import NetworkConfig

game = get_game('othello')

# MLP baseline
model = create_model(game, NetworkConfig(network_type='mlp', hidden_size=128, num_layers=4))
model.load('baselines/othello/mlp/best.pt')

# OthelloNNet baseline
model = create_model(game, NetworkConfig(network_type='othellonet', num_filters=512, dropout=0.3))
model.load('baselines/othello/othellonet/best.pt')
```

## Directory Structure

```
baselines/
├── README.md
├── tictactoe/               # Single arch (MLP)
│   ├── args.json
│   ├── best.pt
│   ├── history.json
│   └── training_curves.png
├── connect4/                 # Single arch (MLP)
│   ├── args.json
│   ├── best.pt
│   ├── history.json
│   └── training_curves.png
└── othello/                  # Multi-arch
    ├── generate.py           # Runs both baselines
    ├── comparison.png        # Side-by-side plot
    ├── mlp/
    │   ├── args.json
    │   ├── best.pt
    │   ├── history.json
    │   └── training_curves.png
    └── othellonet/
        ├── args.json
        ├── best.pt
        ├── history.json
        └── training_curves.png
```
