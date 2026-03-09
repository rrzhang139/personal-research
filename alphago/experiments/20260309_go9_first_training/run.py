#!/usr/bin/env python3
"""First Go 9x9 training run.

Config rationale:
- CNN 64f 4 res blocks: proven architecture, moderate size for 17-channel input
- 50 sims: reasonable for 9x9 (82 action space, similar to Othello 10x10 = 101)
- nn-batch-size 8: GPU batching for MCTS
- 50 games/iter: enough training data per iteration
- 25 iters: standard for our experiments
- dirichlet_alpha 0.03: Go standard (inversely scaled with action space: 10/82 ≈ 0.12, but 0.03 is AlphaGo convention)
- temp_threshold 30: Go games are longer, explore first 30 moves

Estimated cost: ~$0.50-1.50 on RTX A4000 ($0.17/hr × 3-8 hrs)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games.go import Go
from alpha_go.neural_net import create_model
from alpha_go.training.pipeline import run_pipeline
from alpha_go.utils.config import (
    AlphaZeroConfig, ArenaConfig, MCTSConfig, NetworkConfig, TrainingConfig,
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

config = AlphaZeroConfig(
    mcts=MCTSConfig(
        num_simulations=50,
        c_puct=1.0,
        dirichlet_alpha=0.03,      # Go standard (large action space)
        dirichlet_epsilon=0.25,
        temp_threshold=30,          # Go games are longer
        nn_batch_size=8,            # GPU batching
    ),
    network=NetworkConfig(
        network_type='cnn',
        num_filters=64,
        num_res_blocks=4,
        dropout=0.0,
    ),
    training=TrainingConfig(
        lr=0.001,
        batch_size=64,
        epochs_per_iteration=10,
        max_buffer_size=100000,     # Larger buffer for Go (longer games = more examples)
        buffer_strategy='fifo',     # FIFO to avoid OOM (learned from Othello)
        num_iterations=25,
        games_per_iteration=50,
        checkpoint_dir=os.path.join(DATA_DIR, 'checkpoints'),
    ),
    arena=ArenaConfig(
        arena_games=20,             # Fewer arena games (Go games are slow)
        update_threshold=0.55,
    ),
    game='go',
    seed=42,
    num_workers=1,                  # Sequential (GPU batching is our parallelism)
    use_wandb=True,
    wandb_project='alphazero',
)

# Save config
with open(os.path.join(EXP_DIR, 'config.json'), 'w') as f:
    json.dump({
        'game': 'go9',
        'board_size': 9,
        'mcts': {k: getattr(config.mcts, k) for k in config.mcts.__dataclass_fields__},
        'network': {k: getattr(config.network, k) for k in config.network.__dataclass_fields__},
        'training': {k: getattr(config.training, k) for k in config.training.__dataclass_fields__},
        'arena': {k: getattr(config.arena, k) for k in config.arena.__dataclass_fields__},
    }, f, indent=2, default=str)

game = Go(size=9)
model = create_model(game, config.network, lr=config.training.lr)

print(f"\n=== Go 9x9 First Training ===")
print(f"Board: {game.get_board_size()} ({game.get_board_shape()})")
print(f"Actions: {game.get_action_size()}")
print(f"Network: CNN {config.network.num_filters}f {config.network.num_res_blocks} res blocks")
print(f"MCTS: {config.mcts.num_simulations} sims, batch={config.mcts.nn_batch_size}")
print(f"Training: {config.training.num_iterations} iters × {config.training.games_per_iteration} games")
print(f"GPU: {__import__('torch').cuda.get_device_name(0) if __import__('torch').cuda.is_available() else 'CPU'}")
print(f"Data dir: {DATA_DIR}")

start = time.time()
history = run_pipeline(game, model, config)
elapsed = time.time() - start

# Save history
with open(os.path.join(DATA_DIR, 'history.json'), 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n=== Done in {elapsed/60:.1f} min ===")
print(f"History saved to {DATA_DIR}/history.json")

# Upload best model to wandb
try:
    import wandb
    if wandb.run:
        best_path = os.path.join(DATA_DIR, 'checkpoints', 'best.pt')
        if os.path.exists(best_path):
            artifact = wandb.Artifact('go9-cnn-first', type='model')
            artifact.add_file(best_path, name='best.pt')
            wandb.log_artifact(artifact)
            print(f"Uploaded best.pt to wandb artifact 'go9-cnn-first'")
        wandb.finish()
except Exception as e:
    print(f"W&B artifact upload failed: {e}")
