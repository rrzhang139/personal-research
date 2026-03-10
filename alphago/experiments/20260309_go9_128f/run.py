#!/usr/bin/env python3
"""Go 9x9 training run — 128 filter CNN.

Hypothesis: first run (64f) was capacity-limited. 128f doubles network
capacity with negligible MCTS slowdown (CPU-bound tree traversal dominates).
If this doesn't reach 100% vs random, the bottleneck is search depth (need more sims).

Config changes from first run:
- num_filters: 64 -> 128 (2x capacity)
- max_buffer_size: 100k -> 150k (bigger network benefits from more data)

Everything else identical to first run.

Estimated: ~7.5h on A4000, ~$1.28
"""

import json
import os
import subprocess
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
        dirichlet_alpha=0.03,
        dirichlet_epsilon=0.25,
        temp_threshold=30,
        nn_batch_size=8,
    ),
    network=NetworkConfig(
        network_type='cnn',
        num_filters=128,            # 2x first run
        num_res_blocks=4,
        dropout=0.0,
    ),
    training=TrainingConfig(
        lr=0.001,
        batch_size=64,
        epochs_per_iteration=10,
        max_buffer_size=150000,
        buffer_strategy='fifo',
        num_iterations=25,
        games_per_iteration=50,
        checkpoint_dir=os.path.join(DATA_DIR, 'checkpoints'),
    ),
    arena=ArenaConfig(
        arena_games=20,
        update_threshold=0.55,
    ),
    game='go',
    seed=42,
    num_workers=1,
    use_wandb=True,
    wandb_project='alphazero',
)

# Save config
with open(os.path.join(EXP_DIR, 'config.json'), 'w') as f:
    json.dump({
        'game': 'go9',
        'board_size': 9,
        'hypothesis': '128f vs 64f: does doubling network capacity reach 100% vs random?',
        'mcts': {k: getattr(config.mcts, k) for k in config.mcts.__dataclass_fields__},
        'network': {k: getattr(config.network, k) for k in config.network.__dataclass_fields__},
        'training': {k: getattr(config.training, k) for k in config.training.__dataclass_fields__},
        'arena': {k: getattr(config.arena, k) for k in config.arena.__dataclass_fields__},
    }, f, indent=2, default=str)

game = Go(size=9)
model = create_model(game, config.network, lr=config.training.lr)

# Count parameters
total_params = sum(p.numel() for p in model.net.parameters())
print(f"\n=== Go 9x9 — 128f CNN ===")
print(f"Board: {game.get_board_size()} ({game.get_board_shape()})")
print(f"Actions: {game.get_action_size()}")
print(f"Network: CNN {config.network.num_filters}f {config.network.num_res_blocks} res blocks ({total_params:,} params)")
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
            artifact = wandb.Artifact('go9-cnn-128f', type='model')
            artifact.add_file(best_path, name='best.pt')
            wandb.log_artifact(artifact)
            print(f"Uploaded best.pt to wandb artifact 'go9-cnn-128f'")
        wandb.finish()
except Exception as e:
    print(f"W&B artifact upload failed: {e}")

# Auto-push results to git
print("\n=== Auto-pushing results ===")
try:
    repo_root = os.path.join(EXP_DIR, '..', '..', '..')
    subprocess.run(['git', 'add', '-f', f'alphago/experiments/20260309_go9_128f/'],
                   cwd=repo_root, check=True)
    subprocess.run(['git', 'commit', '-m',
                    f'Go 9x9 128f results: {history[-1].get("vs_random", "?")}% final vs random, {elapsed/60:.0f}m\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>'],
                   cwd=repo_root, check=True)
    subprocess.run(['git', 'push'], cwd=repo_root, check=True, capture_output=True)
    print("Results pushed to git!")
except Exception as e:
    print(f"Git push failed: {e}")
    print("Manual push needed: cd /workspace/code/personal-research && git add -f alphago/experiments/20260309_go9_128f/ && git commit -m 'results' && git push")
