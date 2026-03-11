#!/usr/bin/env python3
"""Go 9x9 training with all optimizations — aiming to beat GnuGo Level 1+.

Hypothesis: Lazy expansion + FPU reduction + weight decay + more training
iterations (50 × 100 games = 5000 new games, 6200 total) will produce a model
strong enough to beat GnuGo at low levels.

Changes from previous best (playout_cap, 12 iters × 100 games):
  - FPU reduction = 0.25 (focuses search on promising moves)
  - Weight decay = 1e-4 (L2 regularization)
  - 50 iterations (vs 12) — 5x more training
  - All code optimizations: lazy expansion, suicide fast-path, etc.
  - eval_games=0 (skip vs_random for speed, evaluate with gnugo after)

Warm-start from playout_cap best.pt (100% vs random, 0% vs gnugo L1).

Expected time on RTX A4000: ~15-20 hours
Expected cost: ~$2.50-3.50 at $0.17/hr
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games.go import Go
from alpha_go.neural_net import create_model
from alpha_go.utils.config import (
    AlphaZeroConfig, MCTSConfig, NetworkConfig, TrainingConfig, ArenaConfig,
)
from alpha_go.training.pipeline import run_pipeline

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXPERIMENT_DIR, 'data')
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')
WARM_START = os.path.join(EXPERIMENT_DIR, '..', '20260310_go9_playout_cap', 'data', 'checkpoints', 'best.pt')


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    game = Go(size=9)

    config = AlphaZeroConfig(
        game="go9",
        seed=42,
        mcts=MCTSConfig(
            num_simulations=200,
            c_puct=1.5,
            dirichlet_alpha=0.03,
            dirichlet_epsilon=0.25,
            temp_threshold=20,
            nn_batch_size=8,
            playout_cap_prob=0.125,
            playout_cap_cheap_fraction=0.15,
            fpu_reduction=0.25,
        ),
        network=NetworkConfig(
            network_type="cnn",
            num_filters=128,
            num_res_blocks=4,
        ),
        training=TrainingConfig(
            lr=0.001,
            weight_decay=1e-4,
            batch_size=256,
            epochs_per_iteration=10,
            num_iterations=50,
            games_per_iteration=100,
            max_buffer_size=200000,
            buffer_strategy="fifo",
            checkpoint_dir=CHECKPOINT_DIR,
        ),
        arena=ArenaConfig(arena_games=0, eval_games=10),
        use_wandb=True,
        wandb_project="alphazero",
    )

    # Save config
    config_path = os.path.join(EXPERIMENT_DIR, 'config.json')
    from dataclasses import asdict
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Config saved to {config_path}")

    # Create model
    model = create_model(game, config.network, lr=config.training.lr, weight_decay=config.training.weight_decay)
    print(f"Device: {model.net.device}")

    # Warm-start
    warm_path = WARM_START
    if os.path.exists(warm_path):
        model.load(warm_path)
        print(f"Warm-started from {warm_path}")
    else:
        print(f"WARNING: Warm-start weights not found at {warm_path}, training from scratch")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    total = time.time() - t0

    print(f"\nTotal training time: {total/60:.1f}m ({total/3600:.1f}h)")
    print(f"Best model: {os.path.join(CHECKPOINT_DIR, 'best.pt')}")


if __name__ == "__main__":
    main()
