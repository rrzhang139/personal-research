#!/usr/bin/env python3
"""Go 9x9 v2 training — proven arch + KataGo MCTS params + batch=64.

Key changes from v1 (20260311_go9_optimized):
  - nn_batch_size=64 (1.66x faster on CUDA)
  - KataGo MCTS params: dirichlet_alpha=0.12, c_puct=1.0, temp_decay, FPU
  - cosine LR schedule
  - 200 iterations × 100 games (20,000 total games)
  - Standard CNN arch (with BN, no SE) — SE+no-BN caused exploding loss (14M)
  - Optimized self-play: bytearray flood fill, fast suicide check (2.69x faster)

Warm-start from playout_cap best.pt.

Expected time on RTX 4090: ~13 hours (~232s/iter self-play + training)
Expected cost: ~$2.60 at $0.20/hr
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

# Try warm-starting from v1, fall back to playout_cap
WARM_STARTS = [
    os.path.join(EXPERIMENT_DIR, '..', '20260311_go9_optimized', 'data', 'checkpoints', 'best.pt'),
    os.path.join(EXPERIMENT_DIR, '..', '20260310_go9_playout_cap', 'data', 'checkpoints', 'best.pt'),
]


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    game = Go(size=9)

    config = AlphaZeroConfig(
        game="go9",
        seed=42,
        mcts=MCTSConfig(
            num_simulations=200,
            c_puct=1.0,  # KataGo default
            dirichlet_alpha=0.12,  # 10/81 for 9x9 (0.03 is for 19x19)
            dirichlet_epsilon=0.25,
            temp_threshold=30,
            temp_decay_halflife=19,  # KataGo-style exponential decay
            nn_batch_size=64,  # 1.66x faster on CUDA
            playout_cap_prob=0.125,
            playout_cap_cheap_fraction=0.15,
            fpu_reduction=0.2,  # KataGo fpuReductionMax
            root_fpu_reduction=0.1,  # KataGo rootFpuReductionMax
        ),
        network=NetworkConfig(
            network_type="cnn",
            num_filters=128,
            num_res_blocks=4,
        ),
        training=TrainingConfig(
            lr=0.002,
            weight_decay=1e-4,
            batch_size=256,
            epochs_per_iteration=10,
            num_iterations=200,
            games_per_iteration=100,
            max_buffer_size=200000,
            buffer_strategy="fifo",
            checkpoint_dir=CHECKPOINT_DIR,
            lr_schedule="cosine",
            lr_min=1e-5,
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

    # Create model (standard arch, same as v1)
    model = create_model(game, config.network, lr=config.training.lr, weight_decay=config.training.weight_decay)
    print(f"Device: {model.net.device}")

    # Warm-start from previous best
    for warm_path in WARM_STARTS:
        if os.path.exists(warm_path):
            model.load(warm_path)
            print(f"Warm-started from {warm_path}")
            break
    else:
        print("WARNING: No warm-start weights found, training from scratch")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    total = time.time() - t0

    print(f"\nTotal training time: {total/60:.1f}m ({total/3600:.1f}h)")
    print(f"Best model: {os.path.join(CHECKPOINT_DIR, 'best.pt')}")


if __name__ == "__main__":
    main()
