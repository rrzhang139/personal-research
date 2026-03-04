#!/usr/bin/env python3
"""Experiment: Reference architecture (alpha-zero-general) on Othello 6x6.

Previous attempt with 25 sims + nn_batch_size=32 failed: all sims consumed in
one batch → depth=1 tree → no real MCTS search. Research shows 6x6 Othello Elo
plateaus at ~200 sims (AlphaDDA paper). With 200 sims and batch=32, we get
~6 batches per move → proper tree depth (5-6 levels) + efficient GPU utilization.

Config:
- Network: OthelloNNet (plain CNN, 512 filters, 0.3 dropout)
- Buffer: window strategy, keep last 20 iterations
- 200 sims, 80 iters, 100 games/iter, 10 epochs, lr=0.001
- nn_batch_size=32, num_workers=1, update_threshold=0.55
"""

import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games import get_game
from alpha_go.neural_net import create_model
from alpha_go.training.pipeline import run_pipeline
from alpha_go.utils.config import (
    AlphaZeroConfig, ArenaConfig, MCTSConfig, NetworkConfig, TrainingConfig,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXPERIMENT_DIR, 'data')
FIG_DIR = os.path.join(EXPERIMENT_DIR, 'figures')


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    config = AlphaZeroConfig(
        mcts=MCTSConfig(
            num_simulations=200,
            nn_batch_size=32,
        ),
        network=NetworkConfig(
            network_type="othellonet",
            num_filters=512,
            dropout=0.3,
        ),
        training=TrainingConfig(
            num_iterations=80,
            games_per_iteration=100,
            epochs_per_iteration=10,
            batch_size=64,
            lr=0.001,
            buffer_strategy="window",
            buffer_window=20,
            checkpoint_dir=os.path.join(DATA_DIR, 'othellonet'),
        ),
        arena=ArenaConfig(
            arena_games=40,
            update_threshold=0.55,
        ),
        game='othello',
        seed=42,
        num_workers=1,
    )

    # Save config
    config_path = os.path.join(EXPERIMENT_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'experiment': 'Reference architecture on Othello 6x6',
            'network': {
                'type': 'othellonet',
                'num_filters': 512,
                'dropout': 0.3,
            },
            'buffer': {
                'strategy': 'window',
                'window': 20,
            },
            'mcts': {
                'num_simulations': 200,
                'nn_batch_size': 32,
            },
            'training': {
                'num_iterations': 80,
                'games_per_iteration': 100,
                'epochs_per_iteration': 10,
                'batch_size': 64,
                'lr': 0.001,
            },
            'arena': {
                'arena_games': 40,
                'update_threshold': 0.55,
            },
        }, f, indent=2)

    game = get_game('othello')
    model = create_model(game, config.network, lr=config.training.lr)

    print(f"\n{'#'*70}")
    print(f"#  OthelloNNet (reference arch) — 512f, dropout=0.3, window buffer")
    print(f"#  80 iterations, 200 sims, 100 games/iter, nn_batch=32")
    print(f"{'#'*70}")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    elapsed = time.time() - t0

    history['wall_time'] = elapsed

    # Save results
    results_path = os.path.join(DATA_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # Summary
    final_wr = history['vs_random_win_rate'][-1]
    best_wr = max(history['vs_random_win_rate'])
    accepted = sum(history['model_accepted'])
    total = len(history['model_accepted'])
    time_str = f"{elapsed:.0f}s" if elapsed < 120 else f"{elapsed/60:.1f}m"

    print(f"\n{'='*60}")
    print(f"RESULT: {final_wr:.0%} final vs random, {best_wr:.0%} peak")
    print(f"Models accepted: {accepted}/{total}")
    print(f"Wall time: {time_str}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
