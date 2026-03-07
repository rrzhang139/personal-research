#!/usr/bin/env python3
"""Experiment: OthelloNNet CNN on 10x10 Othello (CNN only, full training).

Same config as 20260305_othello_10x10_cnn_vs_mlp but CNN only.
Previous run was killed at 16/25 iters (98% vs random) and weights were lost.
This run saves weights via Git LFS.

Usage:
    python experiments/20260306_othello_10x10_cnn/run.py          # full (GPU)
    python experiments/20260306_othello_10x10_cnn/run.py --quick  # smoke test
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games.othello import Othello
from alpha_go.neural_net import create_model
from alpha_go.training.arena import play_vs_random
from alpha_go.training.pipeline import run_pipeline
from alpha_go.utils.config import (
    AlphaZeroConfig, ArenaConfig, MCTSConfig, NetworkConfig, TrainingConfig,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXPERIMENT_DIR, 'data')
FIG_DIR = os.path.join(EXPERIMENT_DIR, 'figures')

BOARD_SIZE = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test (2 iters, 10 games, 25 sims)')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    if args.quick:
        n_iters, n_games, n_sims = 2, 10, 25
    else:
        n_iters, n_games, n_sims = 25, 100, 100

    # Save config
    with open(os.path.join(EXPERIMENT_DIR, 'config.json'), 'w') as f:
        json.dump({
            'experiment': 'OthelloNNet CNN on 10x10 Othello',
            'board_size': BOARD_SIZE,
            'num_simulations': n_sims,
            'num_iterations': n_iters,
            'games_per_iteration': n_games,
            'architecture': {
                'network_type': 'othellonet',
                'num_filters': 512,
                'dropout': 0.3,
                'nn_batch_size': 8,
                'buffer_strategy': 'window',
                'buffer_window': 20,
            },
            'quick_mode': args.quick,
        }, f, indent=2)

    config = AlphaZeroConfig(
        mcts=MCTSConfig(num_simulations=n_sims, nn_batch_size=8),
        network=NetworkConfig(
            network_type='othellonet', num_filters=512, dropout=0.3,
        ),
        training=TrainingConfig(
            num_iterations=n_iters,
            games_per_iteration=n_games,
            
            checkpoint_dir=os.path.join(DATA_DIR, 'othellonet'),
        ),
        arena=ArenaConfig(arena_games=40, update_threshold=0.55),
        game='othello10', seed=42,
    )

    game = Othello(size=BOARD_SIZE)
    model = create_model(game, config.network, lr=config.training.lr)

    print(f"\n{'#' * 70}")
    print(f"#  Training: OthelloNNet on {BOARD_SIZE}x{BOARD_SIZE} Othello")
    print(f"#  {n_iters} iters, {n_sims} sims, {n_games} games/iter")
    print(f"#  512 filters, dropout=0.3, window buffer, nn_batch=8")
    print(f"{'#' * 70}")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    elapsed = time.time() - t0
    history['wall_time'] = elapsed

    # Save history
    with open(os.path.join(DATA_DIR, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    final = history['vs_random_win_rate'][-1]
    peak = max(history['vs_random_win_rate'])
    acc = sum(history['model_accepted'])
    total = len(history['model_accepted'])

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  RESULTS — OthelloNNet on {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"{'=' * 60}")
    print(f"  vs Random: {final:.0%} final, {peak:.0%} peak")
    print(f"  Models accepted: {acc}/{total}")
    print(f"  Final loss: {history['total_loss'][-1]:.3f}")
    print(f"  Final entropy: {history['policy_entropy'][-1]:.3f}")
    print(f"  Final depth: {history['mean_search_depth'][-1]:.1f}")
    time_str = f"{elapsed:.0f}s" if elapsed < 120 else f"{elapsed / 60:.1f}m ({elapsed / 3600:.1f}h)"
    print(f"  Training time: {time_str}")
    print(f"{'=' * 60}")

    # Verify weights saved
    best_path = os.path.join(DATA_DIR, 'othellonet', 'best.pt')
    if os.path.exists(best_path):
        size_mb = os.path.getsize(best_path) / (1024 * 1024)
        print(f"\n  Weights saved: {best_path} ({size_mb:.1f} MB)")
    else:
        print(f"\n  WARNING: No best.pt found at {best_path}")

    print(f"\n  NEXT: push weights from pod before terminating!")
    print(f"  cd /workspace/code/personal-research")
    print(f"  git add -f alphago/experiments/20260306_othello_10x10_cnn/")
    print(f"  git commit -m 'results' && git push")


if __name__ == '__main__':
    main()
