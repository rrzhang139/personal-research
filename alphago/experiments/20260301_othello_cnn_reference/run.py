#!/usr/bin/env python3
"""Experiment: Othello CNN with reference-quality hyperparameters.

Matches alpha-zero-general reference settings: 512 filters, dropout 0.3,
200K buffer, 80 iterations, 25 sims. Uses GPU for training + VL batched
inference, sequential self-play (num_workers=1).

Target: >90% vs random (up from 73% peak with 64-filter CNN).
"""

import copy
import json
import os
import sys
import time

import numpy as np

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

# Reference-quality config
MCTS = MCTSConfig(num_simulations=25, nn_batch_size=32)
TRAINING = TrainingConfig(
    num_iterations=80,
    games_per_iteration=100,
    epochs_per_iteration=10,
    batch_size=64,
    lr=0.001,
    max_buffer_size=200_000,
    checkpoint_dir=os.path.join(DATA_DIR, 'cnn'),
)
ARENA = ArenaConfig(arena_games=40, update_threshold=0.55)

CNN_NETWORK = NetworkConfig(
    network_type="cnn",
    num_filters=512,
    num_res_blocks=4,
    dropout=0.3,
)


def main():
    config = AlphaZeroConfig(
        mcts=copy.deepcopy(MCTS),
        network=CNN_NETWORK,
        training=TRAINING,
        arena=ARENA,
        game='othello',
        seed=42,
        num_workers=1,
    )

    # Save config
    with open(os.path.join(EXPERIMENT_DIR, 'config.json'), 'w') as f:
        json.dump({
            'experiment': 'Othello CNN reference-quality',
            'game': 'othello',
            'network': {
                'type': 'cnn', 'num_filters': 512, 'num_res_blocks': 4, 'dropout': 0.3,
            },
            'mcts': {'num_simulations': 25, 'nn_batch_size': 32},
            'training': {
                'num_iterations': 80, 'games_per_iteration': 100,
                'epochs_per_iteration': 10, 'batch_size': 64, 'lr': 0.001,
                'max_buffer_size': 200000,
            },
            'arena': {'arena_games': 40, 'update_threshold': 0.55},
            'num_workers': 1, 'seed': 42,
        }, f, indent=2)

    game = get_game('othello')
    model = create_model(game, config.network, lr=config.training.lr)

    print(f"\n  Model params: {sum(p.numel() for p in model.net.parameters()):,}")
    print()

    t0 = time.time()
    history = run_pipeline(game, model, config)
    elapsed = time.time() - t0

    history['wall_time'] = elapsed

    # Save history
    path = os.path.join(DATA_DIR, 'history.json')
    with open(path, 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Othello CNN Reference — {elapsed/60:.1f} min")
    print(f"  Final vs Random: {history['vs_random_win_rate'][-1]:.0%}")
    print(f"  Best vs Random:  {max(history['vs_random_win_rate']):.0%}")
    print(f"  Final Loss:      {history['total_loss'][-1]:.3f}")
    print(f"  Models Accepted: {sum(history['model_accepted'])}/{len(history['model_accepted'])}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
