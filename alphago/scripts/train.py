#!/usr/bin/env python3
"""Universal AlphaZero training entry point.

Usage:
    python scripts/train.py --game tictactoe
    python scripts/train.py --game tictactoe --num-simulations 50 --lr 0.002 --num-iterations 20
    python scripts/train.py --game tictactoe --wandb
"""

import argparse

from alpha_go.games import get_game
from alpha_go.neural_net.simple_net import SimpleNetWrapper
from alpha_go.training.pipeline import run_pipeline
from alpha_go.utils.config import (
    AlphaZeroConfig,
    ArenaConfig,
    MCTSConfig,
    NetworkConfig,
    TrainingConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Training")

    # Game
    parser.add_argument('--game', type=str, default='tictactoe', help='Game to train on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # MCTS
    parser.add_argument('--num-simulations', type=int, default=25)
    parser.add_argument('--c-puct', type=float, default=1.0)
    parser.add_argument('--dirichlet-alpha', type=float, default=0.3)
    parser.add_argument('--dirichlet-epsilon', type=float, default=0.25)
    parser.add_argument('--temp-threshold', type=int, default=15)

    # Network
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=4)

    # Training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs-per-iteration', type=int, default=10)
    parser.add_argument('--max-buffer-size', type=int, default=50000)
    parser.add_argument('--num-iterations', type=int, default=25)
    parser.add_argument('--games-per-iteration', type=int, default=100)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    # Arena
    parser.add_argument('--arena-games', type=int, default=40)
    parser.add_argument('--update-threshold', type=float, default=0.55)

    # Logging
    parser.add_argument('--wandb', action='store_true', help='Log to W&B')
    parser.add_argument('--wandb-project', type=str, default='alphazero')

    return parser.parse_args()


def main():
    args = parse_args()

    config = AlphaZeroConfig(
        mcts=MCTSConfig(
            num_simulations=args.num_simulations,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_epsilon=args.dirichlet_epsilon,
            temp_threshold=args.temp_threshold,
        ),
        network=NetworkConfig(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        ),
        training=TrainingConfig(
            lr=args.lr,
            batch_size=args.batch_size,
            epochs_per_iteration=args.epochs_per_iteration,
            max_buffer_size=args.max_buffer_size,
            num_iterations=args.num_iterations,
            games_per_iteration=args.games_per_iteration,
            checkpoint_dir=args.checkpoint_dir,
        ),
        arena=ArenaConfig(
            arena_games=args.arena_games,
            update_threshold=args.update_threshold,
        ),
        game=args.game,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    # Setup game and model
    game = get_game(config.game)
    model = SimpleNetWrapper(
        board_size=game.get_board_size(),
        action_size=game.get_action_size(),
        config=config.network,
        lr=config.training.lr,
    )

    # Run — pipeline handles all logging
    history = run_pipeline(game, model, config)

    return history


if __name__ == '__main__':
    main()
