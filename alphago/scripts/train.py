#!/usr/bin/env python3
"""Universal AlphaZero training entry point.

Usage:
    python scripts/train.py --game tictactoe
    python scripts/train.py --game tictactoe --num-simulations 50 --lr 0.002 --num-iterations 20
    python scripts/train.py --game tictactoe --wandb
"""

import argparse

from alpha_go.games import get_game
from alpha_go.neural_net import create_model
from alpha_go.training.pipeline import run_pipeline
from alpha_go.utils.config import (
    AlphaZeroConfig,
    ArenaConfig,
    MCTSConfig,
    NetworkConfig,
    TrainingConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AlphaZero Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scripts/train.py --game tictactoe
  python scripts/train.py --game connect4 --num-simulations 50
  python scripts/train.py --game tictactoe --lr 0.002 --num-iterations 10 --wandb""",
    )

    # Game
    g = parser.add_argument_group('game')
    g.add_argument('--game', type=str, default='tictactoe',
                   help='Game to train on: tictactoe, connect4, othello, othello8, othello10, go, go9, go13, go19 (default: tictactoe)')
    g.add_argument('--board-size', type=int, default=None,
                   help='Override board size for Othello (6,8,10) or Go (9,13,19).')
    g.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility (default: 42)')

    # MCTS
    m = parser.add_argument_group('mcts', 'Monte Carlo Tree Search parameters')
    m.add_argument('--num-simulations', type=int, default=25,
                   help='MCTS sims per move. More = stronger but slower. Knee ~10-25 for ttt, ~50 for connect4 (default: 25)')
    m.add_argument('--c-puct', type=float, default=1.0,
                   help='Exploration constant in PUCT. Higher = explore more, lower = exploit (default: 1.0)')
    m.add_argument('--dirichlet-alpha', type=float, default=0.3,
                   help='Dirichlet noise concentration. 0.3 for small action spaces, 0.03 for Go (default: 0.3)')
    m.add_argument('--dirichlet-epsilon', type=float, default=0.25,
                   help='Noise weight at root. 0 = no noise, 0.25 = AlphaZero default (default: 0.25)')
    m.add_argument('--temp-threshold', type=int, default=15,
                   help='Move number after which play becomes greedy. Higher = explore longer in each game (default: 15)')
    m.add_argument('--nn-batch-size', type=int, default=1,
                   help='NN eval batch size within MCTS. 1=sequential, >1=virtual loss batching (default: 1)')

    # Network
    n = parser.add_argument_group('network', 'Neural network architecture')
    n.add_argument('--network', type=str, default='mlp', choices=['mlp', 'cnn', 'othellonet'],
                   help='Network type: mlp, cnn, or othellonet (default: mlp)')
    n.add_argument('--hidden-size', type=int, default=128,
                   help='Hidden layer width (MLP). 32 works for ttt, 128+ for harder games (default: 128)')
    n.add_argument('--num-layers', type=int, default=4,
                   help='Number of hidden layers (MLP). 2-4 for MLP (default: 4)')
    n.add_argument('--num-filters', type=int, default=64,
                   help='Number of conv filters (CNN). (default: 64)')
    n.add_argument('--num-res-blocks', type=int, default=4,
                   help='Number of residual blocks (CNN). (default: 4)')
    n.add_argument('--dropout', type=float, default=0.0,
                   help='Dropout rate for CNN heads. 0.3 = reference. (default: 0.0)')

    # Training
    t = parser.add_argument_group('training', 'Self-play and optimization')
    t.add_argument('--lr', type=float, default=0.001,
                   help='Adam learning rate (default: 0.001)')
    t.add_argument('--batch-size', type=int, default=64,
                   help='Minibatch size for training (default: 64)')
    t.add_argument('--epochs-per-iteration', type=int, default=10,
                   help='Training passes over replay buffer per iteration (default: 10)')
    t.add_argument('--max-buffer-size', type=int, default=50000,
                   help='Replay buffer capacity. Older examples dropped (default: 50000)')
    t.add_argument('--buffer-strategy', type=str, default='fifo', choices=['fifo', 'window'],
                   help='Buffer strategy: fifo (fixed-size deque) or window (last N iters) (default: fifo)')
    t.add_argument('--buffer-window', type=int, default=20,
                   help='For window strategy: keep examples from last N iterations (default: 20)')
    t.add_argument('--num-iterations', type=int, default=25,
                   help='Total self-play/train/arena cycles (default: 25)')
    t.add_argument('--games-per-iteration', type=int, default=100,
                   help='Self-play games generated per iteration (default: 100)')
    t.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                   help='Directory for model checkpoints and plots (default: checkpoints)')

    # Arena
    a = parser.add_argument_group('arena', 'Model evaluation')
    a.add_argument('--arena-games', type=int, default=40,
                   help='Games played to compare new vs old model (default: 40)')
    a.add_argument('--update-threshold', type=float, default=0.55,
                   help='Win rate to accept new model. 0.55 = must win >55%% (default: 0.55)')

    # Parallelism
    p = parser.add_argument_group('parallelism')
    p.add_argument('--num-workers', type=int, default=1,
                   help='Parallel workers for self-play/arena. 0=auto, 1=sequential (default: 1)')

    # Logging
    l = parser.add_argument_group('logging')
    l.add_argument('--wandb', action='store_true',
                   help='Enable Weights & Biases logging')
    l.add_argument('--wandb-project', type=str, default='alphazero',
                   help='W&B project name (default: alphazero)')

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
            nn_batch_size=args.nn_batch_size,
        ),
        network=NetworkConfig(
            network_type=args.network,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            num_res_blocks=args.num_res_blocks,
            dropout=args.dropout,
        ),
        training=TrainingConfig(
            lr=args.lr,
            batch_size=args.batch_size,
            epochs_per_iteration=args.epochs_per_iteration,
            max_buffer_size=args.max_buffer_size,
            buffer_strategy=args.buffer_strategy,
            buffer_window=args.buffer_window,
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
        num_workers=args.num_workers,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    # Setup game and model
    if args.board_size is not None and 'othello' in config.game:
        from alpha_go.games.othello import Othello
        game = Othello(size=args.board_size)
    elif args.board_size is not None and 'go' in config.game:
        from alpha_go.games.go import Go
        game = Go(size=args.board_size)
    else:
        game = get_game(config.game)
    model = create_model(game, config.network, lr=config.training.lr)

    # Run — pipeline handles all logging
    history = run_pipeline(game, model, config)

    return history


if __name__ == '__main__':
    main()
