#!/usr/bin/env python3
"""Quick training: warm-start from best model, run N iterations, eval vs GnuGo.

Usage:
    python scripts/quick_train.py --iters 5 --games 30 --sims 200
"""
import argparse
import os
import time

from alpha_go.games.go import Go
from alpha_go.neural_net import create_model
from alpha_go.utils.config import (
    AlphaZeroConfig, MCTSConfig, NetworkConfig, TrainingConfig, ArenaConfig,
)
from alpha_go.training.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="experiments/20260310_go9_playout_cap/data/checkpoints/best.pt")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--nn-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--playout-cap", type=float, default=0.125)
    parser.add_argument("--cheap-fraction", type=float, default=0.15)
    parser.add_argument("--output-dir", type=str, default="/tmp/quick_train")
    parser.add_argument("--eval-games", type=int, default=0, help="vsRandom games (0=skip)")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--fpu-reduction", type=float, default=0.0)
    parser.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cosine"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    game = Go(size=9)

    config = AlphaZeroConfig(
        game="go9",
        mcts=MCTSConfig(
            num_simulations=args.sims,
            c_puct=args.c_puct,
            dirichlet_alpha=0.03,
            dirichlet_epsilon=0.25,
            temp_threshold=20,
            nn_batch_size=args.nn_batch_size,
            playout_cap_prob=args.playout_cap,
            playout_cap_cheap_fraction=args.cheap_fraction,
            fpu_reduction=args.fpu_reduction,
        ),
        network=NetworkConfig(
            network_type="cnn",
            num_filters=args.num_filters,
            num_res_blocks=args.num_res_blocks,
        ),
        training=TrainingConfig(
            lr=args.lr,
            batch_size=64,
            epochs_per_iteration=args.epochs,
            lr_schedule=args.lr_schedule,
            num_iterations=args.iters,
            games_per_iteration=args.games,
            max_buffer_size=200000,
            checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        ),
        arena=ArenaConfig(arena_games=0, eval_games=args.eval_games),
    )

    model = create_model(game, config.network, lr=config.training.lr, weight_decay=args.weight_decay)
    print(f"Device: {model.net.device}")

    if args.weights and os.path.exists(args.weights):
        model.load(args.weights)
        print(f"Warm-started from {args.weights}")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    train_time = time.time() - t0

    print(f"\nTotal training time: {train_time:.1f}s ({train_time/60:.1f}m)")
    best_path = os.path.join(args.output_dir, "checkpoints", "best.pt")
    print(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    main()
