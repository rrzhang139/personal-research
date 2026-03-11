#!/usr/bin/env python3
"""Comprehensive evaluation: test vs GnuGo at multiple levels.

Usage:
    python scripts/eval_comprehensive.py --weights path/to/best.pt
    python scripts/eval_comprehensive.py --weights path/to/best.pt --levels 1 3 5 10 --games 20
"""
import argparse
import time

import numpy as np

from alpha_go.games.go import Go
from alpha_go.mcts.search import MCTS
from alpha_go.neural_net import create_model
from alpha_go.utils.config import MCTSConfig, NetworkConfig

from scripts.eval_vs_gnugo import GnuGoGTP, play_game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--num-sims", type=int, default=200)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=4)
    parser.add_argument("--nn-batch-size", type=int, default=8)
    args = parser.parse_args()

    game = Go(size=9)
    net_config = NetworkConfig(
        network_type="cnn",
        num_filters=args.num_filters,
        num_res_blocks=args.num_res_blocks,
    )
    model = create_model(game, net_config, lr=0.001)
    model.load(args.weights)
    print(f"Model: {args.weights}")
    print(f"Device: {model.net.device}")
    print(f"MCTS: {args.num_sims} sims, batch={args.nn_batch_size}")
    print()

    mcts_config = MCTSConfig(
        num_simulations=args.num_sims,
        c_puct=1.5,
        dirichlet_alpha=0.03,
        dirichlet_epsilon=0.0,
        nn_batch_size=args.nn_batch_size,
    )

    results = []
    print(f"{'Level':>5}  {'W':>3}  {'L':>3}  {'D':>3}  {'WR':>6}  {'Time':>6}")
    print("-" * 35)

    for level in args.levels:
        gnugo = GnuGoGTP(level=level, size=9)
        wins, losses, draws = 0, 0, 0
        t0 = time.time()

        for i in range(args.games):
            model_color = 1 if i % 2 == 0 else -1
            result = play_game(game, model, mcts_config, gnugo, model_color=model_color)
            if result == 1: wins += 1
            elif result == -1: losses += 1
            else: draws += 1

        gnugo.close()
        elapsed = time.time() - t0
        total = wins + losses + draws
        wr = wins / total if total > 0 else 0
        results.append((level, wins, losses, draws, wr))
        print(f"{level:>5}  {wins:>3}  {losses:>3}  {draws:>3}  {wr:>5.0%}  {elapsed:>5.1f}s")

    print()
    print("TSV:")
    print("level\twins\tlosses\tdraws\twin_rate")
    for level, w, l, d, wr in results:
        print(f"{level}\t{w}\t{l}\t{d}\t{wr:.3f}")


if __name__ == "__main__":
    main()
