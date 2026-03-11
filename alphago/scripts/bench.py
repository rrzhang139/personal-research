#!/usr/bin/env python3
"""Quick benchmark: MCTS speed + optional GnuGo eval.

Usage:
    python scripts/bench.py                          # MCTS speed only
    python scripts/bench.py --gnugo --gnugo-level 1  # + GnuGo eval
    python scripts/bench.py --weights path/to/best.pt --num-sims 200
"""
import argparse
import time
import numpy as np

from alpha_go.games.go import Go
from alpha_go.mcts.search import MCTS
from alpha_go.neural_net import create_model
from alpha_go.utils.config import MCTSConfig, NetworkConfig


def bench_mcts(game, model, mcts_config, num_moves=20):
    """Play num_moves from initial state, return avg time per search."""
    state = game.get_initial_state()
    player = 1
    mcts_engine = MCTS(game, model, mcts_config)
    mcts_engine.temperature = 0.01

    times = []
    for i in range(num_moves):
        t0 = time.perf_counter()
        pi, _ = mcts_engine.search(state, player)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        action = np.argmax(pi)
        state = game.get_next_state(state, action, player)
        is_terminal, _ = game.check_terminal(state, action, player)
        if is_terminal:
            break
        player = -player

    return times


def bench_gnugo(game, model, mcts_config, level=1, num_games=10):
    """Quick eval vs GnuGo. Returns (wins, losses, draws)."""
    from scripts.eval_vs_gnugo import GnuGoGTP, play_game

    gnugo = GnuGoGTP(level=level, size=game.size)
    wins, losses, draws = 0, 0, 0

    for i in range(num_games):
        model_color = 1 if i % 2 == 0 else -1
        result = play_game(game, model, mcts_config, gnugo, model_color=model_color)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

    gnugo.close()
    return wins, losses, draws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="experiments/20260310_go9_playout_cap/data/checkpoints/best.pt")
    parser.add_argument("--num-sims", type=int, default=200)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=4)
    parser.add_argument("--nn-batch-size", type=int, default=8)
    parser.add_argument("--num-moves", type=int, default=20)
    parser.add_argument("--gnugo", action="store_true")
    parser.add_argument("--gnugo-level", type=int, default=1)
    parser.add_argument("--gnugo-games", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps"])
    args = parser.parse_args()

    game = Go(size=9)
    net_config = NetworkConfig(
        network_type="cnn",
        num_filters=args.num_filters,
        num_res_blocks=args.num_res_blocks,
    )
    model = create_model(game, net_config, lr=0.001)
    model.load(args.weights)

    # Move to device
    if args.device == "mps":
        import torch
        model.net.device = torch.device("mps")
        model.net.to(model.net.device)

    mcts_config = MCTSConfig(
        num_simulations=args.num_sims,
        c_puct=1.5,
        dirichlet_alpha=0.03,
        dirichlet_epsilon=0.0,
        nn_batch_size=args.nn_batch_size,
    )

    print(f"Config: {args.num_sims} sims, batch={args.nn_batch_size}, {args.num_filters}f, device={args.device}")

    # MCTS speed
    print(f"\n--- MCTS Speed ({args.num_moves} moves) ---")
    times = bench_mcts(game, model, mcts_config, num_moves=args.num_moves)
    avg = np.mean(times)
    total = np.sum(times)
    print(f"  Total: {total:.2f}s  Avg/move: {avg:.3f}s  Moves: {len(times)}")
    print(f"  Sims/sec: {args.num_sims / avg:.0f}")

    # GnuGo eval
    if args.gnugo:
        print(f"\n--- GnuGo Level {args.gnugo_level} ({args.gnugo_games} games) ---")
        t0 = time.perf_counter()
        w, l, d = bench_gnugo(game, model, mcts_config, level=args.gnugo_level, num_games=args.gnugo_games)
        elapsed = time.perf_counter() - t0
        total_g = w + l + d
        print(f"  {w}W-{l}L-{d}D  WR: {w/total_g:.0%}  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
