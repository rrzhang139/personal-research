#!/usr/bin/env python3
"""Evaluate our Go model against GnuGo via GTP protocol.

Usage:
    python scripts/eval_vs_gnugo.py --weights path/to/best.pt --num-games 50
    python scripts/eval_vs_gnugo.py --weights path/to/best.pt --gnugo-level 5 --num-sims 200
"""

import argparse
import subprocess
import sys
import time

import numpy as np

from alpha_go.games.go import Go
from alpha_go.mcts.search import MCTS
from alpha_go.neural_net import create_model
from alpha_go.utils.config import MCTSConfig, NetworkConfig

# GTP column labels: A-H, J (skip I)
GTP_COLS = "ABCDEFGHJ"


def action_to_gtp(action: int, size: int = 9) -> str:
    """Convert our action index to GTP coordinate string."""
    if action == size * size:
        return "PASS"
    row, col = divmod(action, size)
    gtp_row = size - row  # our row 0 = GTP row 9
    return f"{GTP_COLS[col]}{gtp_row}"


def gtp_to_action(gtp_move: str, size: int = 9) -> int:
    """Convert GTP coordinate string to our action index."""
    gtp_move = gtp_move.strip().upper()
    if gtp_move == "PASS":
        return size * size
    if gtp_move == "RESIGN":
        return -1  # special: gnugo resigned
    col = GTP_COLS.index(gtp_move[0])
    gtp_row = int(gtp_move[1:])
    row = size - gtp_row
    return row * size + col


class GnuGoGTP:
    """Communicate with GnuGo via GTP protocol."""

    def __init__(self, level: int = 10, size: int = 9):
        self.proc = subprocess.Popen(
            ["gnugo", "--mode", "gtp", "--level", str(level), "--boardsize", str(size)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.size = size
        # Set boardsize and komi
        self._send(f"boardsize {size}")
        self._send("komi 7.5")

    def _send(self, command: str) -> str:
        """Send a GTP command and return the response."""
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        response = ""
        while True:
            line = self.proc.stdout.readline()
            if line.strip() == "":
                if response:
                    break
                continue
            response += line
        # GTP responses start with "= " (success) or "? " (error)
        return response.strip().lstrip("=").lstrip("?").strip()

    def clear(self):
        self._send("clear_board")

    def play(self, color: str, gtp_move: str):
        """Tell GnuGo that a move was played."""
        self._send(f"play {color} {gtp_move}")

    def genmove(self, color: str) -> str:
        """Ask GnuGo to generate a move. Returns GTP coordinate."""
        return self._send(f"genmove {color}")

    def final_score(self) -> str:
        """Ask GnuGo for the final score."""
        return self._send("final_score")

    def close(self):
        self._send("quit")
        self.proc.wait()


def play_game(game: Go, model, mcts_config: MCTSConfig, gnugo: GnuGoGTP,
              model_color: int = 1, verbose: bool = False) -> int:
    """Play one game: our model vs GnuGo.

    Args:
        model_color: 1 = our model plays Black, -1 = our model plays White.

    Returns:
        1 if our model wins, -1 if GnuGo wins, 0 for draw.
    """
    gnugo.clear()
    state = game.get_initial_state()
    player = 1  # Black goes first
    move_count = 0
    mcts_engine = MCTS(game, model, mcts_config)

    color_name = {1: "black", -1: "white"}

    while True:
        if player == model_color:
            # Our model's turn
            canonical = game.get_canonical_state(state, player)
            temp = 0.01  # always greedy during eval
            mcts_engine.temperature = temp
            pi, _ = mcts_engine.search(state, player)
            action = np.argmax(pi)  # greedy
            gtp_move = action_to_gtp(action, game.size)

            # Tell GnuGo about our move
            gnugo.play(color_name[player], gtp_move)

            if verbose:
                print(f"  Model ({color_name[model_color]}): {gtp_move}")
        else:
            # GnuGo's turn
            gtp_move = gnugo.genmove(color_name[player])

            if gtp_move.upper() == "RESIGN":
                if verbose:
                    print(f"  GnuGo resigned!")
                return 1  # our model wins

            action = gtp_to_action(gtp_move, game.size)

            if verbose:
                print(f"  GnuGo ({color_name[player]}): {gtp_move}")

        state = game.get_next_state(state, action, player)
        move_count += 1

        is_terminal, terminal_value = game.check_terminal(state, action, player)
        if is_terminal:
            # Use GnuGo's scoring for authoritative result
            gnugo_score = gnugo.final_score()
            if verbose:
                print(f"  GnuGo score: {gnugo_score}")
                print(game.display(state))

            # Parse score: "B+7.5" or "W+0.5" or "0"
            if gnugo_score.startswith("B+"):
                winner = 1  # Black wins
            elif gnugo_score.startswith("W+"):
                winner = -1  # White wins
            else:
                return 0  # draw

            return 1 if winner == model_color else -1

        player = -player

        if move_count > 200:
            # Safety: shouldn't happen in 9x9
            gnugo_score = gnugo.final_score()
            if gnugo_score.startswith("B+"):
                winner = 1
            elif gnugo_score.startswith("W+"):
                winner = -1
            else:
                return 0
            return 1 if winner == model_color else -1


def main():
    parser = argparse.ArgumentParser(description="Evaluate Go model vs GnuGo")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--num-games", type=int, default=50, help="Number of games to play")
    parser.add_argument("--num-sims", type=int, default=200, help="MCTS simulations per move")
    parser.add_argument("--gnugo-level", type=int, default=10, help="GnuGo level (1-10)")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=4)
    parser.add_argument("--nn-batch-size", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    game = Go(size=args.board_size)
    net_config = NetworkConfig(
        network_type="cnn",
        num_filters=args.num_filters,
        num_res_blocks=args.num_res_blocks,
    )
    model = create_model(game, net_config, lr=0.001)
    model.load(args.weights)
    print(f"Loaded weights from {args.weights}")

    mcts_config = MCTSConfig(
        num_simulations=args.num_sims,
        c_puct=1.5,
        dirichlet_alpha=0.03,
        dirichlet_epsilon=0.0,  # no noise during eval
        nn_batch_size=args.nn_batch_size,
    )

    print(f"\nModel vs GnuGo (level {args.gnugo_level}) on {args.board_size}x{args.board_size}")
    print(f"MCTS: {args.num_sims} sims, batch={args.nn_batch_size}")
    print(f"Games: {args.num_games} (half as Black, half as White)\n")

    gnugo = GnuGoGTP(level=args.gnugo_level, size=args.board_size)

    wins = 0
    losses = 0
    draws = 0
    t_start = time.time()

    for i in range(args.num_games):
        # Alternate colors
        model_color = 1 if i % 2 == 0 else -1
        color_str = "Black" if model_color == 1 else "White"

        result = play_game(game, model, mcts_config, gnugo,
                          model_color=model_color, verbose=args.verbose)

        if result == 1:
            wins += 1
            result_str = "WIN"
        elif result == -1:
            losses += 1
            result_str = "LOSS"
        else:
            draws += 1
            result_str = "DRAW"

        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed * 60

        print(f"Game {i+1:>3}/{args.num_games}: {result_str} (as {color_str})  "
              f"| {wins}W-{losses}L-{draws}D  "
              f"| WR: {wins/(i+1):.0%}  "
              f"| {rate:.1f} games/min")

    gnugo.close()

    total = wins + losses + draws
    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Final: {wins}W-{losses}L-{draws}D ({wins/total:.0%} win rate)")
    print(f"Time: {elapsed/60:.1f} min ({total/elapsed*60:.1f} games/min)")
    print(f"GnuGo level: {args.gnugo_level}")


if __name__ == "__main__":
    main()
