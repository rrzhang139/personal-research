#!/usr/bin/env python3
"""Play against a trained AlphaZero model in the terminal.

Usage:
    python scripts/play.py --game tictactoe
    python scripts/play.py --game tictactoe --checkpoint checkpoints/best.pt --num-simulations 100
"""

import argparse
import os

import numpy as np

from alpha_go.games import get_game
from alpha_go.mcts.search import MCTS
from alpha_go.neural_net import create_model
from alpha_go.utils.config import MCTSConfig, NetworkConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play against a trained AlphaZero model in the terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scripts/play.py --game tictactoe
  python scripts/play.py --game connect4 --checkpoint baselines/connect4/best.pt
  python scripts/play.py --game tictactoe --num-simulations 200""",
    )
    parser.add_argument('--game', type=str, default='tictactoe',
                        help='Game to play: tictactoe, connect4 (default: tictactoe)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                        help='Path to model checkpoint (default: checkpoints/best.pt)')
    parser.add_argument('--num-simulations', type=int, default=100,
                        help='MCTS sims per AI move. More = stronger opponent (default: 100)')
    parser.add_argument('--network', type=str, default='mlp', choices=['mlp', 'cnn'],
                        help='Network type: mlp or cnn — must match checkpoint (default: mlp)')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden layer width (MLP) — must match checkpoint (default: 128)')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of hidden layers (MLP) — must match checkpoint (default: 4)')
    parser.add_argument('--num-filters', type=int, default=64,
                        help='Number of conv filters (CNN) — must match checkpoint (default: 64)')
    parser.add_argument('--num-res-blocks', type=int, default=4,
                        help='Number of residual blocks (CNN) — must match checkpoint (default: 4)')
    return parser.parse_args()


def main():
    args = parse_args()
    game = get_game(args.game)

    # Load model
    net_config = NetworkConfig(
        network_type=args.network,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        num_res_blocks=args.num_res_blocks,
    )
    model = create_model(game, net_config)

    if os.path.exists(args.checkpoint):
        model.load(args.checkpoint)
        print(f"Loaded model from {args.checkpoint}")
    else:
        print(f"Warning: No checkpoint found at {args.checkpoint}, using untrained model")

    mcts_config = MCTSConfig(
        num_simulations=args.num_simulations,
        c_puct=1.0,
        dirichlet_epsilon=0.0,  # no noise when playing
        temperature=0.01,       # nearly greedy
        temp_threshold=0,
    )

    print(f"\nPlaying {args.game}! You are X (player 1), AI is O (player -1).")
    print("Enter your move as a number (0-indexed).\n")

    state = game.get_initial_state()
    player = 1

    while True:
        print(game.display(state))
        print()

        valid = game.get_valid_moves(state, player)
        valid_actions = np.where(valid > 0)[0]

        if player == 1:
            # Human turn
            print(f"Valid moves: {valid_actions.tolist()}")
            while True:
                try:
                    action = int(input("Your move: "))
                    if valid[action] > 0:
                        break
                    print("Invalid move, try again.")
                except (ValueError, IndexError):
                    print("Enter a valid number.")
        else:
            # AI turn
            mcts = MCTS(game, model, mcts_config)
            pi, _ = mcts.search(state, player)
            action = np.argmax(pi)
            print(f"AI plays: {action}")

        state = game.get_next_state(state, action, player)
        is_terminal, value = game.check_terminal(state, action, player)

        if is_terminal:
            print(game.display(state))
            print()
            if value == 0:
                print("It's a draw!")
            elif player == 1:
                print("You win!")
            else:
                print("AI wins!")
            break

        player = -player


if __name__ == '__main__':
    main()
