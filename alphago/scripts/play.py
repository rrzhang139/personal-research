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
from alpha_go.neural_net.simple_net import SimpleNetWrapper
from alpha_go.utils.config import MCTSConfig, NetworkConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Play against AlphaZero")
    parser.add_argument('--game', type=str, default='tictactoe')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
    parser.add_argument('--num-simulations', type=int, default=100)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    game = get_game(args.game)

    # Load model
    net_config = NetworkConfig(hidden_size=args.hidden_size, num_layers=args.num_layers)
    model = SimpleNetWrapper(
        board_size=game.get_board_size(),
        action_size=game.get_action_size(),
        config=net_config,
    )

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

        valid = game.get_valid_moves(state)
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
        is_terminal, value = game.check_terminal(state, action)

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
