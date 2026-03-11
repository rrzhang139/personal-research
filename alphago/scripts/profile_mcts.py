#!/usr/bin/env python3
"""Profile MCTS to find bottlenecks."""
import cProfile
import pstats
import io
import numpy as np

from alpha_go.games.go import Go
from alpha_go.mcts.search import MCTS
from alpha_go.neural_net import create_model
from alpha_go.utils.config import MCTSConfig, NetworkConfig


def run_search(game, model, mcts_config, num_moves=5):
    state = game.get_initial_state()
    player = 1
    mcts_engine = MCTS(game, model, mcts_config)
    mcts_engine.temperature = 0.01

    for _ in range(num_moves):
        pi, _ = mcts_engine.search(state, player)
        action = np.argmax(pi)
        state = game.get_next_state(state, action, player)
        is_terminal, _ = game.check_terminal(state, action, player)
        if is_terminal:
            break
        player = -player


def main():
    game = Go(size=9)
    net_config = NetworkConfig(network_type="cnn", num_filters=128, num_res_blocks=4)
    model = create_model(game, net_config, lr=0.001)
    model.load("experiments/20260310_go9_playout_cap/data/checkpoints/best.pt")

    mcts_config = MCTSConfig(
        num_simulations=200, c_puct=1.5, dirichlet_alpha=0.03,
        dirichlet_epsilon=0.0, nn_batch_size=8,
    )

    pr = cProfile.Profile()
    pr.enable()
    run_search(game, model, mcts_config, num_moves=5)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

    print("\n--- By tottime (self time) ---")
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
    ps2.print_stats(30)
    print(s2.getvalue())


if __name__ == "__main__":
    main()
