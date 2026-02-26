"""Arena: pit two models against each other to decide if the new model is better.

Plays a series of games where each model takes turns going first.
Returns the win rate of the new model.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from ..games.base_game import Game
from ..mcts.search import MCTS
from ..utils.config import MCTSConfig


def play_arena_game(
    game: Game,
    model1,
    model2,
    mcts_config: MCTSConfig,
) -> int:
    """Play one game between model1 (player 1) and model2 (player -1).

    Returns:
        1 if model1 wins, -1 if model2 wins, 0 for draw.
    """
    # Use greedy play in arena (low temperature)
    arena_config = MCTSConfig(
        num_simulations=mcts_config.num_simulations,
        c_puct=mcts_config.c_puct,
        dirichlet_alpha=mcts_config.dirichlet_alpha,
        dirichlet_epsilon=0.0,  # no noise in arena
        temperature=0.01,       # nearly greedy
        temp_threshold=0,
    )

    mcts1 = MCTS(game, model1, arena_config)
    mcts2 = MCTS(game, model2, arena_config)

    state = game.get_initial_state()
    player = 1

    while True:
        if player == 1:
            pi, _ = mcts1.search(state, player)
        else:
            pi, _ = mcts2.search(state, player)

        action = np.argmax(pi)
        state = game.get_next_state(state, action, player)

        is_terminal, value = game.check_terminal(state, action)
        if is_terminal:
            # value is from perspective of player who just moved
            if value == 0:
                return 0  # draw
            return player  # player who just moved won

        player = -player


def arena_compare(
    game: Game,
    new_model,
    old_model,
    mcts_config: MCTSConfig,
    num_games: int,
) -> tuple[float, dict]:
    """Compare new model vs old model over multiple games.

    Each model plays as both player 1 and player -1 (half the games each).

    Returns:
        (win_rate, stats): win_rate of new model, stats dict with wins/draws/losses.
    """
    new_wins = 0
    old_wins = 0
    draws = 0

    half = num_games // 2

    for i in range(num_games):
        if i < half:
            # New model is player 1
            result = play_arena_game(game, new_model, old_model, mcts_config)
            if result == 1:
                new_wins += 1
            elif result == -1:
                old_wins += 1
            else:
                draws += 1
        else:
            # New model is player -1
            result = play_arena_game(game, old_model, new_model, mcts_config)
            if result == -1:
                new_wins += 1
            elif result == 1:
                old_wins += 1
            else:
                draws += 1

    total = new_wins + old_wins + draws
    win_rate = (new_wins + 0.5 * draws) / total if total > 0 else 0.5

    stats = {
        'new_wins': new_wins,
        'old_wins': old_wins,
        'draws': draws,
        'win_rate': win_rate,
    }
    return win_rate, stats


def play_vs_random(game: Game, model, mcts_config: MCTSConfig, num_games: int = 100) -> float:
    """Evaluate model against a random player. Returns model's win rate."""
    wins = 0
    draws = 0

    arena_config = MCTSConfig(
        num_simulations=mcts_config.num_simulations,
        c_puct=mcts_config.c_puct,
        dirichlet_alpha=mcts_config.dirichlet_alpha,
        dirichlet_epsilon=0.0,
        temperature=0.01,
        temp_threshold=0,
    )

    half = num_games // 2

    for i in range(num_games):
        mcts = MCTS(game, model, arena_config)
        state = game.get_initial_state()
        player = 1
        model_player = 1 if i < half else -1

        while True:
            if player == model_player:
                pi, _ = mcts.search(state, player)
                action = np.argmax(pi)
            else:
                valid = game.get_valid_moves(state)
                valid_actions = np.where(valid > 0)[0]
                action = np.random.choice(valid_actions)

            state = game.get_next_state(state, action, player)
            is_terminal, value = game.check_terminal(state, action)

            if is_terminal:
                if value == 0:
                    draws += 1
                elif player == model_player:
                    wins += 1  # model just moved and won
                break

            player = -player

    return (wins + 0.5 * draws) / num_games
