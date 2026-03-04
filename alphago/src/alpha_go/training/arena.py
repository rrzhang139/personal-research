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

        is_terminal, value = game.check_terminal(state, action, player)
        if is_terminal:
            # value is from perspective of player who just moved
            if value == 0:
                return 0  # draw
            return player if value > 0 else -player

        player = -player


def _worker_arena_game(args):
    """Top-level function for pool.map: play one arena game using process-global models."""
    from .parallel import _worker_model1, _worker_model2
    game_name, mcts_config, game_index, half = args

    from ..games import get_game
    game = get_game(game_name)

    if game_index < half:
        # model1 (new) is player 1
        result = play_arena_game(game, _worker_model1, _worker_model2, mcts_config)
        if result == 1:
            return 'new_win'
        elif result == -1:
            return 'old_win'
        return 'draw'
    else:
        # model1 (new) is player -1
        result = play_arena_game(game, _worker_model2, _worker_model1, mcts_config)
        if result == -1:
            return 'new_win'
        elif result == 1:
            return 'old_win'
        return 'draw'


def arena_compare(
    game: Game,
    new_model,
    old_model,
    mcts_config: MCTSConfig,
    num_games: int,
    num_workers: int = 1,
    game_name: str | None = None,
) -> tuple[float, dict]:
    """Compare new model vs old model over multiple games.

    Each model plays as both player 1 and player -1 (half the games each).

    Returns:
        (win_rate, stats): win_rate of new model, stats dict with wins/draws/losses.
    """
    if num_workers > 1 and game_name is not None:
        return _arena_compare_parallel(
            new_model, old_model, mcts_config, num_games, num_workers, game_name
        )

    return _arena_compare_sequential(game, new_model, old_model, mcts_config, num_games)


def _arena_compare_sequential(
    game: Game,
    new_model,
    old_model,
    mcts_config: MCTSConfig,
    num_games: int,
) -> tuple[float, dict]:
    """Original sequential arena comparison."""
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


def _arena_compare_parallel(
    new_model,
    old_model,
    mcts_config: MCTSConfig,
    num_games: int,
    num_workers: int,
    game_name: str,
) -> tuple[float, dict]:
    """Parallel arena comparison using multiprocessing pool."""
    from .parallel import create_arena_pool

    half = num_games // 2
    pool = create_arena_pool(new_model, old_model, num_workers)
    try:
        args_list = [
            (game_name, mcts_config, i, half)
            for i in range(num_games)
        ]
        results = pool.map(_worker_arena_game, args_list)
    finally:
        pool.close()
        pool.join()

    new_wins = results.count('new_win')
    old_wins = results.count('old_win')
    draws = results.count('draw')

    total = new_wins + old_wins + draws
    win_rate = (new_wins + 0.5 * draws) / total if total > 0 else 0.5

    stats = {
        'new_wins': new_wins,
        'old_wins': old_wins,
        'draws': draws,
        'win_rate': win_rate,
    }
    return win_rate, stats


def _worker_vs_random_game(args):
    """Top-level function for pool.map: play one vs-random game using process-global model."""
    from .parallel import _worker_model
    game_name, mcts_config, model_player = args

    from ..games import get_game
    game = get_game(game_name)

    arena_config = MCTSConfig(
        num_simulations=mcts_config.num_simulations,
        c_puct=mcts_config.c_puct,
        dirichlet_alpha=mcts_config.dirichlet_alpha,
        dirichlet_epsilon=0.0,
        temperature=0.01,
        temp_threshold=0,
    )

    mcts = MCTS(game, _worker_model, arena_config)
    state = game.get_initial_state()
    player = 1

    while True:
        if player == model_player:
            pi, _ = mcts.search(state, player)
            action = np.argmax(pi)
        else:
            valid = game.get_valid_moves(state, player)
            valid_actions = np.where(valid > 0)[0]
            action = np.random.choice(valid_actions)

        state = game.get_next_state(state, action, player)
        is_terminal, value = game.check_terminal(state, action, player)

        if is_terminal:
            if value == 0:
                return 'draw'
            winner = player if value > 0 else -player
            if winner == model_player:
                return 'win'
            return 'loss'

        player = -player


def play_vs_random(
    game: Game,
    model,
    mcts_config: MCTSConfig,
    num_games: int = 100,
    num_workers: int = 1,
    game_name: str | None = None,
) -> float:
    """Evaluate model against a random player. Returns model's win rate."""
    if num_workers > 1 and game_name is not None:
        return _play_vs_random_parallel(model, mcts_config, num_games, num_workers, game_name)

    return _play_vs_random_sequential(game, model, mcts_config, num_games)


def _play_vs_random_sequential(
    game: Game,
    model,
    mcts_config: MCTSConfig,
    num_games: int,
) -> float:
    """Original sequential vs-random evaluation."""
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
                valid = game.get_valid_moves(state, player)
                valid_actions = np.where(valid > 0)[0]
                action = np.random.choice(valid_actions)

            state = game.get_next_state(state, action, player)
            is_terminal, value = game.check_terminal(state, action, player)

            if is_terminal:
                if value == 0:
                    draws += 1
                else:
                    winner = player if value > 0 else -player
                    if winner == model_player:
                        wins += 1
                break

            player = -player

    return (wins + 0.5 * draws) / num_games


def _play_vs_random_parallel(
    model,
    mcts_config: MCTSConfig,
    num_games: int,
    num_workers: int,
    game_name: str,
) -> float:
    """Parallel vs-random evaluation using multiprocessing pool."""
    from .parallel import create_pool

    half = num_games // 2
    pool = create_pool(model, num_workers)
    try:
        args_list = [
            (game_name, mcts_config, 1 if i < half else -1)
            for i in range(num_games)
        ]
        results = pool.map(_worker_vs_random_game, args_list)
    finally:
        pool.close()
        pool.join()

    wins = results.count('win')
    draws = results.count('draw')
    return (wins + 0.5 * draws) / num_games
