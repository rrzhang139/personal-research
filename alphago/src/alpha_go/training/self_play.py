"""Self-play: generate training data by playing games with MCTS + current model.

Each self-play game produces a list of (state, policy, value) tuples.
The state is the canonical board, the policy is the MCTS visit distribution,
and the value is the final game outcome from that player's perspective.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..games.base_game import Game
from ..mcts.search import MCTS
from ..utils.config import MCTSConfig


@dataclass
class SelfPlayStats:
    """Aggregated statistics from a batch of self-play games."""
    p1_wins: int = 0
    p2_wins: int = 0
    draws: int = 0
    mean_game_length: float = 0.0
    mean_root_value: float = 0.0
    mean_policy_entropy: float = 0.0
    mean_search_depth: float = 0.0

    @property
    def outcomes_tuple(self) -> tuple[int, int, int]:
        return (self.p1_wins, self.p2_wins, self.draws)


def self_play_game(
    game: Game,
    model,
    mcts_config: MCTSConfig,
    collect_diagnostics: bool = False,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], int, dict]:
    """Play one game of self-play, returning training examples, outcome, and diagnostics.

    Returns:
        (examples, outcome, diag): examples is list of (canonical_state, mcts_policy, value).
        outcome: 1 = player 1 won, -1 = player 2 won, 0 = draw.
        diag: dict with diagnostic values (empty if collect_diagnostics=False).
    """
    mcts = MCTS(game, model, mcts_config)

    state = game.get_initial_state()
    player = 1
    trajectory = []  # (canonical_state, player, mcts_policy)
    move_count = 0
    root_values = []
    policy_entropies = []
    search_depths = []

    while True:
        canonical = game.get_canonical_state(state, player)

        # Use temperature: exploratory early, greedy late
        # NOTE: set on the MCTS instance's copy, not the shared config
        if move_count < mcts_config.temp_threshold:
            temp = 1.0
        else:
            temp = 0.01  # nearly greedy
        mcts.temperature = temp

        pi, diag = mcts.search(state, player, collect_diagnostics=collect_diagnostics)
        trajectory.append((canonical.copy(), player, pi.copy()))

        if diag is not None:
            root_values.append(diag.root_value)
            policy_entropies.append(diag.policy_entropy)
            search_depths.append(diag.max_depth)

        # Sample action from policy
        action = np.random.choice(len(pi), p=pi)
        state = game.get_next_state(state, action, player)
        move_count += 1

        is_terminal, terminal_value = game.check_terminal(state, action, player)
        if is_terminal:
            # Determine game outcome from P1's perspective
            if terminal_value == 0:
                outcome = 0
            else:
                outcome = player if terminal_value > 0 else -player

            # Assign per-position values
            examples = []
            for canonical_state, traj_player, traj_pi in trajectory:
                if traj_player == player:
                    v = terminal_value
                else:
                    v = -terminal_value
                examples.append((canonical_state, traj_pi, v))

            game_diag = {}
            if collect_diagnostics:
                game_diag = {
                    'game_length': move_count,
                    'mean_root_value': float(np.mean(root_values)) if root_values else 0.0,
                    'mean_policy_entropy': float(np.mean(policy_entropies)) if policy_entropies else 0.0,
                    'mean_search_depth': float(np.mean(search_depths)) if search_depths else 0.0,
                }

            return examples, outcome, game_diag

        player = -player


def _worker_self_play_game(args):
    """Top-level function for pool.map: play one self-play game using process-global model."""
    from .parallel import _worker_model
    game_name, mcts_config = args

    from ..games import get_game
    game = get_game(game_name)

    examples, outcome, diag = self_play_game(
        game, _worker_model, mcts_config, collect_diagnostics=True
    )
    return examples, outcome, diag


def generate_self_play_data(
    game: Game,
    model,
    mcts_config: MCTSConfig,
    num_games: int,
    augment: bool = True,
    num_workers: int = 1,
    game_name: str | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], SelfPlayStats]:
    """Generate training data from multiple self-play games.

    Args:
        num_workers: Number of parallel workers. 1 = sequential (no overhead).
        game_name: Game name string (required for parallel, used to reconstruct game in workers).

    Returns:
        (examples, stats): examples is the training data, stats has outcomes + diagnostics.
    """
    if num_workers > 1 and game_name is not None:
        return _generate_parallel(game, model, mcts_config, num_games, augment, num_workers, game_name)

    return _generate_sequential(game, model, mcts_config, num_games, augment)


def _generate_sequential(
    game: Game,
    model,
    mcts_config: MCTSConfig,
    num_games: int,
    augment: bool,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], SelfPlayStats]:
    """Original sequential implementation."""
    all_examples = []
    stats = SelfPlayStats()
    game_lengths = []
    root_values = []
    policy_entropies = []
    search_depths = []

    for _ in range(num_games):
        examples, outcome, diag = self_play_game(
            game, model, mcts_config, collect_diagnostics=True
        )

        if outcome == 1:
            stats.p1_wins += 1
        elif outcome == -1:
            stats.p2_wins += 1
        else:
            stats.draws += 1

        if diag:
            game_lengths.append(diag['game_length'])
            root_values.append(diag['mean_root_value'])
            policy_entropies.append(diag['mean_policy_entropy'])
            search_depths.append(diag['mean_search_depth'])

        if augment:
            for state, pi, v in examples:
                for sym_state, sym_pi in game.get_symmetries(state, pi):
                    all_examples.append((sym_state, sym_pi, v))
        else:
            all_examples.extend(examples)

    if game_lengths:
        stats.mean_game_length = float(np.mean(game_lengths))
        stats.mean_root_value = float(np.mean(root_values))
        stats.mean_policy_entropy = float(np.mean(policy_entropies))
        stats.mean_search_depth = float(np.mean(search_depths))

    return all_examples, stats


def _generate_parallel(
    game: Game,
    model,
    mcts_config: MCTSConfig,
    num_games: int,
    augment: bool,
    num_workers: int,
    game_name: str,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], SelfPlayStats]:
    """Parallel implementation using multiprocessing pool."""
    from .parallel import create_pool

    pool = create_pool(model, num_workers)
    try:
        args_list = [(game_name, mcts_config)] * num_games
        results = pool.map(_worker_self_play_game, args_list)
    finally:
        pool.close()
        pool.join()

    # Aggregate results
    all_examples = []
    stats = SelfPlayStats()
    game_lengths = []
    root_values = []
    policy_entropies = []
    search_depths = []

    for examples, outcome, diag in results:
        if outcome == 1:
            stats.p1_wins += 1
        elif outcome == -1:
            stats.p2_wins += 1
        else:
            stats.draws += 1

        if diag:
            game_lengths.append(diag['game_length'])
            root_values.append(diag['mean_root_value'])
            policy_entropies.append(diag['mean_policy_entropy'])
            search_depths.append(diag['mean_search_depth'])

        if augment:
            for state, pi, v in examples:
                for sym_state, sym_pi in game.get_symmetries(state, pi):
                    all_examples.append((sym_state, sym_pi, v))
        else:
            all_examples.extend(examples)

    if game_lengths:
        stats.mean_game_length = float(np.mean(game_lengths))
        stats.mean_root_value = float(np.mean(root_values))
        stats.mean_policy_entropy = float(np.mean(policy_entropies))
        stats.mean_search_depth = float(np.mean(search_depths))

    return all_examples, stats
