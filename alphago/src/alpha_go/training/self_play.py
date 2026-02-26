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
        if move_count < mcts_config.temp_threshold:
            mcts.config.temperature = 1.0
        else:
            mcts.config.temperature = 0.01  # nearly greedy

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

        is_terminal, terminal_value = game.check_terminal(state, action)
        if is_terminal:
            # Determine game outcome from P1's perspective
            if terminal_value == 0:
                outcome = 0
            else:
                outcome = player  # player who just moved won

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


def generate_self_play_data(
    game: Game,
    model,
    mcts_config: MCTSConfig,
    num_games: int,
    augment: bool = True,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], SelfPlayStats]:
    """Generate training data from multiple self-play games.

    Returns:
        (examples, stats): examples is the training data, stats has outcomes + diagnostics.
    """
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
