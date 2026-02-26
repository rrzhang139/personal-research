"""MCTS search: selection, expansion, evaluation, backpropagation.

Implements the AlphaZero-style MCTS that uses a neural network
for both the prior policy (to guide search) and the value estimate
(instead of random rollouts).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..games.base_game import Game
from .node import MCTSNode


@dataclass
class SearchDiagnostics:
    """Diagnostics collected during a single MCTS search call."""
    root_value: float = 0.0       # Q value at root after search
    policy_entropy: float = 0.0   # entropy of the visit-count policy
    max_depth: int = 0            # deepest node visited during search


class MCTS:
    """AlphaZero-style Monte Carlo Tree Search."""

    def __init__(self, game: Game, model, config):
        """
        Args:
            game: Game instance.
            model: Neural network with predict(state) -> (policy, value).
            config: MCTSConfig with search parameters.
        """
        self.game = game
        self.model = model
        self.config = config

    def search(self, state: np.ndarray, player: int, collect_diagnostics: bool = False) -> tuple[np.ndarray, SearchDiagnostics | None]:
        """Run MCTS from the given state and return action probabilities.

        Args:
            state: Current board state.
            player: Current player (1 or -1).
            collect_diagnostics: If True, return SearchDiagnostics alongside the policy.

        Returns:
            (action_probs, diagnostics): diagnostics is None if collect_diagnostics=False.
        """
        root = MCTSNode(state=state, player=player)

        # Get initial policy for root expansion
        canonical = self.game.get_canonical_state(state, player)
        policy, _ = self.model.predict(canonical)
        root.expand(self.game, policy)

        # Add Dirichlet noise to root for exploration
        self._add_noise(root)

        max_depth = 0

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            depth = 0

            # SELECT: walk down tree picking best PUCT child
            while not node.is_leaf():
                node = node.select_child(self.config.c_puct)
                depth += 1

            max_depth = max(max_depth, depth)

            # Check if this leaf is terminal
            if node.parent is not None:
                is_terminal, terminal_value = self.game.check_terminal(
                    node.state, node.action
                )
                if is_terminal:
                    # terminal_value is from perspective of player who just moved
                    # (which is node.parent.player). We need value from node.player's perspective.
                    node.backpropagate(-terminal_value)
                    continue

            # EXPAND & EVALUATE: use neural net
            canonical = self.game.get_canonical_state(node.state, node.player)
            policy, value = self.model.predict(canonical)
            node.expand(self.game, policy)

            # BACKPROPAGATE: value is from current player's perspective
            node.backpropagate(value)

        # Extract visit counts from root children
        action_probs = np.zeros(self.game.get_action_size(), dtype=np.float32)
        for action, child in root.children.items():
            action_probs[action] = child.N

        # Apply temperature
        if action_probs.sum() > 0:
            if self.config.temperature <= 0.01:
                # Nearly greedy: pick the most visited action
                best = np.argmax(action_probs)
                action_probs = np.zeros_like(action_probs)
                action_probs[best] = 1.0
            else:
                action_probs = action_probs ** (1.0 / self.config.temperature)
                action_probs /= action_probs.sum()

        # Collect diagnostics if requested
        diag = None
        if collect_diagnostics:
            # Root value: from the perspective of the current player
            # Since children store Q from their own perspective, root.Q after backprop
            # reflects the root player's view (the first backprop negation handles it)
            root_value = 0.0
            if root.children:
                # Best child Q (negated because child Q is from child's perspective)
                best_child = max(root.children.values(), key=lambda c: c.N)
                root_value = -best_child.Q

            # Policy entropy: H(pi) = -sum(pi * log(pi))
            pi = action_probs[action_probs > 0]
            policy_entropy = -np.sum(pi * np.log(pi + 1e-10))

            diag = SearchDiagnostics(
                root_value=root_value,
                policy_entropy=policy_entropy,
                max_depth=max_depth,
            )

        return action_probs, diag

    def _add_noise(self, root: MCTSNode):
        """Add Dirichlet noise to root priors for exploration."""
        if self.config.dirichlet_epsilon == 0:
            return

        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))
        eps = self.config.dirichlet_epsilon

        for i, action in enumerate(actions):
            root.children[action].P = (1 - eps) * root.children[action].P + eps * noise[i]
