"""MCTS search: selection, expansion, evaluation, backpropagation.

Implements the AlphaZero-style MCTS that uses a neural network
for both the prior policy (to guide search) and the value estimate
(instead of random rollouts).

Supports two modes:
- Sequential (nn_batch_size=1): one NN eval per simulation (original)
- Batched (nn_batch_size>1): virtual loss + batched NN eval for throughput
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
        self.temperature = config.temperature  # can be overridden per-move
        self._fpu_reduction = getattr(config, 'fpu_reduction', 0.0)
        root_fpu = getattr(config, 'root_fpu_reduction', -1.0)
        self._root_fpu_reduction = self._fpu_reduction if root_fpu < 0 else root_fpu
        self._c_puct_base = getattr(config, 'c_puct_base', 0.0)

    def search(self, state: np.ndarray, player: int, collect_diagnostics: bool = False,
               reuse_root: 'MCTSNode | None' = None) -> tuple[np.ndarray, SearchDiagnostics | None]:
        """Run MCTS from the given state and return action probabilities.

        Args:
            reuse_root: If provided, reuse this node as root (tree reuse).
                Must already be expanded and have state set.

        Dispatches to sequential or batched search based on config.nn_batch_size.
        """
        is_fresh_root = True
        if reuse_root is not None and reuse_root.is_expanded:
            root = reuse_root
            # Detach from parent to prevent memory leaks
            root.parent = None
            is_fresh_root = False
        else:
            root = MCTSNode(state=state, player=player)
            # Get initial policy for root expansion
            canonical = self.game.get_canonical_state(state, player)
            policy, _ = self.model.predict(canonical)
            root.expand(self.game, policy)

        # Add Dirichlet noise only to fresh roots (not reused ones)
        if is_fresh_root:
            self._add_noise(root)

        # Run simulations
        nn_batch = getattr(self.config, 'nn_batch_size', 1)
        if nn_batch > 1:
            max_depth = self._run_batched(root, nn_batch)
        else:
            max_depth = self._run_sequential(root)

        # Store root for potential tree reuse
        self._last_root = root

        return self._extract_policy(root, max_depth, collect_diagnostics)

    def get_subtree(self, action: int) -> 'MCTSNode | None':
        """Get the child subtree for the given action (for tree reuse)."""
        root = getattr(self, '_last_root', None)
        if root is None:
            return None
        for child in root.children:
            if child.action == action:
                child.ensure_state(self.game)
                return child
        return None

    def _run_sequential(self, root: MCTSNode) -> int:
        """Original sequential MCTS: one NN eval per simulation."""
        max_depth = 0

        for _ in range(self.config.num_simulations):
            node = root
            depth = 0

            # SELECT: walk down tree picking best PUCT child
            while not node.is_leaf():
                fpu = self._root_fpu_reduction if node.parent is None else self._fpu_reduction
                node = node.select_child(self.config.c_puct, fpu, self._c_puct_base)
                depth += 1

            max_depth = max(max_depth, depth)

            # Lazy expansion: compute state on first visit
            node.ensure_state(self.game)

            # Check if this leaf is terminal
            if node.parent is not None:
                is_terminal, terminal_value = self.game.check_terminal(
                    node.state, node.action, node.parent.player
                )
                if is_terminal:
                    node.backpropagate(-terminal_value)
                    continue

            # EXPAND & EVALUATE: use neural net
            canonical = self.game.get_canonical_state(node.state, node.player)
            policy, value = self.model.predict(canonical)
            node.expand(self.game, policy)

            # BACKPROPAGATE: value is from current player's perspective
            node.backpropagate(value)

        return max_depth

    def _run_batched(self, root: MCTSNode, batch_size: int) -> int:
        """Batched MCTS with virtual loss.

        Collects up to batch_size leaf nodes per batch, applies virtual loss
        during selection to diversify paths, then batch-evaluates all leaves
        in one forward pass. This amortizes the per-call NN overhead.
        """
        max_depth = 0
        sims_done = 0
        total_sims = self.config.num_simulations

        while sims_done < total_sims:
            current_batch = min(batch_size, total_sims - sims_done)

            # Phase 1: Select paths with virtual loss
            leaves = []
            for _ in range(current_batch):
                node = root
                depth = 0

                while not node.is_leaf():
                    fpu = self._root_fpu_reduction if node.parent is None else self._fpu_reduction
                    node = node.select_child(self.config.c_puct, fpu, self._c_puct_base)
                    depth += 1

                max_depth = max(max_depth, depth)

                # Lazy expansion: compute state on first visit
                node.ensure_state(self.game)

                # Terminal nodes: backprop directly, no VL needed
                if node.parent is not None:
                    is_terminal, terminal_value = self.game.check_terminal(
                        node.state, node.action, node.parent.player
                    )
                    if is_terminal:
                        node.backpropagate(-terminal_value)
                        sims_done += 1
                        continue

                # Apply virtual loss so next sim in batch avoids this path
                node.apply_virtual_loss()
                leaves.append(node)

            if not leaves:
                continue

            # Phase 2: Batch evaluate unique leaves
            unique_leaves = list(dict.fromkeys(leaves))  # preserve order, deduplicate
            states = [
                self.game.get_canonical_state(leaf.state, leaf.player)
                for leaf in unique_leaves
            ]
            policies, values = self.model.predict_batch(states)
            leaf_results = {
                id(leaf): (policy, value)
                for leaf, policy, value in zip(unique_leaves, policies, values)
            }

            # Phase 3: Revert VL, expand, backpropagate
            for leaf in leaves:
                leaf.revert_virtual_loss()
                policy, value = leaf_results[id(leaf)]
                if not leaf.is_expanded:
                    leaf.expand(self.game, policy)
                leaf.backpropagate(value)
                sims_done += 1

        return max_depth

    def _extract_policy(self, root: MCTSNode, max_depth: int, collect_diagnostics: bool) -> tuple[np.ndarray, SearchDiagnostics | None]:
        """Extract visit-count policy and diagnostics from the root."""
        action_probs = np.zeros(self.game.get_action_size(), dtype=np.float32)
        for child in root.children:
            action_probs[child.action] = child.N

        # Apply temperature
        if action_probs.sum() > 0:
            if self.temperature <= 0.01:
                best = np.argmax(action_probs)
                action_probs = np.zeros_like(action_probs)
                action_probs[best] = 1.0
            else:
                action_probs = action_probs ** (1.0 / self.temperature)
                action_probs /= action_probs.sum()

        diag = None
        if collect_diagnostics:
            root_value = 0.0
            if root.children:
                best_child = max(root.children, key=lambda c: c.N)
                root_value = -best_child.Q

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

        n = len(root.children)
        if n == 0:
            return
        noise = np.random.dirichlet(np.full(n, self.config.dirichlet_alpha))
        eps = self.config.dirichlet_epsilon

        for i, child in enumerate(root.children):
            child.P = (1 - eps) * child.P + eps * noise[i]
