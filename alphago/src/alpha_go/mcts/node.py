"""MCTS tree node.

Each node stores:
- N: visit count
- W: total value (sum of backpropagated values)
- P: prior probability from the neural network
- children: list of child nodes
"""

from __future__ import annotations

import math

import numpy as np


class MCTSNode:
    """A node in the Monte Carlo search tree."""

    __slots__ = ['state', 'player', 'parent', 'action', 'N', 'W', 'P',
                 'children', 'is_expanded']

    def __init__(self, state: np.ndarray, player: int, parent: 'MCTSNode | None' = None, action: int = -1, prior: float = 0.0):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.N = 0
        self.W = 0.0
        self.P = prior
        self.children: list[MCTSNode] = []
        self.is_expanded = False

    def is_leaf(self) -> bool:
        return not self.is_expanded

    @property
    def Q(self) -> float:
        """Mean value, computed on demand."""
        return self.W / self.N if self.N > 0 else 0.0

    def select_child(self, c_puct: float, fpu_reduction: float = 0.0, c_puct_base: float = 0.0) -> 'MCTSNode':
        """Select child with highest PUCT score using pure Python loop."""
        children = self.children
        if not children:
            return None

        sqrt_parent = math.sqrt(self.N)

        if c_puct_base > 0:
            c_puct = c_puct * (math.log((self.N + c_puct_base + 1) / c_puct_base) + 1)

        # FPU for unvisited children
        if fpu_reduction > 0.0 and self.N > 0:
            fpu_value = self.Q - fpu_reduction
        else:
            fpu_value = 0.0

        best_score = -1e18
        best_child = children[0]

        for child in children:
            n = child.N
            if n > 0:
                exploit = -child.W / n
            else:
                exploit = fpu_value
            explore = c_puct * child.P * sqrt_parent / (1 + n)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def ensure_state(self, game):
        """Lazily compute state from parent on first visit."""
        if self.state is None and self.parent is not None:
            self.state = game.get_next_state(self.parent.state, self.action, self.parent.player)

    def expand(self, game, action_priors: np.ndarray):
        """Expand node by creating children for legal actions with significant prior."""
        valid_moves = game.get_valid_moves(self.state, self.player)
        action_priors = action_priors * valid_moves
        prior_sum = action_priors.sum()
        if prior_sum > 0:
            action_priors /= prior_sum
        else:
            action_priors = valid_moves / valid_moves.sum()

        # Sparse iteration: only legal moves with sufficient prior
        actions = np.nonzero((valid_moves > 0) & (action_priors > 1e-6))[0]

        # Batch-convert numpy→Python (much faster than per-element int()/float())
        action_list = actions.tolist()
        prior_list = action_priors[actions].tolist()

        next_player = -self.player
        _new = object.__new__
        _cls = MCTSNode
        children = []
        _append = children.append
        for i in range(len(action_list)):
            child = _new(_cls)
            child.state = None
            child.player = next_player
            child.parent = self
            child.action = action_list[i]
            child.N = 0
            child.W = 0.0
            child.P = prior_list[i]
            child.children = []
            child.is_expanded = False
            _append(child)

        self.children = children
        self.is_expanded = True

    def backpropagate(self, value: float):
        """Propagate value up the tree. Q is computed on demand (W/N)."""
        node = self
        while node is not None:
            node.N += 1
            node.W += value
            value = -value
            node = node.parent

    def apply_virtual_loss(self):
        """Apply virtual loss up the path."""
        node = self
        while node is not None:
            node.N += 1
            node.W += 1.0
            node = node.parent

    def revert_virtual_loss(self):
        """Revert virtual loss."""
        node = self
        while node is not None:
            node.N -= 1
            node.W -= 1.0
            node = node.parent
