"""MCTS tree node.

Each node stores:
- N: visit count
- W: total value (sum of backpropagated values)
- P: prior probability from the neural network
- children: list of child nodes (for fast vectorized selection)
"""

from __future__ import annotations

import math

import numpy as np


class MCTSNode:
    """A node in the Monte Carlo search tree."""

    __slots__ = ['state', 'player', 'parent', 'action', 'N', 'W', 'P',
                 'children', '_child_actions', '_child_N', '_child_W', '_child_P',
                 'is_expanded', '_num_children', '_parent_idx']

    def __init__(self, state: np.ndarray, player: int, parent: 'MCTSNode | None' = None, action: int = -1, prior: float = 0.0):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.N = 0
        self.W = 0.0
        self.P = prior
        self.children: list[MCTSNode] = []
        # Parallel arrays for vectorized select_child
        self._child_actions: np.ndarray | None = None
        self._child_N: np.ndarray | None = None
        self._child_W: np.ndarray | None = None
        self._child_P: np.ndarray | None = None
        self._num_children = 0
        self._parent_idx = -1  # index in parent's children list
        self.is_expanded = False

    def is_leaf(self) -> bool:
        return not self.is_expanded

    @property
    def Q(self) -> float:
        """Mean value, computed on demand."""
        return self.W / self.N if self.N > 0 else 0.0

    def select_child(self, c_puct: float, fpu_reduction: float = 0.0, c_puct_base: float = 0.0) -> 'MCTSNode':
        """Select child with highest PUCT score using vectorized numpy ops."""
        n = self._num_children
        if n == 0:
            return None

        sqrt_parent = math.sqrt(self.N)

        if c_puct_base > 0:
            c_puct = c_puct * (math.log((self.N + c_puct_base + 1) / c_puct_base) + 1)

        child_N = self._child_N[:n]
        child_W = self._child_W[:n]
        child_P = self._child_P[:n]

        # FPU for unvisited children
        if fpu_reduction > 0.0 and self.N > 0:
            fpu_value = self.Q - fpu_reduction
        else:
            fpu_value = 0.0

        # Vectorized PUCT computation
        visited = child_N > 0
        exploit = np.where(visited, -child_W / np.maximum(child_N, 1), fpu_value)
        explore = c_puct * child_P * sqrt_parent / (1 + child_N)
        scores = exploit + explore

        best_idx = int(np.argmax(scores))
        return self.children[best_idx]

    def ensure_state(self, game):
        """Lazily compute state from parent on first visit."""
        if self.state is None and self.parent is not None:
            self.state = game.get_next_state(self.parent.state, self.action, self.parent.player)

    def expand(self, game, action_priors: np.ndarray):
        """Expand node by creating children for legal actions with significant prior.

        Uses sparse iteration (only legal moves) and stores parallel arrays
        for vectorized select_child.
        """
        valid_moves = game.get_valid_moves(self.state, self.player)
        action_priors = action_priors * valid_moves
        prior_sum = action_priors.sum()
        if prior_sum > 0:
            action_priors /= prior_sum
        else:
            action_priors = valid_moves / valid_moves.sum()

        # Sparse iteration: only legal moves with sufficient prior
        mask = (valid_moves > 0) & (action_priors > 1e-6)
        actions = np.nonzero(mask)[0]
        n = len(actions)

        children = []
        child_P = np.empty(n, dtype=np.float32)
        for i, action in enumerate(actions):
            p = float(action_priors[action])
            child = MCTSNode(
                state=None,
                player=-self.player,
                parent=self,
                action=int(action),
                prior=p,
            )
            child._parent_idx = i
            children.append(child)
            child_P[i] = p

        self.children = children
        self._child_actions = actions.astype(np.int32)
        self._child_N = np.zeros(n, dtype=np.float64)
        self._child_W = np.zeros(n, dtype=np.float64)
        self._child_P = child_P
        self._num_children = n
        self.is_expanded = True

    def backpropagate(self, value: float):
        """Propagate value up the tree. No Q division — Q is computed on demand."""
        node = self
        while node is not None:
            node.N += 1
            node.W += value
            # Sync to parent's parallel arrays using stored index
            idx = node._parent_idx
            if idx >= 0 and node.parent is not None:
                node.parent._child_N[idx] = node.N
                node.parent._child_W[idx] = node.W
            value = -value
            node = node.parent

    def apply_virtual_loss(self):
        """Apply virtual loss up the path. Skip Q recomputation."""
        node = self
        while node is not None:
            node.N += 1
            node.W += 1.0
            idx = node._parent_idx
            if idx >= 0 and node.parent is not None:
                node.parent._child_N[idx] = node.N
                node.parent._child_W[idx] = node.W
            node = node.parent

    def revert_virtual_loss(self):
        """Revert virtual loss. Skip Q recomputation."""
        node = self
        while node is not None:
            node.N -= 1
            node.W -= 1.0
            idx = node._parent_idx
            if idx >= 0 and node.parent is not None:
                node.parent._child_N[idx] = node.N
                node.parent._child_W[idx] = node.W
            node = node.parent
