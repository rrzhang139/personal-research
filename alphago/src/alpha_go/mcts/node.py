"""MCTS tree node.

Each node stores:
- N: visit count
- W: total value (sum of backpropagated values)
- Q: mean value (W / N)
- P: prior probability from the neural network
- children: dict mapping action -> child node
"""

from __future__ import annotations

import numpy as np


class MCTSNode:
    """A node in the Monte Carlo search tree."""

    __slots__ = ['state', 'player', 'parent', 'action', 'N', 'W', 'Q', 'P', 'children', 'is_expanded']

    def __init__(self, state: np.ndarray, player: int, parent: 'MCTSNode | None' = None, action: int = -1, prior: float = 0.0):
        self.state = state
        self.player = player  # player whose turn it is at this node
        self.parent = parent
        self.action = action  # action that led to this node
        self.N = 0            # visit count
        self.W = 0.0          # total value
        self.Q = 0.0          # mean value (W/N)
        self.P = prior        # prior from neural net
        self.children: dict[int, MCTSNode] = {}
        self.is_expanded = False

    def is_leaf(self) -> bool:
        return not self.is_expanded

    def select_child(self, c_puct: float, fpu_reduction: float = 0.0) -> 'MCTSNode':
        """Select the child with the highest PUCT score.

        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

        Q is from the perspective of the player at this node.
        For unvisited children (N=0), use FPU: parent's value - fpu_reduction.
        """
        best_score = -float('inf')
        best_child = None
        sqrt_parent = np.sqrt(self.N)

        # FPU: unvisited children get parent's value minus a reduction
        if fpu_reduction > 0.0 and self.N > 0:
            fpu_value = self.Q - fpu_reduction  # parent's Q from parent perspective = self.Q
        else:
            fpu_value = 0.0

        for child in self.children.values():
            if child.N == 0:
                exploit = fpu_value
            else:
                # Q is stored from the child's perspective, negate for parent's view
                exploit = -child.Q
            explore = c_puct * child.P * sqrt_parent / (1 + child.N)
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
        """Expand this node by creating children for all legal actions.

        Children are created without states (lazy expansion). States are
        computed on demand via ensure_state() when a child is first visited.

        Args:
            game: Game instance for computing next states.
            action_priors: Policy vector from neural network (over all actions).
        """
        valid_moves = game.get_valid_moves(self.state, self.player)
        # Mask and renormalize priors to only legal moves
        action_priors = action_priors * valid_moves
        prior_sum = action_priors.sum()
        if prior_sum > 0:
            action_priors /= prior_sum
        else:
            # If network gives 0 to all legal moves, use uniform
            action_priors = valid_moves / valid_moves.sum()

        for action in range(game.get_action_size()):
            if valid_moves[action] > 0 and action_priors[action] > 1e-6:
                child = MCTSNode(
                    state=None,
                    player=-self.player,
                    parent=self,
                    action=action,
                    prior=action_priors[action],
                )
                self.children[action] = child

        self.is_expanded = True

    def backpropagate(self, value: float):
        """Propagate the evaluation value up the tree.

        Value is from the perspective of the player at the evaluated node.
        As we go up, we negate because parent has opposite player.
        """
        node = self
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value  # flip perspective for parent
            node = node.parent

    def apply_virtual_loss(self):
        """Apply virtual loss up the path to discourage parallel sims from same path.

        Increments N and adds +1 to W at every node from self to root.
        Since select_child uses exploit = -child.Q, increasing child.Q (via W+=1)
        makes -child.Q smaller, reducing the node's attractiveness to its parent.
        Sign does NOT alternate (unlike backprop) because every node should
        independently look worse to its own parent.
        """
        node = self
        while node is not None:
            node.N += 1
            node.W += 1.0
            node.Q = node.W / node.N
            node = node.parent

    def revert_virtual_loss(self):
        """Revert virtual loss from the path (undo one apply_virtual_loss call)."""
        node = self
        while node is not None:
            node.N -= 1
            node.W -= 1.0
            node.Q = node.W / node.N if node.N > 0 else 0.0
            node = node.parent
