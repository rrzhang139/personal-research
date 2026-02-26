"""Abstract neural network interface.

The network takes a board state and outputs:
- policy: probability distribution over actions (where to play)
- value: scalar estimate of position value (who's winning)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseNet(ABC):
    """Abstract base class for AlphaZero neural networks."""

    @abstractmethod
    def predict(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Run inference on a single state.

        Args:
            state: Canonical board state (from current player's perspective).

        Returns:
            (policy, value): policy is probability vector over actions,
            value is scalar in [-1, 1] estimating current player's advantage.
        """

    @abstractmethod
    def train_step(self, states: np.ndarray, target_pis: np.ndarray, target_vs: np.ndarray) -> dict[str, float]:
        """Run one training step on a batch.

        Args:
            states: Batch of canonical board states, shape (B, board_size).
            target_pis: Target policy vectors, shape (B, action_size).
            target_vs: Target values, shape (B,).

        Returns:
            Dict of loss components: {'total_loss', 'policy_loss', 'value_loss'}.
        """

    @abstractmethod
    def save(self, path: str):
        """Save model weights to disk."""

    @abstractmethod
    def load(self, path: str):
        """Load model weights from disk."""

    @abstractmethod
    def clone(self) -> 'BaseNet':
        """Return a deep copy of this network (for arena comparison)."""
