"""Abstract Game interface.

Any game playable by AlphaZero must implement this interface.
The game is fully defined by: state representation, legal moves,
transition function, and terminal detection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Game(ABC):
    """Abstract base class for two-player zero-sum board games.

    Conventions:
    - Players are 1 and -1. Player 1 moves first.
    - State is a numpy array (flat or multi-dimensional).
    - Actions are integers indexing into a fixed-size action space.
    """

    @abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """Return the starting board state."""

    @abstractmethod
    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        """Apply action by player, return the new state."""

    @abstractmethod
    def get_valid_moves(self, state: np.ndarray, player: int = 1) -> np.ndarray:
        """Return binary mask of valid actions (1 = legal, 0 = illegal).

        Args:
            state: Current board state.
            player: Player whose turn it is (1 or -1). Most games ignore this
                    since legal moves are player-independent, but Othello needs it.
        """

    @abstractmethod
    def check_terminal(self, state: np.ndarray, action: int, player: int = 1) -> tuple[bool, float]:
        """Check if the game is over after the given action was played.

        Args:
            state: Board state after the action was applied.
            action: The action that was just played.
            player: The player who just moved (1 or -1). Most games infer this
                    from state[action], but Othello needs it for pass moves.

        Returns:
            (is_terminal, value): value is 1 if the player who just moved won,
            -1 if they lost, 0 for draw. Only meaningful if is_terminal=True.
        """

    @abstractmethod
    def get_board_size(self) -> int:
        """Return the size of the state representation (flat)."""

    @abstractmethod
    def get_board_shape(self) -> tuple[int, int]:
        """Return the 2D board dimensions (rows, cols) for CNN reshaping."""

    @abstractmethod
    def get_action_size(self) -> int:
        """Return the total number of possible actions."""

    @abstractmethod
    def get_canonical_state(self, state: np.ndarray, player: int) -> np.ndarray:
        """Return state from the perspective of the given player.

        This lets the neural network always see the board as if it's player 1's turn.
        For symmetric games, this is just state * player.
        """

    @abstractmethod
    def get_symmetries(self, state: np.ndarray, pi: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return equivalent (state, policy) pairs under game symmetries.

        Used for data augmentation. For tic-tac-toe, there are 8 symmetries
        (4 rotations x 2 reflections). Return at least [(state, pi)].
        """

    def display(self, state: np.ndarray) -> str:
        """Return a human-readable string representation of the board."""
        return str(state)
