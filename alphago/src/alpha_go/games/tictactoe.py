"""Tic-tac-toe implementation.

Board is a flat numpy array of length 9.
- 0 = empty, 1 = player 1 (X), -1 = player -1 (O)
- Actions are indices 0-8 corresponding to board positions.

Board layout:
  0 | 1 | 2
  ---------
  3 | 4 | 5
  ---------
  6 | 7 | 8
"""

from __future__ import annotations

import numpy as np

from .base_game import Game


class TicTacToe(Game):

    # All possible three-in-a-row lines
    WIN_LINES = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6],              # diagonals
    ]

    def get_initial_state(self) -> np.ndarray:
        return np.zeros(9, dtype=np.float32)

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        new_state = state.copy()
        new_state[action] = player
        return new_state

    def get_valid_moves(self, state: np.ndarray, player: int = 1) -> np.ndarray:
        return (state == 0).astype(np.float32)

    def check_terminal(self, state: np.ndarray, action: int, player: int = 1) -> tuple[bool, float]:
        # Check if the player who placed at `action` won
        player = state[action]
        for line in self.WIN_LINES:
            if action in line:
                if state[line[0]] == state[line[1]] == state[line[2]] == player:
                    return True, 1.0  # player who just moved wins

        # Check draw (board full)
        if np.all(state != 0):
            return True, 0.0

        return False, 0.0

    def get_board_size(self) -> int:
        return 9

    def get_board_shape(self) -> tuple[int, int]:
        return (3, 3)

    def get_action_size(self) -> int:
        return 9

    def get_canonical_state(self, state: np.ndarray, player: int) -> np.ndarray:
        return state * player

    def get_symmetries(self, state: np.ndarray, pi: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """8 symmetries: 4 rotations x 2 reflections."""
        symmetries = []
        board = state.reshape(3, 3)
        policy = pi.reshape(3, 3)

        for rotation in range(4):
            rb = np.rot90(board, rotation)
            rp = np.rot90(policy, rotation)
            symmetries.append((rb.flatten(), rp.flatten()))
            # Add horizontal flip
            symmetries.append((np.fliplr(rb).flatten(), np.fliplr(rp).flatten()))

        return symmetries

    def display(self, state: np.ndarray) -> str:
        symbols = {0: '.', 1: 'X', -1: 'O'}
        board = state.reshape(3, 3)
        rows = []
        for r in range(3):
            rows.append(' '.join(symbols[int(board[r, c])] for c in range(3)))
        return '\n'.join(rows)
