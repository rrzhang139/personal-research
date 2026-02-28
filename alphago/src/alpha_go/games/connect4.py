"""Connect Four implementation.

6 rows x 7 columns board. Pieces drop to the lowest empty row in a column.
- 0 = empty, 1 = player 1 (X), -1 = player -1 (O)
- Actions are column indices 0-6.
- First player wins with perfect play (solved by Allis 1988).

Board layout (row 0 = top, row 5 = bottom):
  .  .  .  .  .  .  .
  .  .  .  .  .  .  .
  .  .  .  .  .  .  .
  .  .  .  .  .  .  .
  .  .  .  .  .  .  .
  .  .  .  .  .  .  .
  0  1  2  3  4  5  6   ← column actions

State is stored as a flat array of length 42 (row-major, top to bottom).
Index = row * 7 + col.
"""

from __future__ import annotations

import numpy as np

from .base_game import Game

ROWS = 6
COLS = 7


class ConnectFour(Game):

    def get_initial_state(self) -> np.ndarray:
        return np.zeros(ROWS * COLS, dtype=np.float32)

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        new_state = state.copy()
        board = new_state.reshape(ROWS, COLS)
        # Drop piece to lowest empty row in the column
        for row in range(ROWS - 1, -1, -1):
            if board[row, action] == 0:
                board[row, action] = player
                break
        return new_state

    def get_valid_moves(self, state: np.ndarray, player: int = 1) -> np.ndarray:
        board = state.reshape(ROWS, COLS)
        # A column is valid if its top row is empty
        return (board[0, :] == 0).astype(np.float32)

    def check_terminal(self, state: np.ndarray, action: int, player: int = 1) -> tuple[bool, float]:
        board = state.reshape(ROWS, COLS)

        # Find the row where the last piece was placed
        row = -1
        for r in range(ROWS):
            if board[r, action] != 0:
                row = r
                break
        if row == -1:
            return False, 0.0

        player = board[row, action]

        # Check all four directions from the placed piece
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag-down-right, diag-down-left
        for dr, dc in directions:
            count = 1
            # Count in positive direction
            for i in range(1, 4):
                r, c = row + dr * i, action + dc * i
                if 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == player:
                    count += 1
                else:
                    break
            # Count in negative direction
            for i in range(1, 4):
                r, c = row - dr * i, action - dc * i
                if 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == player:
                    count += 1
                else:
                    break
            if count >= 4:
                return True, 1.0  # player who just moved wins

        # Check draw (board full)
        if np.all(board[0, :] != 0):  # top row full = board full
            return True, 0.0

        return False, 0.0

    def get_board_size(self) -> int:
        return ROWS * COLS  # 42

    def get_board_shape(self) -> tuple[int, int]:
        return (ROWS, COLS)  # (6, 7)

    def get_action_size(self) -> int:
        return COLS  # 7

    def get_canonical_state(self, state: np.ndarray, player: int) -> np.ndarray:
        return state * player

    def get_symmetries(self, state: np.ndarray, pi: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Connect Four has 1 symmetry: left-right mirror.

        Vertical flip doesn't work (gravity). Rotations don't work (not square).
        """
        board = state.reshape(ROWS, COLS)
        mirrored_board = np.fliplr(board).flatten()
        mirrored_pi = pi[::-1].copy()  # reverse column order
        return [
            (state.copy(), pi.copy()),
            (mirrored_board, mirrored_pi),
        ]

    def display(self, state: np.ndarray) -> str:
        symbols = {0: '.', 1: 'X', -1: 'O'}
        board = state.reshape(ROWS, COLS)
        rows = []
        for r in range(ROWS):
            rows.append(' '.join(symbols[int(board[r, c])] for c in range(COLS)))
        rows.append(' '.join(str(c) for c in range(COLS)))
        return '\n'.join(rows)
