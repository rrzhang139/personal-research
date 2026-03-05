"""Othello (Reversi) implementation — configurable board size (default 6x6).

Pieces flip opponent discs when sandwiched.
- 0 = empty, 1 = player 1 (X/Black), -1 = player -1 (O/White)
- Actions 0..N*N-1 = board positions (row * N + col). Action N*N = pass.
- A player must pass when they have no legal placements.
- Game ends when both players must pass (or board is full).
- Winner = player with more pieces.
"""

from __future__ import annotations

import numpy as np

from .base_game import Game

# Module-level constants for backward compatibility (tests import these)
SIZE = 6
PASS_ACTION = SIZE * SIZE  # 36

# 8 directions: (row_delta, col_delta)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]


class Othello(Game):

    def __init__(self, size: int = 6):
        self.size = size
        self.pass_action = size * size

    def get_initial_state(self) -> np.ndarray:
        n = self.size
        state = np.zeros(n * n, dtype=np.float32)
        mid = n // 2
        state[(mid - 1) * n + (mid - 1)] = -1
        state[(mid - 1) * n + mid] = 1
        state[mid * n + (mid - 1)] = 1
        state[mid * n + mid] = -1
        return state

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        n = self.size
        new_state = state.copy()
        if action == self.pass_action:
            return new_state

        board = new_state.reshape(n, n)
        row, col = divmod(action, n)
        board[row, col] = player

        for flip_r, flip_c in self._get_flips(board, row, col, player):
            board[flip_r, flip_c] = player

        return new_state

    def get_valid_moves(self, state: np.ndarray, player: int = 1) -> np.ndarray:
        n = self.size
        board = state.reshape(n, n)
        valid = np.zeros(n * n + 1, dtype=np.float32)

        for r in range(n):
            for c in range(n):
                if board[r, c] == 0 and self._get_flips(board, r, c, player):
                    valid[r * n + c] = 1.0

        if valid[:n * n].sum() == 0:
            valid[self.pass_action] = 1.0

        return valid

    def check_terminal(self, state: np.ndarray, action: int, player: int = 1) -> tuple[bool, float]:
        n = self.size
        board = state.reshape(n, n)

        p1_has_moves = self._has_any_move(board, 1)
        p2_has_moves = self._has_any_move(board, -1)
        board_full = np.all(board != 0)

        if not (p1_has_moves or p2_has_moves) or board_full:
            p1_count = np.sum(board == 1)
            p2_count = np.sum(board == -1)

            if p1_count > p2_count:
                winner = 1
            elif p2_count > p1_count:
                winner = -1
            else:
                return True, 0.0

            return True, 1.0 if winner == player else -1.0

        return False, 0.0

    def get_board_size(self) -> int:
        return self.size * self.size

    def get_board_shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    def get_action_size(self) -> int:
        return self.size * self.size + 1

    def get_canonical_state(self, state: np.ndarray, player: int) -> np.ndarray:
        return state * player

    def get_symmetries(self, state: np.ndarray, pi: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """8 symmetries: 4 rotations x 2 reflections."""
        n = self.size
        symmetries = []
        board = state.reshape(n, n)
        board_pi = pi[:n * n].reshape(n, n)
        pass_prob = pi[self.pass_action]

        for rotation in range(4):
            rb = np.rot90(board, rotation)
            rp = np.rot90(board_pi, rotation)
            sym_pi = np.append(rp.flatten(), pass_prob)
            symmetries.append((rb.flatten(), sym_pi))
            fb = np.fliplr(rb)
            fp = np.fliplr(rp)
            sym_pi_flip = np.append(fp.flatten(), pass_prob)
            symmetries.append((fb.flatten(), sym_pi_flip))

        return symmetries

    def display(self, state: np.ndarray) -> str:
        n = self.size
        symbols = {0: '.', 1: 'X', -1: 'O'}
        board = state.reshape(n, n)
        rows = ['  ' + ' '.join(str(c) for c in range(n))]
        for r in range(n):
            rows.append(f'{r} ' + ' '.join(symbols[int(board[r, c])] for c in range(n)))
        p1 = int(np.sum(board == 1))
        p2 = int(np.sum(board == -1))
        rows.append(f'X:{p1} O:{p2}')
        return '\n'.join(rows)

    # --- Internal helpers ---

    def _get_flips(self, board: np.ndarray, row: int, col: int, player: int) -> list[tuple[int, int]]:
        """Return list of (row, col) positions that would be flipped by placing player at (row, col)."""
        n = self.size
        opponent = -player
        all_flips = []

        for dr, dc in DIRECTIONS:
            flips = []
            r, c = row + dr, col + dc
            while 0 <= r < n and 0 <= c < n and board[r, c] == opponent:
                flips.append((r, c))
                r += dr
                c += dc
            if flips and 0 <= r < n and 0 <= c < n and board[r, c] == player:
                all_flips.extend(flips)

        return all_flips

    def _has_any_move(self, board: np.ndarray, player: int) -> bool:
        """Check if player has any legal placement (not counting pass)."""
        n = self.size
        opponent = -player
        for r in range(n):
            for c in range(n):
                if board[r, c] != 0:
                    continue
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    found_opponent = False
                    while 0 <= nr < n and 0 <= nc < n and board[nr, nc] == opponent:
                        found_opponent = True
                        nr += dr
                        nc += dc
                    if found_opponent and 0 <= nr < n and 0 <= nc < n and board[nr, nc] == player:
                        return True
        return False
