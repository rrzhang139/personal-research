"""Othello (Reversi) implementation — 6x6 board.

6x6 board for fast iteration. Pieces flip opponent discs when sandwiched.
- 0 = empty, 1 = player 1 (X/Black), -1 = player -1 (O/White)
- Actions 0-35 = board positions (row * 6 + col). Action 36 = pass.
- A player must pass when they have no legal placements.
- Game ends when both players must pass (or board is full).
- Winner = player with more pieces.

Board layout:
  0  1  2  3  4  5
  6  7  8  9 10 11
 12 13 14 15 16 17
 18 19 20 21 22 23
 24 25 26 27 28 29
 30 31 32 33 34 35

Initial state (center 4 squares):
  . . . . . .
  . . . . . .
  . . O X . .
  . . X O . .
  . . . . . .
  . . . . . .
"""

from __future__ import annotations

import numpy as np

from .base_game import Game

SIZE = 6
PASS_ACTION = SIZE * SIZE  # 36

# 8 directions: (row_delta, col_delta)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]


class Othello(Game):

    def get_initial_state(self) -> np.ndarray:
        state = np.zeros(SIZE * SIZE, dtype=np.float32)
        mid = SIZE // 2
        # Standard Othello opening: white on main diagonal, black on anti-diagonal
        state[(mid - 1) * SIZE + (mid - 1)] = -1  # (2,2) = O
        state[(mid - 1) * SIZE + mid] = 1          # (2,3) = X
        state[mid * SIZE + (mid - 1)] = 1           # (3,2) = X
        state[mid * SIZE + mid] = -1                # (3,3) = O
        return state

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        new_state = state.copy()
        if action == PASS_ACTION:
            return new_state  # pass: board unchanged

        board = new_state.reshape(SIZE, SIZE)
        row, col = divmod(action, SIZE)
        board[row, col] = player

        # Flip captured pieces
        for flip_r, flip_c in self._get_flips(board, row, col, player):
            board[flip_r, flip_c] = player

        return new_state

    def get_valid_moves(self, state: np.ndarray, player: int = 1) -> np.ndarray:
        board = state.reshape(SIZE, SIZE)
        valid = np.zeros(SIZE * SIZE + 1, dtype=np.float32)

        for r in range(SIZE):
            for c in range(SIZE):
                if board[r, c] == 0 and self._get_flips(board, r, c, player):
                    valid[r * SIZE + c] = 1.0

        # Pass is legal only when no placements exist
        if valid[:SIZE * SIZE].sum() == 0:
            valid[PASS_ACTION] = 1.0

        return valid

    def check_terminal(self, state: np.ndarray, action: int, player: int = 1) -> tuple[bool, float]:
        board = state.reshape(SIZE, SIZE)

        # Check if both players must pass
        p1_has_moves = self._has_any_move(board, 1)
        p2_has_moves = self._has_any_move(board, -1)
        board_full = np.all(board != 0)

        if not (p1_has_moves or p2_has_moves) or board_full:
            # Game over — count pieces
            p1_count = np.sum(board == 1)
            p2_count = np.sum(board == -1)

            if p1_count > p2_count:
                winner = 1
            elif p2_count > p1_count:
                winner = -1
            else:
                return True, 0.0  # draw

            # Return value from perspective of player who just moved
            return True, 1.0 if winner == player else -1.0

        return False, 0.0

    def get_board_size(self) -> int:
        return SIZE * SIZE  # 36

    def get_board_shape(self) -> tuple[int, int]:
        return (SIZE, SIZE)  # (6, 6)

    def get_action_size(self) -> int:
        return SIZE * SIZE + 1  # 37 (36 board + 1 pass)

    def get_canonical_state(self, state: np.ndarray, player: int) -> np.ndarray:
        return state * player

    def get_symmetries(self, state: np.ndarray, pi: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """8 symmetries: 4 rotations x 2 reflections.

        The pass action probability is preserved across all symmetries.
        """
        symmetries = []
        board = state.reshape(SIZE, SIZE)
        board_pi = pi[:SIZE * SIZE].reshape(SIZE, SIZE)
        pass_prob = pi[PASS_ACTION]

        for rotation in range(4):
            rb = np.rot90(board, rotation)
            rp = np.rot90(board_pi, rotation)
            sym_pi = np.append(rp.flatten(), pass_prob)
            symmetries.append((rb.flatten(), sym_pi))
            # Add horizontal flip
            fb = np.fliplr(rb)
            fp = np.fliplr(rp)
            sym_pi_flip = np.append(fp.flatten(), pass_prob)
            symmetries.append((fb.flatten(), sym_pi_flip))

        return symmetries

    def display(self, state: np.ndarray) -> str:
        symbols = {0: '.', 1: 'X', -1: 'O'}
        board = state.reshape(SIZE, SIZE)
        rows = ['  ' + ' '.join(str(c) for c in range(SIZE))]
        for r in range(SIZE):
            rows.append(f'{r} ' + ' '.join(symbols[int(board[r, c])] for c in range(SIZE)))
        p1 = int(np.sum(board == 1))
        p2 = int(np.sum(board == -1))
        rows.append(f'X:{p1} O:{p2}')
        return '\n'.join(rows)

    # --- Internal helpers ---

    @staticmethod
    def _get_flips(board: np.ndarray, row: int, col: int, player: int) -> list[tuple[int, int]]:
        """Return list of (row, col) positions that would be flipped by placing player at (row, col)."""
        opponent = -player
        all_flips = []

        for dr, dc in DIRECTIONS:
            flips = []
            r, c = row + dr, col + dc
            while 0 <= r < SIZE and 0 <= c < SIZE and board[r, c] == opponent:
                flips.append((r, c))
                r += dr
                c += dc
            # Must end on our own piece to capture
            if flips and 0 <= r < SIZE and 0 <= c < SIZE and board[r, c] == player:
                all_flips.extend(flips)

        return all_flips

    @staticmethod
    def _has_any_move(board: np.ndarray, player: int) -> bool:
        """Check if player has any legal placement (not counting pass)."""
        opponent = -player
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r, c] != 0:
                    continue
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    found_opponent = False
                    while 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr, nc] == opponent:
                        found_opponent = True
                        nr += dr
                        nc += dc
                    if found_opponent and 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr, nc] == player:
                        return True
        return False
