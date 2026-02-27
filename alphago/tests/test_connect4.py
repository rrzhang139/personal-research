"""Tests for Connect Four game engine."""

import numpy as np
import pytest

from alpha_go.games import get_game
from alpha_go.games.connect4 import ConnectFour, ROWS, COLS


class TestConnectFour:

    def setup_method(self):
        self.game = ConnectFour()

    def test_initial_state(self):
        state = self.game.get_initial_state()
        assert state.shape == (42,)
        assert np.all(state == 0)

    def test_board_size(self):
        assert self.game.get_board_size() == 42

    def test_action_size(self):
        assert self.game.get_action_size() == 7

    def test_valid_moves_initial(self):
        state = self.game.get_initial_state()
        valid = self.game.get_valid_moves(state)
        assert valid.shape == (7,)
        assert np.all(valid == 1)

    def test_drop_piece(self):
        """Pieces should drop to the bottom row."""
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state, 3, 1)  # drop in column 3
        board = state.reshape(ROWS, COLS)
        # Piece should be at bottom row (row 5), column 3
        assert board[5, 3] == 1
        # Rest of column 3 should be empty
        for r in range(5):
            assert board[r, 3] == 0

    def test_stacking(self):
        """Pieces stack on top of each other."""
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state, 3, 1)   # P1 drops in col 3
        state = self.game.get_next_state(state, 3, -1)  # P2 drops in col 3
        board = state.reshape(ROWS, COLS)
        assert board[5, 3] == 1   # P1 at bottom
        assert board[4, 3] == -1  # P2 on top

    def test_column_full(self):
        """A full column should be invalid."""
        state = self.game.get_initial_state()
        # Fill column 0 (6 pieces)
        for i in range(6):
            player = 1 if i % 2 == 0 else -1
            state = self.game.get_next_state(state, 0, player)
        valid = self.game.get_valid_moves(state)
        assert valid[0] == 0  # column 0 full
        assert valid[1] == 1  # column 1 still open

    def test_horizontal_win(self):
        """Four in a row horizontally."""
        state = self.game.get_initial_state()
        # P1 plays columns 0,1,2,3 on bottom row
        # P2 plays columns 0,1,2 on second row (to not block)
        for col in range(3):
            state = self.game.get_next_state(state, col, 1)
            state = self.game.get_next_state(state, col, -1)
        state = self.game.get_next_state(state, 3, 1)
        is_terminal, value = self.game.check_terminal(state, 3)
        assert is_terminal
        assert value == 1.0

    def test_vertical_win(self):
        """Four in a column."""
        state = self.game.get_initial_state()
        # P1 stacks 4 in column 2, P2 plays elsewhere
        for i in range(3):
            state = self.game.get_next_state(state, 2, 1)
            state = self.game.get_next_state(state, 3, -1)
        state = self.game.get_next_state(state, 2, 1)  # 4th piece
        is_terminal, value = self.game.check_terminal(state, 2)
        assert is_terminal
        assert value == 1.0

    def test_diagonal_win(self):
        """Four in a diagonal (bottom-left to top-right)."""
        state = self.game.get_initial_state()
        board = state.reshape(ROWS, COLS)
        # Manually set up a diagonal win for player 1
        board[5, 0] = 1
        board[4, 1] = 1
        board[3, 2] = 1
        board[2, 3] = 1
        # Add support pieces
        board[5, 1] = -1
        board[5, 2] = -1
        board[4, 2] = -1
        board[5, 3] = -1
        board[4, 3] = -1
        board[3, 3] = -1
        is_terminal, value = self.game.check_terminal(state, 3)  # last placed at col 3
        assert is_terminal
        assert value == 1.0

    def test_not_terminal(self):
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state, 3, 1)
        is_terminal, _ = self.game.check_terminal(state, 3)
        assert not is_terminal

    def test_draw(self):
        """Full board with no winner."""
        state = self.game.get_initial_state()
        board = state.reshape(ROWS, COLS)
        # Fill with a pattern that has no four-in-a-row
        # Pattern: alternating columns of 3
        pattern = [
            [1, -1,  1, -1,  1, -1,  1],
            [1, -1,  1, -1,  1, -1,  1],
            [1, -1,  1, -1,  1, -1,  1],
            [-1, 1, -1,  1, -1,  1, -1],
            [-1, 1, -1,  1, -1,  1, -1],
            [-1, 1, -1,  1, -1,  1, -1],
        ]
        for r in range(ROWS):
            for c in range(COLS):
                board[r, c] = pattern[r][c]
        # Verify no four-in-a-row exists by checking from the "last" move
        is_terminal, value = self.game.check_terminal(state, 6)
        assert is_terminal
        assert value == 0.0

    def test_canonical_state(self):
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state, 3, 1)
        canon = self.game.get_canonical_state(state, -1)
        board = canon.reshape(ROWS, COLS)
        assert board[5, 3] == -1  # flipped

    def test_symmetries(self):
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state, 0, 1)
        pi = np.zeros(7, dtype=np.float32)
        pi[0] = 1.0
        syms = self.game.get_symmetries(state, pi)
        assert len(syms) == 2  # original + mirror

        # Mirror: piece in col 0 → piece in col 6
        mirrored_state, mirrored_pi = syms[1]
        board = mirrored_state.reshape(ROWS, COLS)
        assert board[5, 6] == 1
        assert mirrored_pi[6] == 1.0

    def test_display(self):
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state, 3, 1)
        d = self.game.display(state)
        assert 'X' in d

    def test_get_game(self):
        game = get_game('connect4')
        assert isinstance(game, ConnectFour)


class TestConnectFourFullGame:
    """Test playing a complete random game."""

    def test_random_game_terminates(self):
        game = ConnectFour()
        state = game.get_initial_state()
        player = 1

        for _ in range(42):  # max 42 moves
            valid = game.get_valid_moves(state)
            valid_actions = np.where(valid > 0)[0]
            if len(valid_actions) == 0:
                break
            action = np.random.choice(valid_actions)
            state = game.get_next_state(state, action, player)
            is_terminal, _ = game.check_terminal(state, action)
            if is_terminal:
                return
            player = -player

    def test_many_random_games(self):
        """Run 50 random games — all should terminate without errors."""
        game = ConnectFour()
        for _ in range(50):
            state = game.get_initial_state()
            player = 1
            for _ in range(42):
                valid = game.get_valid_moves(state)
                valid_actions = np.where(valid > 0)[0]
                if len(valid_actions) == 0:
                    break
                action = np.random.choice(valid_actions)
                state = game.get_next_state(state, action, player)
                is_terminal, _ = game.check_terminal(state, action)
                if is_terminal:
                    break
                player = -player
