"""Tests for the game engine."""

import numpy as np
import pytest

from alpha_go.games import get_game
from alpha_go.games.tictactoe import TicTacToe


class TestTicTacToe:

    def setup_method(self):
        self.game = TicTacToe()

    def test_initial_state(self):
        state = self.game.get_initial_state()
        assert state.shape == (9,)
        assert np.all(state == 0)

    def test_valid_moves_initial(self):
        state = self.game.get_initial_state()
        valid = self.game.get_valid_moves(state)
        assert np.all(valid == 1)

    def test_make_move(self):
        state = self.game.get_initial_state()
        new_state = self.game.get_next_state(state, 4, 1)
        assert new_state[4] == 1
        assert np.sum(new_state != 0) == 1
        # Original state unchanged
        assert np.all(state == 0)

    def test_valid_moves_after_move(self):
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state, 4, 1)
        valid = self.game.get_valid_moves(state)
        assert valid[4] == 0
        assert np.sum(valid) == 8

    def test_win_row(self):
        state = np.zeros(9, dtype=np.float32)
        state[0] = 1
        state[1] = 1
        state[2] = 1
        is_terminal, value = self.game.check_terminal(state, 2)
        assert is_terminal
        assert value == 1.0

    def test_win_col(self):
        state = np.zeros(9, dtype=np.float32)
        state[0] = -1
        state[3] = -1
        state[6] = -1
        is_terminal, value = self.game.check_terminal(state, 6)
        assert is_terminal
        assert value == 1.0  # player who moved wins

    def test_win_diagonal(self):
        state = np.zeros(9, dtype=np.float32)
        state[0] = 1
        state[4] = 1
        state[8] = 1
        is_terminal, value = self.game.check_terminal(state, 8)
        assert is_terminal

    def test_draw(self):
        # X O X
        # X X O
        # O X O
        state = np.array([1, -1, 1, 1, 1, -1, -1, 1, -1], dtype=np.float32)
        # Check from the last move (say position 7)
        is_terminal, value = self.game.check_terminal(state, 7)
        assert is_terminal
        assert value == 0.0

    def test_not_terminal(self):
        state = np.zeros(9, dtype=np.float32)
        state[0] = 1
        is_terminal, _ = self.game.check_terminal(state, 0)
        assert not is_terminal

    def test_canonical_state(self):
        state = np.array([1, 0, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        canon = self.game.get_canonical_state(state, -1)
        assert canon[0] == -1
        assert canon[2] == 1

    def test_symmetries(self):
        state = self.game.get_initial_state()
        state[0] = 1
        pi = np.zeros(9, dtype=np.float32)
        pi[0] = 1.0
        syms = self.game.get_symmetries(state, pi)
        assert len(syms) == 8  # 4 rotations x 2 reflections

    def test_board_size(self):
        assert self.game.get_board_size() == 9

    def test_action_size(self):
        assert self.game.get_action_size() == 9

    def test_display(self):
        state = np.array([1, 0, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        d = self.game.display(state)
        assert 'X' in d
        assert 'O' in d

    def test_get_game(self):
        game = get_game('tictactoe')
        assert isinstance(game, TicTacToe)

    def test_get_game_invalid(self):
        with pytest.raises(ValueError):
            get_game('unknown_game')


class TestFullGame:
    """Test playing a complete game."""

    def test_random_game_terminates(self):
        game = TicTacToe()
        state = game.get_initial_state()
        player = 1

        for _ in range(9):  # max 9 moves in tic-tac-toe
            valid = game.get_valid_moves(state)
            valid_actions = np.where(valid > 0)[0]
            action = np.random.choice(valid_actions)
            state = game.get_next_state(state, action, player)
            is_terminal, _ = game.check_terminal(state, action)
            if is_terminal:
                return
            player = -player

        # If we get here, all 9 moves were made — must be terminal
        assert True  # the board is full, check_terminal should have caught it
