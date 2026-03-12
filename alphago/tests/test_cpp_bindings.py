"""Cross-validate C++ GoGame against Python Go implementation.

Run: python -m pytest tests/test_cpp_bindings.py -x -v
"""

import numpy as np
import pytest

try:
    from mcts_cpp._mcts_cpp import GoGame as CppGoGame
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

from alpha_go.games.go import Go as PyGo

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ module not built")


def idx(size, r, c):
    return r * size + c


class TestGoGameCrossValidation:
    """Cross-validate every C++ GoGame method against Python Go."""

    def setup_method(self):
        self.py = PyGo(size=9)
        self.cpp = CppGoGame(9)

    def test_dimensions(self):
        assert self.cpp.size == self.py.size
        assert self.cpp.n2 == self.py.n2
        assert self.cpp.nn_input_size == self.py.nn_input_size
        assert self.cpp.pass_action == self.py.pass_action
        assert self.cpp.get_board_size() == self.py.get_board_size()
        assert self.cpp.get_action_size() == self.py.get_action_size()

    def test_initial_state(self):
        py_state = self.py.get_initial_state()
        cpp_state = np.asarray(self.cpp.get_initial_state())
        np.testing.assert_array_equal(cpp_state, py_state)

    def test_get_next_state_place_stone(self):
        py_state = self.py.get_initial_state()
        cpp_state = np.asarray(self.cpp.get_initial_state())

        action = idx(9, 4, 4)
        py_next = self.py.get_next_state(py_state, action, 1)
        cpp_next = np.asarray(self.cpp.get_next_state(cpp_state, action, 1))

        np.testing.assert_allclose(cpp_next, py_next, atol=1e-6)

    def test_get_next_state_pass(self):
        py_state = self.py.get_initial_state()
        cpp_state = np.asarray(self.cpp.get_initial_state())

        pass_action = self.py.pass_action
        py_next = self.py.get_next_state(py_state, pass_action, 1)
        cpp_next = np.asarray(self.cpp.get_next_state(cpp_state, pass_action, 1))

        np.testing.assert_allclose(cpp_next, py_next, atol=1e-6)

    def test_get_valid_moves(self):
        py_state = self.py.get_initial_state()
        cpp_state = np.asarray(self.cpp.get_initial_state())

        py_valid = self.py.get_valid_moves(py_state, 1)
        cpp_valid = np.asarray(self.cpp.get_valid_moves(cpp_state, 1))

        np.testing.assert_array_equal(cpp_valid, py_valid)

    def test_check_terminal_not_terminal(self):
        py_state = self.py.get_initial_state()
        cpp_state = np.asarray(self.cpp.get_initial_state())

        py_term, py_val = self.py.check_terminal(py_state, 0, 1)
        cpp_term, cpp_val = self.cpp.check_terminal(cpp_state, 0, 1)

        assert py_term == cpp_term
        assert py_val == pytest.approx(cpp_val)

    def test_check_terminal_double_pass(self):
        py_state = self.py.get_initial_state()
        cpp_state = np.asarray(self.cpp.get_initial_state())

        pass_a = self.py.pass_action
        py_s1 = self.py.get_next_state(py_state, pass_a, 1)
        py_s2 = self.py.get_next_state(py_s1, pass_a, -1)
        cpp_s1 = np.asarray(self.cpp.get_next_state(cpp_state, pass_a, 1))
        cpp_s2 = np.asarray(self.cpp.get_next_state(cpp_s1, pass_a, -1))

        py_term, py_val = self.py.check_terminal(py_s2, pass_a, -1)
        cpp_term, cpp_val = self.cpp.check_terminal(cpp_s2, pass_a, -1)

        assert py_term == cpp_term
        assert py_val == pytest.approx(cpp_val)

    def test_canonical_state_player1(self):
        py_state = self.py.get_initial_state()
        cpp_state = np.asarray(self.cpp.get_initial_state())

        # Place a stone first
        action = idx(9, 4, 4)
        py_next = self.py.get_next_state(py_state, action, 1)
        cpp_next = np.asarray(self.cpp.get_next_state(cpp_state, action, 1))

        py_canon = self.py.get_canonical_state(py_next, 1)
        cpp_canon = np.asarray(self.cpp.get_canonical_state(cpp_next, 1))

        np.testing.assert_allclose(cpp_canon, py_canon, atol=1e-6)

    def test_canonical_state_player_minus1(self):
        py_state = self.py.get_initial_state()
        cpp_state = np.asarray(self.cpp.get_initial_state())

        action = idx(9, 4, 4)
        py_next = self.py.get_next_state(py_state, action, 1)
        cpp_next = np.asarray(self.cpp.get_next_state(cpp_state, action, 1))

        py_canon = self.py.get_canonical_state(py_next, -1)
        cpp_canon = np.asarray(self.cpp.get_canonical_state(cpp_next, -1))

        np.testing.assert_allclose(cpp_canon, py_canon, atol=1e-6)

    def test_capture(self):
        """Single stone capture should match between C++ and Python."""
        py_state = self.py.get_initial_state()
        cpp_state = np.asarray(self.cpp.get_initial_state())

        # Build identical game sequence: surround and capture
        moves = [
            (idx(9, 3, 4), 1),   # Black
            (idx(9, 4, 4), -1),  # White at center
            (idx(9, 5, 4), 1),   # Black
            (idx(9, 0, 0), -1),  # White elsewhere
            (idx(9, 4, 3), 1),   # Black
            (idx(9, 0, 1), -1),  # White elsewhere
            (idx(9, 4, 5), 1),   # Black captures white at (4,4)
        ]

        for action, player in moves:
            py_state = self.py.get_next_state(py_state, action, player)
            cpp_state = np.asarray(self.cpp.get_next_state(cpp_state, action, player))

        np.testing.assert_allclose(cpp_state, py_state, atol=1e-6)

    def test_random_game_cross_validation(self):
        """Play a random game with both engines, states must match at every step."""
        rng = np.random.RandomState(42)
        py_state = self.py.get_initial_state()
        cpp_state = np.asarray(self.cpp.get_initial_state())
        player = 1

        for move_num in range(100):
            # Valid moves should match
            py_valid = self.py.get_valid_moves(py_state, player)
            cpp_valid = np.asarray(self.cpp.get_valid_moves(cpp_state, player))
            np.testing.assert_array_equal(
                cpp_valid, py_valid,
                err_msg=f"Valid moves differ at move {move_num}"
            )

            # Pick same random action
            valid_actions = np.where(py_valid > 0)[0]
            action = rng.choice(valid_actions)

            py_state = self.py.get_next_state(py_state, action, player)
            cpp_state = np.asarray(self.cpp.get_next_state(cpp_state, action, player))

            np.testing.assert_allclose(
                cpp_state, py_state, atol=1e-6,
                err_msg=f"States differ after move {move_num}, action={action}"
            )

            # Terminal check should match
            py_term, py_val = self.py.check_terminal(py_state, action, player)
            cpp_term, cpp_val = self.cpp.check_terminal(cpp_state, action, player)
            assert py_term == cpp_term, f"Terminal differs at move {move_num}"
            if py_term:
                assert py_val == pytest.approx(cpp_val)
                break

            player = -player

    def test_ko_detection(self):
        """Ko point should match between C++ and Python."""
        py_state = self.py.get_initial_state()
        cpp_state = np.asarray(self.cpp.get_initial_state())

        # Set up ko pattern:
        # Black at (0,1),(1,0),(2,1), White at (0,2),(1,1),(1,3),(2,2)
        # Then Black plays (1,2) to capture White (1,1)
        moves = [
            (idx(9, 0, 1), 1),
            (idx(9, 0, 2), -1),
            (idx(9, 1, 0), 1),
            (idx(9, 1, 1), -1),
            (idx(9, 2, 1), 1),
            (idx(9, 1, 3), -1),
            (idx(9, 8, 8), 1),   # Black tempo
            (idx(9, 2, 2), -1),
            (idx(9, 1, 2), 1),   # Black captures, ko at (1,1)
        ]

        for action, player in moves:
            py_state = self.py.get_next_state(py_state, action, player)
            cpp_state = np.asarray(self.cpp.get_next_state(cpp_state, action, player))

        np.testing.assert_allclose(cpp_state, py_state, atol=1e-6)

        # Valid moves should match (ko point banned for White)
        py_valid = self.py.get_valid_moves(py_state, -1)
        cpp_valid = np.asarray(self.cpp.get_valid_moves(cpp_state, -1))
        np.testing.assert_array_equal(cpp_valid, py_valid)

    def test_suicide_illegal(self):
        """Suicide move should be illegal in both engines."""
        py_state = self.py.get_initial_state()
        cpp_state = np.asarray(self.cpp.get_initial_state())

        # White at (0,1) and (1,0), black plays (0,0) is suicide
        moves = [
            (idx(9, 8, 8), 1),   # Black elsewhere
            (idx(9, 0, 1), -1),
            (idx(9, 8, 7), 1),   # Black elsewhere
            (idx(9, 1, 0), -1),
        ]
        for action, player in moves:
            py_state = self.py.get_next_state(py_state, action, player)
            cpp_state = np.asarray(self.cpp.get_next_state(cpp_state, action, player))

        py_valid = self.py.get_valid_moves(py_state, 1)
        cpp_valid = np.asarray(self.cpp.get_valid_moves(cpp_state, 1))
        np.testing.assert_array_equal(cpp_valid, py_valid)
        # (0,0) should be illegal
        assert py_valid[idx(9, 0, 0)] == 0

    def test_multiple_random_games(self):
        """Run 10 random games, cross-validating at every step."""
        for seed in range(10):
            rng = np.random.RandomState(seed + 100)
            py_state = self.py.get_initial_state()
            cpp_state = np.asarray(self.cpp.get_initial_state())
            player = 1

            for move_num in range(150):
                py_valid = self.py.get_valid_moves(py_state, player)
                cpp_valid = np.asarray(self.cpp.get_valid_moves(cpp_state, player))
                np.testing.assert_array_equal(
                    cpp_valid, py_valid,
                    err_msg=f"Seed {seed}, move {move_num}: valid moves differ"
                )

                valid_actions = np.where(py_valid > 0)[0]
                action = rng.choice(valid_actions)

                py_state = self.py.get_next_state(py_state, action, player)
                cpp_state = np.asarray(self.cpp.get_next_state(cpp_state, action, player))

                np.testing.assert_allclose(
                    cpp_state, py_state, atol=1e-6,
                    err_msg=f"Seed {seed}, move {move_num}: states differ"
                )

                py_term, py_val = self.py.check_terminal(py_state, action, player)
                cpp_term, cpp_val = self.cpp.check_terminal(cpp_state, action, player)
                assert py_term == cpp_term
                if py_term:
                    assert py_val == pytest.approx(cpp_val)
                    break

                player = -player
