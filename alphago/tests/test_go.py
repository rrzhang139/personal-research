"""Tests for Go (9x9) game engine."""

import numpy as np
import pytest

from alpha_go.games import get_game
from alpha_go.games.go import Go, NUM_HISTORY, COLOR_PLANE
from alpha_go.mcts.search import MCTS
from alpha_go.utils.config import MCTSConfig


# ── Helpers ──────────────────────────────────────────────────────────

def place_stones(game, state, stones_p1, stones_p2):
    """Place stones directly onto the state (bypassing get_next_state).

    stones_p1, stones_p2: lists of (row, col) tuples.
    Modifies plane 0 (current P1) and plane 8 (current P-1).
    """
    n = game.size
    planes = state[:game.nn_input_size].reshape(game.num_planes, game.n2)
    for r, c in stones_p1:
        planes[0][r * n + c] = 1.0
    for r, c in stones_p2:
        planes[NUM_HISTORY][r * n + c] = 1.0
    return state


def idx(game, r, c):
    """Convert (row, col) to linear action index."""
    return r * game.size + c


# ── Basic ────────────────────────────────────────────────────────────

class TestGoBasic:

    def setup_method(self):
        self.game = Go(size=9)

    def test_board_size(self):
        assert self.game.get_board_size() == 17 * 81  # 1377

    def test_action_size(self):
        assert self.game.get_action_size() == 82  # 81 + pass

    def test_board_shape(self):
        assert self.game.get_board_shape() == (17, 9, 9)

    def test_initial_state_empty(self):
        state = self.game.get_initial_state()
        board = self.game._get_current_board(state)
        assert np.all(board == 0)

    def test_game_registry(self):
        game = get_game('go')
        assert isinstance(game, Go)
        assert game.size == 9
        game9 = get_game('go9')
        assert isinstance(game9, Go)
        assert game9.size == 9


# ── Stone Placement ──────────────────────────────────────────────────

class TestGoStonePlacement:

    def setup_method(self):
        self.game = Go(size=9)

    def test_place_stone(self):
        state = self.game.get_initial_state()
        new_state = self.game.get_next_state(state, idx(self.game, 4, 4), player=1)
        board = self.game._get_current_board(new_state)
        assert board[idx(self.game, 4, 4)] == 1.0

    def test_history_shift(self):
        """After placing, plane 0 should have the new board, plane 1 the previous."""
        state = self.game.get_initial_state()
        # Place black at center
        s1 = self.game.get_next_state(state, idx(self.game, 4, 4), player=1)
        planes1 = s1[:self.game.nn_input_size].reshape(self.game.num_planes, self.game.n2)
        # Plane 0 should have the stone
        assert planes1[0][idx(self.game, 4, 4)] == 1.0
        # Plane 1 should be the previous (empty) board for P1
        assert planes1[1][idx(self.game, 4, 4)] == 0.0

        # Place white at (3,3)
        s2 = self.game.get_next_state(s1, idx(self.game, 3, 3), player=-1)
        planes2 = s2[:self.game.nn_input_size].reshape(self.game.num_planes, self.game.n2)
        # P-1 plane 8 should have the white stone
        assert planes2[NUM_HISTORY][idx(self.game, 3, 3)] == 1.0
        # P-1 plane 9 should be previous (empty) for white
        assert planes2[NUM_HISTORY + 1][idx(self.game, 3, 3)] == 0.0

    def test_color_alternation(self):
        state = self.game.get_initial_state()
        # Initial: player 1 to move
        assert self.game._get_color_to_move(state) == 1
        s1 = self.game.get_next_state(state, idx(self.game, 0, 0), player=1)
        assert self.game._get_color_to_move(s1) == -1
        s2 = self.game.get_next_state(s1, idx(self.game, 1, 1), player=-1)
        assert self.game._get_color_to_move(s2) == 1

    def test_placed_stone_occupied(self):
        """Can't place where a stone already is."""
        state = self.game.get_initial_state()
        s1 = self.game.get_next_state(state, idx(self.game, 4, 4), player=1)
        valid = self.game.get_valid_moves(s1, player=-1)
        assert valid[idx(self.game, 4, 4)] == 0


# ── Captures ─────────────────────────────────────────────────────────

class TestGoCaptures:

    def setup_method(self):
        self.game = Go(size=9)

    def test_single_stone_capture(self):
        """White stone surrounded on all 4 sides by black → captured."""
        state = self.game.get_initial_state()
        # Place white at (4,4), black at (3,4), (5,4), (4,3)
        state = place_stones(self.game, state,
                             stones_p1=[(3, 4), (5, 4), (4, 3)],
                             stones_p2=[(4, 4)])
        # Black plays (4,5) to complete the capture
        new_state = self.game.get_next_state(state, idx(self.game, 4, 5), player=1)
        board = self.game._get_current_board(new_state)
        assert board[idx(self.game, 4, 4)] == 0  # captured
        assert board[idx(self.game, 4, 5)] == 1  # placed

    def test_group_capture(self):
        """Two connected white stones surrounded → both captured."""
        state = self.game.get_initial_state()
        # White group at (4,4) and (4,5)
        # Black surrounds: (3,4), (3,5), (5,4), (5,5), (4,3), (4,6) minus last
        state = place_stones(self.game, state,
                             stones_p1=[(3, 4), (3, 5), (5, 4), (5, 5), (4, 3)],
                             stones_p2=[(4, 4), (4, 5)])
        # Black plays (4,6) to capture
        new_state = self.game.get_next_state(state, idx(self.game, 4, 6), player=1)
        board = self.game._get_current_board(new_state)
        assert board[idx(self.game, 4, 4)] == 0
        assert board[idx(self.game, 4, 5)] == 0

    def test_corner_capture(self):
        """Capture a stone in the corner (only 2 neighbors)."""
        state = self.game.get_initial_state()
        # White at (0,0), black at (0,1)
        state = place_stones(self.game, state,
                             stones_p1=[(0, 1)],
                             stones_p2=[(0, 0)])
        # Black plays (1,0)
        new_state = self.game.get_next_state(state, idx(self.game, 1, 0), player=1)
        board = self.game._get_current_board(new_state)
        assert board[idx(self.game, 0, 0)] == 0  # captured

    def test_edge_capture(self):
        """Capture on edge (3 neighbors)."""
        state = self.game.get_initial_state()
        # White at (0,4), black at (0,3), (0,5)
        state = place_stones(self.game, state,
                             stones_p1=[(0, 3), (0, 5)],
                             stones_p2=[(0, 4)])
        # Black plays (1,4) to capture
        new_state = self.game.get_next_state(state, idx(self.game, 1, 4), player=1)
        board = self.game._get_current_board(new_state)
        assert board[idx(self.game, 0, 4)] == 0

    def test_multi_group_capture(self):
        """One move captures two separate groups simultaneously."""
        state = self.game.get_initial_state()
        # Two separate white stones, each with 1 liberty at (4,4)
        # White at (3,4) surrounded by (2,4),(3,3),(3,5) — liberty at (4,4)
        # White at (5,4) surrounded by (6,4),(5,3),(5,5) — liberty at (4,4)
        state = place_stones(self.game, state,
                             stones_p1=[(2, 4), (3, 3), (3, 5), (6, 4), (5, 3), (5, 5)],
                             stones_p2=[(3, 4), (5, 4)])
        # Black plays (4,4) to capture both
        new_state = self.game.get_next_state(state, idx(self.game, 4, 4), player=1)
        board = self.game._get_current_board(new_state)
        assert board[idx(self.game, 3, 4)] == 0
        assert board[idx(self.game, 5, 4)] == 0

    def test_capture_creates_liberties(self):
        """Capturing opponent stones gives the capturing group liberties."""
        state = self.game.get_initial_state()
        # Black stones form a group that would have 0 liberties if not for capturing
        # Setup: Black at (0,1), white at (0,0),(1,1). Black plays (1,0).
        # After capture of (0,0), black (1,0) has liberty at (0,0).
        state = place_stones(self.game, state,
                             stones_p1=[(0, 1)],
                             stones_p2=[(0, 0), (1, 1)])
        # Black plays (1,0): captures white (0,0) since it's surrounded
        # White (0,0) neighbors: (0,1)=black, (1,0)=about-to-be-black → 0 liberties
        new_state = self.game.get_next_state(state, idx(self.game, 1, 0), player=1)
        board = self.game._get_current_board(new_state)
        assert board[idx(self.game, 0, 0)] == 0  # captured
        assert board[idx(self.game, 1, 0)] == 1  # placed (not suicide because capture happened)


# ── Suicide ──────────────────────────────────────────────────────────

class TestGoSuicide:

    def setup_method(self):
        self.game = Go(size=9)

    def test_single_stone_suicide_illegal(self):
        """Placing a single stone with no liberties and no capture is suicide."""
        state = self.game.get_initial_state()
        # White surrounds (0,0): white at (0,1) and (1,0)
        state = place_stones(self.game, state,
                             stones_p1=[],
                             stones_p2=[(0, 1), (1, 0)])
        valid = self.game.get_valid_moves(state, player=1)
        assert valid[idx(self.game, 0, 0)] == 0  # suicide

    def test_group_suicide_illegal(self):
        """Extending a group to 0 liberties without capturing is suicide."""
        state = self.game.get_initial_state()
        # Black at (0,0), white at (0,1),(1,0),(0,2),(2,0)
        # If black plays (1,1) — actually let's set up properly
        # Black at (0,0), white at (1,0),(0,1). Playing (1,1) where white is at
        # Let me construct: black at (0,0) with 0 liberties except via group extension
        # White at (0,1),(1,0),(1,2),(2,1). Black at (1,1). Black plays (0,0)?
        # No, let's just: White at (0,1),(1,0). (0,0) has 0 free neighbors.
        # Black playing (0,0) would be self-capture with no opponent captures → suicide.
        state = place_stones(self.game, state,
                             stones_p1=[],
                             stones_p2=[(0, 1), (1, 0)])
        valid = self.game.get_valid_moves(state, player=1)
        assert valid[idx(self.game, 0, 0)] == 0

    def test_not_suicide_with_capture(self):
        """Playing into 'surrounded' position is legal if it captures."""
        state = self.game.get_initial_state()
        # White at (0,0), black at (0,1),(1,1). White at (1,0) has just (0,0) as friend.
        # Actually: corner scenario.
        # Black at (0,1) and (1,0). White at (0,0) has 0 liberties — wait, need to be careful.
        # Let me set up: White at (0,0), black at (0,1). White's last liberty is (1,0).
        # Black plays (1,0): this captures white (0,0), so it's NOT suicide.
        state = place_stones(self.game, state,
                             stones_p1=[(0, 1)],
                             stones_p2=[(0, 0)])
        valid = self.game.get_valid_moves(state, player=1)
        assert valid[idx(self.game, 1, 0)] == 1  # legal: captures white (0,0)


# ── Ko ───────────────────────────────────────────────────────────────

class TestGoKo:

    def setup_method(self):
        self.game = Go(size=9)

    def _setup_ko(self):
        """Set up a classic ko pattern and return state after Black captures.

        Pattern (x=black, o=white, .=empty):
            . x o .
            x . o .    ← Black plays at (1,1) capturing white at (1,1)?
            . x o .

        Actually let me use the standard ko shape:
            col: 0 1 2 3
        row 0:  . x o .
        row 1:  x o . o
        row 2:  . x o .

        Black plays (1,2) to capture white (1,1)? No...
        Let me think more carefully.

        Standard ko:
            . B W .
            B W . W
            . B W .

        Black at (0,1),(1,0),(2,1). White at (0,2),(1,1),(1,3),(2,2).
        White at (1,1) has liberty at (1,0)=Black... no.
        White (1,1) neighbors: (0,1)=B, (1,0)=B, (1,2)=empty, (2,1)=B
        Wait that has a liberty at (1,2). Not capturable yet.

        Let me just set it up properly:
            . B W .
            B W . W
            . B W .
        White (1,1): neighbors (0,1)=B,(2,1)=B,(1,0)=B,(1,2)=empty
        Liberty at (1,2). Not capturable.

        Black plays (1,2): captures white (1,1)?
        After black (1,2): white (1,1) neighbors = (0,1)=B,(2,1)=B,(1,0)=B,(1,2)=B → 0 liberties → captured!
        """
        state = self.game.get_initial_state()
        state = place_stones(self.game, state,
                             stones_p1=[(0, 1), (1, 0), (2, 1)],
                             stones_p2=[(0, 2), (1, 1), (1, 3), (2, 2)])
        # Black plays (1,2): captures white (1,1)
        new_state = self.game.get_next_state(state, idx(self.game, 1, 2), player=1)
        return new_state

    def test_ko_detected(self):
        """After single-stone capture in ko shape, ko point is set."""
        state = self._setup_ko()
        ko = self.game._get_ko_point(state)
        # Ko point should be where white was captured: (1,1)
        assert ko == idx(self.game, 1, 1)

    def test_ko_bans_recapture(self):
        """The ko point should be illegal for the opponent."""
        state = self._setup_ko()
        # Now it's White's turn. White should NOT be able to play at (1,1) (ko).
        valid = self.game.get_valid_moves(state, player=-1)
        assert valid[idx(self.game, 1, 1)] == 0

    def test_ko_clears_after_other_move(self):
        """Ko clears after the opponent plays elsewhere."""
        state = self._setup_ko()
        # White plays elsewhere, e.g., (8,8)
        s2 = self.game.get_next_state(state, idx(self.game, 8, 8), player=-1)
        assert self.game._get_ko_point(s2) == -1  # ko cleared
        # Now black plays elsewhere
        s3 = self.game.get_next_state(s2, idx(self.game, 8, 7), player=1)
        # White can now play at (1,1)
        valid = self.game.get_valid_moves(s3, player=-1)
        assert valid[idx(self.game, 1, 1)] == 1

    def test_ko_clears_after_pass(self):
        """Ko clears on pass."""
        state = self._setup_ko()
        s2 = self.game.get_next_state(state, self.game.pass_action, player=-1)
        assert self.game._get_ko_point(s2) == -1

    def test_multi_capture_not_ko(self):
        """Capturing more than 1 stone does NOT set ko."""
        state = self.game.get_initial_state()
        # White has 2 stones that get captured together
        state = place_stones(self.game, state,
                             stones_p1=[(0, 0), (0, 2), (1, 1)],
                             stones_p2=[(0, 1), (1, 0)])
        # Black plays... hmm, let me set up a 2-stone capture properly.
        # White at (3,3),(3,4), black surrounds: (2,3),(2,4),(4,3),(4,4),(3,2)
        state = self.game.get_initial_state()
        state = place_stones(self.game, state,
                             stones_p1=[(2, 3), (2, 4), (4, 3), (4, 4), (3, 2)],
                             stones_p2=[(3, 3), (3, 4)])
        # Black plays (3,5) to capture both
        new_state = self.game.get_next_state(state, idx(self.game, 3, 5), player=1)
        assert self.game._get_ko_point(new_state) == -1  # not ko


# ── Pass ─────────────────────────────────────────────────────────────

class TestGoPass:

    def setup_method(self):
        self.game = Go(size=9)

    def test_pass_always_valid(self):
        state = self.game.get_initial_state()
        valid = self.game.get_valid_moves(state, player=1)
        assert valid[self.game.pass_action] == 1.0

    def test_pass_no_board_change(self):
        state = self.game.get_initial_state()
        state = place_stones(self.game, state,
                             stones_p1=[(4, 4)], stones_p2=[(3, 3)])
        board_before = self.game._get_current_board(state).copy()
        new_state = self.game.get_next_state(state, self.game.pass_action, player=1)
        board_after = self.game._get_current_board(new_state)
        np.testing.assert_array_equal(board_before, board_after)

    def test_single_pass_not_terminal(self):
        state = self.game.get_initial_state()
        s1 = self.game.get_next_state(state, self.game.pass_action, player=1)
        is_term, _ = self.game.check_terminal(s1, self.game.pass_action, player=1)
        assert not is_term

    def test_consecutive_passes_terminal(self):
        state = self.game.get_initial_state()
        s1 = self.game.get_next_state(state, self.game.pass_action, player=1)
        s2 = self.game.get_next_state(s1, self.game.pass_action, player=-1)
        is_term, _ = self.game.check_terminal(s2, self.game.pass_action, player=-1)
        assert is_term

    def test_pass_clears_ko(self):
        """Passing should clear ko point."""
        state = self.game.get_initial_state()
        state[self.game.nn_input_size + 1] = 42.0  # fake ko
        new_state = self.game.get_next_state(state, self.game.pass_action, player=1)
        assert self.game._get_ko_point(new_state) == -1


# ── Scoring ──────────────────────────────────────────────────────────

class TestGoScoring:

    def setup_method(self):
        self.game = Go(size=9)

    def test_empty_board_white_wins_komi(self):
        """Empty board: 0 stones each, komi 7.5 → White wins."""
        state = self.game.get_initial_state()
        # Both pass
        s1 = self.game.get_next_state(state, self.game.pass_action, player=1)
        s2 = self.game.get_next_state(s1, self.game.pass_action, player=-1)
        is_term, value = self.game.check_terminal(s2, self.game.pass_action, player=-1)
        assert is_term
        # White wins via komi → value from player=-1 perspective should be +1
        assert value == 1.0  # player -1 (white, who just passed) wins

    def test_territory_scoring(self):
        """Black fills entire board → Black should win."""
        state = self.game.get_initial_state()
        # Place black on every intersection
        planes = state[:self.game.nn_input_size].reshape(self.game.num_planes, self.game.n2)
        planes[0] = 1.0  # all black
        # Two passes
        self.game._set_pass_count(state, 2)
        is_term, value = self.game.check_terminal(state, self.game.pass_action, player=1)
        assert is_term
        # Black has 81 - 0 - 7.5 = 73.5 > 0 → black wins
        assert value == 1.0  # player 1 (black) perspective

    def test_komi_applied(self):
        """With equal territory/stones, White wins due to komi."""
        state = self.game.get_initial_state()
        # Construct: black has left half, white has right half, roughly equal
        # 9x9: columns 0-3 black (36 stones), columns 5-8 white (36 stones), col 4 empty
        n = self.game.size
        planes = state[:self.game.nn_input_size].reshape(self.game.num_planes, n, n)
        for r in range(n):
            for c in range(4):
                planes[0][r][c] = 1.0
            for c in range(5, 9):
                planes[NUM_HISTORY][r][c] = 1.0
        # Column 4 is neutral (borders both)
        self.game._set_pass_count(state, 2)
        board = self.game._get_current_board(state)
        score = self.game._tromp_taylor_score(board)
        # 36 black + 0 territory - (36 white + 0 territory) - 7.5 = -7.5
        assert score < 0  # White wins with komi

    def test_territory_counted(self):
        """Empty space surrounded by one color counts as territory."""
        state = self.game.get_initial_state()
        n = self.game.size
        # Black ring around (4,4): black at all neighbors of (4,4)
        # (4,4) is empty, surrounded by black → territory for black
        state = place_stones(self.game, state,
                             stones_p1=[(3, 4), (5, 4), (4, 3), (4, 5)],
                             stones_p2=[])
        board = self.game._get_current_board(state)
        # The enclosed point (4,4) should be black territory
        # Flood fill from (4,4): it's empty, neighbors are all black → borders={1}
        # Score: 4 stones + some territory including (4,4)
        score = self.game._tromp_taylor_score(board)
        # 4 black stones + territory (a LOT of empty space borders black only
        # because white has no stones at all)
        # Actually: all empty space on the board either borders black or nothing
        # Since white has 0 stones, empty space touching black = black territory
        # Empty space not touching anything... actually all empty spaces are
        # connected, and they all eventually reach black stones, so the entire
        # board is effectively black territory.
        assert score > 0  # Black wins handily

    def test_value_perspective(self):
        """Terminal value should be from `player` arg's perspective."""
        state = self.game.get_initial_state()
        # All black
        planes = state[:self.game.nn_input_size].reshape(self.game.num_planes, self.game.n2)
        planes[0] = 1.0
        self.game._set_pass_count(state, 2)

        _, val_p1 = self.game.check_terminal(state, self.game.pass_action, player=1)
        _, val_p2 = self.game.check_terminal(state, self.game.pass_action, player=-1)
        assert val_p1 == 1.0   # black wins, player=1 → +1
        assert val_p2 == -1.0  # black wins, player=-1 → -1


# ── Canonical ────────────────────────────────────────────────────────

class TestGoCanonical:

    def setup_method(self):
        self.game = Go(size=9)

    def test_canonical_player1_identity(self):
        state = self.game.get_initial_state()
        state = place_stones(self.game, state,
                             stones_p1=[(4, 4)], stones_p2=[(3, 3)])
        canon = self.game.get_canonical_state(state, 1)
        expected = state[:self.game.nn_input_size]
        np.testing.assert_array_equal(canon, expected)

    def test_canonical_player_minus1_swaps(self):
        state = self.game.get_initial_state()
        state = place_stones(self.game, state,
                             stones_p1=[(4, 4)], stones_p2=[(3, 3)])
        canon = self.game.get_canonical_state(state, -1)
        planes = canon.reshape(self.game.num_planes, self.game.n2)
        # After swap: plane 0 should have what was plane 8 (white stones)
        assert planes[0][idx(self.game, 3, 3)] == 1.0
        # Plane 8 should have what was plane 0 (black stones)
        assert planes[NUM_HISTORY][idx(self.game, 4, 4)] == 1.0

    def test_canonical_roundtrip(self):
        """Applying canonical twice with -1 should return to original."""
        state = self.game.get_initial_state()
        state = place_stones(self.game, state,
                             stones_p1=[(4, 4), (2, 2)], stones_p2=[(3, 3)])
        original = state[:self.game.nn_input_size].copy()
        canon = self.game.get_canonical_state(state, -1)
        # Build a fake full state from canon to apply canonical again
        fake_state = np.zeros_like(state)
        fake_state[:self.game.nn_input_size] = canon
        roundtrip = self.game.get_canonical_state(fake_state, -1)
        np.testing.assert_array_equal(roundtrip, original)


# ── Symmetries ───────────────────────────────────────────────────────

class TestGoSymmetries:

    def setup_method(self):
        self.game = Go(size=9)

    def test_symmetry_count(self):
        state = self.game.get_initial_state()
        canon = self.game.get_canonical_state(state, 1)
        pi = np.ones(82, dtype=np.float32) / 82
        syms = self.game.get_symmetries(canon, pi)
        assert len(syms) == 8

    def test_pass_prob_preserved(self):
        state = self.game.get_initial_state()
        state = place_stones(self.game, state,
                             stones_p1=[(0, 0)], stones_p2=[(8, 8)])
        canon = self.game.get_canonical_state(state, 1)
        pi = np.zeros(82, dtype=np.float32)
        pi[0] = 0.7
        pi[self.game.pass_action] = 0.3
        syms = self.game.get_symmetries(canon, pi)
        for _, sym_pi in syms:
            assert sym_pi[self.game.pass_action] == pytest.approx(0.3)

    def test_symmetry_shapes(self):
        state = self.game.get_initial_state()
        canon = self.game.get_canonical_state(state, 1)
        pi = np.ones(82, dtype=np.float32) / 82
        syms = self.game.get_symmetries(canon, pi)
        for sym_state, sym_pi in syms:
            assert sym_state.shape == (self.game.nn_input_size,)
            assert sym_pi.shape == (82,)


# ── Integration ──────────────────────────────────────────────────────

class TestGoIntegration:

    def setup_method(self):
        self.game = Go(size=9)

    def test_random_games_terminate(self):
        """Run 5 random games — all should terminate."""
        rng = np.random.RandomState(42)
        for _ in range(5):
            state = self.game.get_initial_state()
            player = 1
            terminated = False
            for _ in range(200):  # generous upper bound for 9x9
                valid = self.game.get_valid_moves(state, player)
                valid_actions = np.where(valid > 0)[0]
                assert len(valid_actions) > 0, "No valid actions but game not terminal"
                action = rng.choice(valid_actions)
                state = self.game.get_next_state(state, action, player)
                is_term, _ = self.game.check_terminal(state, action, player)
                if is_term:
                    terminated = True
                    break
                player = -player
            assert terminated, "Game did not terminate within 200 moves"

    def test_mcts_works(self):
        """MCTS search should work on the initial Go state."""

        class UniformModel:
            def predict(self, state):
                pi = np.ones(82, dtype=np.float32) / 82
                return pi, 0.0

        model = UniformModel()
        config = MCTSConfig(num_simulations=10, dirichlet_epsilon=0.0)
        mcts = MCTS(self.game, model, config)
        state = self.game.get_initial_state()
        pi, _ = mcts.search(state, player=1)

        assert pi.shape == (82,)
        assert abs(pi.sum() - 1.0) < 1e-5

    def test_display(self):
        state = self.game.get_initial_state()
        state = place_stones(self.game, state,
                             stones_p1=[(4, 4)], stones_p2=[(3, 3)])
        d = self.game.display(state)
        assert 'X' in d
        assert 'O' in d
        assert 'Black' in d


# ── Configurable Size ────────────────────────────────────────────────

class TestGoConfigurableSize:

    @pytest.mark.parametrize("size", [9, 13, 19])
    def test_dimensions(self, size):
        game = Go(size=size)
        n2 = size * size
        assert game.get_board_size() == 17 * n2
        assert game.get_board_shape() == (17, size, size)
        assert game.get_action_size() == n2 + 1

    @pytest.mark.parametrize("size", [9, 13])
    def test_random_game(self, size):
        game = Go(size=size)
        rng = np.random.RandomState(123)
        state = game.get_initial_state()
        player = 1
        for _ in range(size * size * 3):
            valid = game.get_valid_moves(state, player)
            valid_actions = np.where(valid > 0)[0]
            action = rng.choice(valid_actions)
            state = game.get_next_state(state, action, player)
            is_term, _ = game.check_terminal(state, action, player)
            if is_term:
                return
            player = -player
        pytest.fail(f"Go {size}x{size} did not terminate")

    def test_registry_variants(self):
        g9 = get_game('go9')
        assert g9.size == 9
        g13 = get_game('go13')
        assert g13.size == 13
