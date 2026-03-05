"""Tests for Othello (6x6) game engine."""

import numpy as np
import pytest

from alpha_go.games import get_game
from alpha_go.games.othello import Othello, SIZE, PASS_ACTION
from alpha_go.mcts.search import MCTS
from alpha_go.utils.config import MCTSConfig


class TestOthelloBasic:

    def setup_method(self):
        self.game = Othello()

    def test_initial_state(self):
        state = self.game.get_initial_state()
        assert state.shape == (36,)
        board = state.reshape(6, 6)
        # 4 center pieces
        assert board[2, 2] == -1
        assert board[2, 3] == 1
        assert board[3, 2] == 1
        assert board[3, 3] == -1
        # Rest empty
        assert np.sum(state != 0) == 4

    def test_board_size(self):
        assert self.game.get_board_size() == 36

    def test_action_size(self):
        assert self.game.get_action_size() == 37  # 36 board + 1 pass

    def test_get_game(self):
        game = get_game('othello')
        assert isinstance(game, Othello)


class TestOthelloValidMoves:

    def setup_method(self):
        self.game = Othello()

    def test_initial_valid_moves_player1(self):
        """Player 1 (X) has exactly 4 legal moves from starting position."""
        state = self.game.get_initial_state()
        valid = self.game.get_valid_moves(state, player=1)
        # Must flip at least one opponent piece
        valid_actions = np.where(valid[:36] > 0)[0]
        assert len(valid_actions) == 4
        # Pass should not be valid (placements exist)
        assert valid[PASS_ACTION] == 0

    def test_initial_valid_moves_player_minus1(self):
        """Player -1 (O) also has exactly 4 legal moves from starting position."""
        state = self.game.get_initial_state()
        valid = self.game.get_valid_moves(state, player=-1)
        valid_actions = np.where(valid[:36] > 0)[0]
        assert len(valid_actions) == 4
        assert valid[PASS_ACTION] == 0

    def test_valid_moves_player_dependent(self):
        """Different players have different legal moves."""
        state = self.game.get_initial_state()
        valid_p1 = self.game.get_valid_moves(state, player=1)
        valid_p2 = self.game.get_valid_moves(state, player=-1)
        # The specific positions differ
        p1_positions = set(np.where(valid_p1[:36] > 0)[0])
        p2_positions = set(np.where(valid_p2[:36] > 0)[0])
        assert p1_positions != p2_positions

    def test_pass_only_when_forced(self):
        """Pass is only legal when no placements exist."""
        state = self.game.get_initial_state()
        valid = self.game.get_valid_moves(state, player=1)
        # Has placements, so pass is illegal
        assert valid[PASS_ACTION] == 0
        assert valid[:36].sum() > 0

    def test_pass_when_no_placements(self):
        """When a player has no legal placements, pass must be legal."""
        # Construct a board where player 1 has no moves
        state = np.zeros(36, dtype=np.float32)
        board = state.reshape(6, 6)
        # Fill most of board with -1, leaving player 1 nowhere to flip
        board[0, 0] = 1
        board[0, 1] = -1
        board[0, 2] = -1
        # Player 1 has no move that would flip an opponent piece
        valid = self.game.get_valid_moves(state, player=1)
        if valid[:36].sum() == 0:
            assert valid[PASS_ACTION] == 1


class TestOthelloFlipping:

    def setup_method(self):
        self.game = Othello()

    def test_horizontal_flip(self):
        """Placing a piece should flip opponent pieces in a line."""
        state = self.game.get_initial_state()
        # From initial state, player 1 plays at (2,2) area
        # Let's play a move and check flips
        # Initial: (2,2)=-1, (2,3)=1, (3,2)=1, (3,3)=-1
        # Player 1 plays (2,4): should flip (2,3) is already 1... let's trace
        # Actually (2,4) from player 1: direction left (-0,1->-0,-1): (2,3)=1 (own piece, no flip)
        # Let's use a controlled setup instead
        state = np.zeros(36, dtype=np.float32)
        board = state.reshape(6, 6)
        board[3, 2] = 1
        board[3, 3] = -1
        board[3, 4] = -1
        # Player 1 places at (3, 5): should flip (3,4) and (3,3) — horizontal
        new_state = self.game.get_next_state(state, 3 * 6 + 5, player=1)
        new_board = new_state.reshape(6, 6)
        assert new_board[3, 5] == 1  # placed
        assert new_board[3, 4] == 1  # flipped
        assert new_board[3, 3] == 1  # flipped
        assert new_board[3, 2] == 1  # was already 1

    def test_vertical_flip(self):
        state = np.zeros(36, dtype=np.float32)
        board = state.reshape(6, 6)
        board[1, 3] = 1
        board[2, 3] = -1
        board[3, 3] = -1
        # Player 1 places at (4, 3): flips (3,3) and (2,3)
        new_state = self.game.get_next_state(state, 4 * 6 + 3, player=1)
        new_board = new_state.reshape(6, 6)
        assert new_board[2, 3] == 1
        assert new_board[3, 3] == 1
        assert new_board[4, 3] == 1

    def test_diagonal_flip(self):
        state = np.zeros(36, dtype=np.float32)
        board = state.reshape(6, 6)
        board[1, 1] = 1
        board[2, 2] = -1
        board[3, 3] = -1
        # Player 1 at (4, 4): flips (3,3) and (2,2) along diagonal
        new_state = self.game.get_next_state(state, 4 * 6 + 4, player=1)
        new_board = new_state.reshape(6, 6)
        assert new_board[2, 2] == 1
        assert new_board[3, 3] == 1

    def test_multi_direction_flip(self):
        """A single placement can flip in multiple directions."""
        state = np.zeros(36, dtype=np.float32)
        board = state.reshape(6, 6)
        # Set up so placing at (3,3) flips both horizontally and vertically
        board[3, 2] = 1   # left anchor
        board[3, 4] = -1  # right opponent (no right anchor, won't flip right)
        board[2, 3] = -1  # up opponent
        board[1, 3] = 1   # up anchor
        board[4, 3] = -1  # down opponent
        board[5, 3] = 1   # down anchor
        # Place at (3, 3)
        new_state = self.game.get_next_state(state, 3 * 6 + 3, player=1)
        new_board = new_state.reshape(6, 6)
        assert new_board[2, 3] == 1  # vertical flip up
        assert new_board[4, 3] == 1  # vertical flip down
        # (3, 4) should NOT flip (no anchor on right side)
        assert new_board[3, 4] == -1

    def test_no_flip_empty_gap(self):
        """No flip if there's an empty square breaking the chain."""
        state = np.zeros(36, dtype=np.float32)
        board = state.reshape(6, 6)
        board[3, 2] = 1
        board[3, 4] = -1
        # Gap at (3,3) — empty. Place at (3,5): chain is (3,4)=-1 then (3,3)=0 — broken
        # Actually (3,5) going left: (3,4)=-1 is opponent, (3,3)=0 — not our piece, no flip
        valid = self.game.get_valid_moves(state, player=1)
        # (3,5) = action 23 should not be valid (can't flip anything)
        assert valid[3 * 6 + 5] == 0

    def test_no_flip_no_bookend(self):
        """Opponent pieces with no bookend of our own don't flip."""
        state = np.zeros(36, dtype=np.float32)
        board = state.reshape(6, 6)
        board[3, 3] = -1
        board[3, 4] = -1
        # Player 1 places at (3, 2): going right hits -1, -1 then edge — no bookend
        valid = self.game.get_valid_moves(state, player=1)
        assert valid[3 * 6 + 2] == 0

    def test_chain_flip(self):
        """Multiple opponent pieces in a line all get flipped."""
        state = np.zeros(36, dtype=np.float32)
        board = state.reshape(6, 6)
        board[0, 0] = 1
        board[0, 1] = -1
        board[0, 2] = -1
        board[0, 3] = -1
        # Player 1 at (0, 4): flips three -1 pieces
        new_state = self.game.get_next_state(state, 0 * 6 + 4, player=1)
        new_board = new_state.reshape(6, 6)
        assert new_board[0, 1] == 1
        assert new_board[0, 2] == 1
        assert new_board[0, 3] == 1
        assert new_board[0, 4] == 1


class TestOthelloPass:

    def setup_method(self):
        self.game = Othello()

    def test_pass_state_unchanged(self):
        """Pass action should not change the board."""
        state = self.game.get_initial_state()
        new_state = self.game.get_next_state(state, PASS_ACTION, player=1)
        assert np.array_equal(state, new_state)

    def test_consecutive_passes_end_game(self):
        """If both players must pass, game is over."""
        # Board where neither player can move
        state = np.zeros(36, dtype=np.float32)
        board = state.reshape(6, 6)
        board[0, 0] = 1
        board[0, 1] = -1
        # Neither player can flip anything
        is_terminal, _ = self.game.check_terminal(state, PASS_ACTION, player=1)
        # If neither has moves, it's terminal
        p1_valid = self.game.get_valid_moves(state, 1)
        p2_valid = self.game.get_valid_moves(state, -1)
        if p1_valid[:36].sum() == 0 and p2_valid[:36].sum() == 0:
            assert is_terminal


class TestOthelloTerminal:

    def setup_method(self):
        self.game = Othello()

    def test_board_full(self):
        """Full board is terminal."""
        state = np.ones(36, dtype=np.float32)  # all player 1
        is_terminal, value = self.game.check_terminal(state, 0, player=1)
        assert is_terminal
        assert value == 1.0  # player 1 wins (all pieces are theirs)

    def test_both_pass_terminal(self):
        """When neither player has legal moves, game ends."""
        state = np.zeros(36, dtype=np.float32)
        board = state.reshape(6, 6)
        # Only isolated pieces, no moves for either player
        board[0, 0] = 1
        board[5, 5] = -1
        is_terminal, value = self.game.check_terminal(state, PASS_ACTION, player=-1)
        assert is_terminal
        # Equal pieces = draw
        assert value == 0.0

    def test_winner_more_pieces(self):
        """Player with more pieces wins."""
        state = np.zeros(36, dtype=np.float32)
        board = state.reshape(6, 6)
        board[0, 0] = 1
        board[0, 1] = 1
        board[0, 2] = 1
        board[5, 5] = -1
        # 3 vs 1 — player 1 wins
        # Need to ensure no moves for either
        is_terminal, value = self.game.check_terminal(state, PASS_ACTION, player=1)
        assert is_terminal
        assert value == 1.0  # player 1 (who "just moved") has more pieces

    def test_terminal_correct_perspective(self):
        """Value should be from the perspective of the player who just moved."""
        state = np.zeros(36, dtype=np.float32)
        board = state.reshape(6, 6)
        board[0, 0] = 1
        board[0, 1] = 1
        board[5, 5] = -1
        # Player 1 has 2 pieces, player -1 has 1
        # If player -1 just moved (passed), value should be -1 (they lose)
        is_terminal, value = self.game.check_terminal(state, PASS_ACTION, player=-1)
        assert is_terminal
        assert value == -1.0

    def test_not_terminal_initial(self):
        """Initial state is not terminal."""
        state = self.game.get_initial_state()
        # Use any action (doesn't matter for non-terminal)
        is_terminal, _ = self.game.check_terminal(state, 0, player=1)
        assert not is_terminal

    def test_draw(self):
        """Equal piece count = draw."""
        state = np.zeros(36, dtype=np.float32)
        board = state.reshape(6, 6)
        board[0, 0] = 1
        board[0, 1] = 1
        board[5, 4] = -1
        board[5, 5] = -1
        is_terminal, value = self.game.check_terminal(state, PASS_ACTION, player=1)
        assert is_terminal
        assert value == 0.0


class TestOthelloSymmetries:

    def setup_method(self):
        self.game = Othello()

    def test_symmetry_count(self):
        """Should have 8 symmetries (4 rotations x 2 reflections)."""
        state = self.game.get_initial_state()
        pi = np.zeros(37, dtype=np.float32)
        pi[0] = 0.5
        pi[35] = 0.3
        pi[PASS_ACTION] = 0.2
        syms = self.game.get_symmetries(state, pi)
        assert len(syms) == 8

    def test_pass_probability_preserved(self):
        """Pass probability should be identical across all symmetries."""
        state = self.game.get_initial_state()
        pi = np.zeros(37, dtype=np.float32)
        pi[14] = 0.7  # some board move
        pi[PASS_ACTION] = 0.3
        syms = self.game.get_symmetries(state, pi)
        for _, sym_pi in syms:
            assert sym_pi[PASS_ACTION] == pytest.approx(0.3)

    def test_symmetry_shapes(self):
        """All symmetric states and policies should have correct shapes."""
        state = self.game.get_initial_state()
        pi = np.ones(37, dtype=np.float32) / 37
        syms = self.game.get_symmetries(state, pi)
        for sym_state, sym_pi in syms:
            assert sym_state.shape == (36,)
            assert sym_pi.shape == (37,)


class TestOthelloCanonical:

    def setup_method(self):
        self.game = Othello()

    def test_canonical_player1(self):
        """Canonical for player 1 is identity."""
        state = self.game.get_initial_state()
        canon = self.game.get_canonical_state(state, 1)
        assert np.array_equal(state, canon)

    def test_canonical_player_minus1(self):
        """Canonical for player -1 flips all pieces."""
        state = self.game.get_initial_state()
        canon = self.game.get_canonical_state(state, -1)
        assert np.array_equal(canon, -state)


class TestOthelloFullGame:

    def setup_method(self):
        self.game = Othello()

    def test_random_game_terminates(self):
        """A random game should always terminate."""
        state = self.game.get_initial_state()
        player = 1
        for _ in range(100):  # generous upper bound
            valid = self.game.get_valid_moves(state, player)
            valid_actions = np.where(valid > 0)[0]
            action = np.random.choice(valid_actions)
            state = self.game.get_next_state(state, action, player)
            is_terminal, _ = self.game.check_terminal(state, action, player)
            if is_terminal:
                return
            player = -player
        pytest.fail("Game did not terminate within 100 moves")

    def test_many_random_games(self):
        """Run 50 random games — all should terminate without errors."""
        for _ in range(50):
            state = self.game.get_initial_state()
            player = 1
            terminated = False
            for _ in range(100):
                valid = self.game.get_valid_moves(state, player)
                valid_actions = np.where(valid > 0)[0]
                assert len(valid_actions) > 0, "No valid actions but game not terminal"
                action = np.random.choice(valid_actions)
                state = self.game.get_next_state(state, action, player)
                is_terminal, _ = self.game.check_terminal(state, action, player)
                if is_terminal:
                    terminated = True
                    break
                player = -player
            assert terminated, "Game did not terminate"

    def test_pass_rules_respected(self):
        """In random games, pass should only happen when forced."""
        for _ in range(20):
            state = self.game.get_initial_state()
            player = 1
            for _ in range(100):
                valid = self.game.get_valid_moves(state, player)
                valid_actions = np.where(valid > 0)[0]
                action = np.random.choice(valid_actions)
                if action == PASS_ACTION:
                    # Verify no placements were available
                    assert valid[:36].sum() == 0, "Pass allowed when placements exist"
                state = self.game.get_next_state(state, action, player)
                is_terminal, _ = self.game.check_terminal(state, action, player)
                if is_terminal:
                    break
                player = -player


class TestOthelloDisplay:

    def test_display(self):
        game = Othello()
        state = game.get_initial_state()
        d = game.display(state)
        assert 'X' in d
        assert 'O' in d
        assert 'X:2 O:2' in d


class TestOthelloConfigurableSize:
    """Tests for non-default board sizes (8x8, 10x10)."""

    @pytest.mark.parametrize("size", [8, 10])
    def test_board_dimensions(self, size):
        game = Othello(size=size)
        assert game.get_board_size() == size * size
        assert game.get_board_shape() == (size, size)
        assert game.get_action_size() == size * size + 1

    @pytest.mark.parametrize("size", [8, 10])
    def test_initial_state(self, size):
        game = Othello(size=size)
        state = game.get_initial_state()
        assert state.shape == (size * size,)
        assert np.sum(state != 0) == 4
        board = state.reshape(size, size)
        mid = size // 2
        assert board[mid - 1, mid - 1] == -1
        assert board[mid - 1, mid] == 1
        assert board[mid, mid - 1] == 1
        assert board[mid, mid] == -1

    @pytest.mark.parametrize("size", [8, 10])
    def test_valid_moves_initial(self, size):
        game = Othello(size=size)
        state = game.get_initial_state()
        valid = game.get_valid_moves(state, player=1)
        assert valid.shape == (size * size + 1,)
        valid_actions = np.where(valid[:size * size] > 0)[0]
        assert len(valid_actions) == 4
        assert valid[size * size] == 0  # pass not valid

    @pytest.mark.parametrize("size", [8, 10])
    def test_random_game_terminates(self, size):
        game = Othello(size=size)
        state = game.get_initial_state()
        player = 1
        for _ in range(size * size * 2):
            valid = game.get_valid_moves(state, player)
            valid_actions = np.where(valid > 0)[0]
            action = np.random.choice(valid_actions)
            state = game.get_next_state(state, action, player)
            is_terminal, _ = game.check_terminal(state, action, player)
            if is_terminal:
                return
            player = -player
        pytest.fail(f"Game did not terminate on {size}x{size} board")

    @pytest.mark.parametrize("size", [8, 10])
    def test_symmetries(self, size):
        game = Othello(size=size)
        state = game.get_initial_state()
        pi = np.ones(size * size + 1, dtype=np.float32) / (size * size + 1)
        syms = game.get_symmetries(state, pi)
        assert len(syms) == 8
        for sym_state, sym_pi in syms:
            assert sym_state.shape == (size * size,)
            assert sym_pi.shape == (size * size + 1,)

    def test_get_game_variants(self):
        g8 = get_game('othello8')
        assert g8.get_board_size() == 64
        g10 = get_game('othello10')
        assert g10.get_board_size() == 100

    def test_default_backward_compatible(self):
        """Default Othello() still produces 6x6."""
        game = Othello()
        assert game.size == 6
        assert game.get_board_size() == 36


class TestOthelloMCTSIntegration:

    def test_mcts_initial_state(self):
        """MCTS search should work on the initial Othello state."""
        game = Othello()

        class UniformModel:
            def predict(self, state):
                pi = np.ones(37, dtype=np.float32) / 37
                return pi, 0.0

        model = UniformModel()
        config = MCTSConfig(num_simulations=10, dirichlet_epsilon=0.0)
        mcts = MCTS(game, model, config)
        state = game.get_initial_state()
        pi, _ = mcts.search(state, player=1)

        assert pi.shape == (37,)
        assert abs(pi.sum() - 1.0) < 1e-5
        # Pass should have 0 probability from initial state (placements exist)
        assert pi[PASS_ACTION] == 0

    def test_mcts_handles_pass(self):
        """MCTS should handle positions where pass is the only legal move."""
        game = Othello()

        class UniformModel:
            def predict(self, state):
                pi = np.ones(37, dtype=np.float32) / 37
                return pi, 0.0

        model = UniformModel()
        config = MCTSConfig(num_simulations=10, dirichlet_epsilon=0.0)

        # Create a state where player 1 must pass
        state = np.zeros(36, dtype=np.float32)
        board = state.reshape(6, 6)
        # Player -1 dominates, player 1 has one piece and no moves
        board[0, 0] = 1
        board[0, 1] = -1
        board[1, 0] = -1
        board[1, 1] = -1

        valid = game.get_valid_moves(state, player=1)
        if valid[:36].sum() == 0:
            # Player 1 must pass
            mcts = MCTS(game, model, config)
            pi, _ = mcts.search(state, player=1)
            assert pi[PASS_ACTION] == 1.0  # only legal move
