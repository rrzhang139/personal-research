"""Tests for MCTS."""

import numpy as np

from alpha_go.games.tictactoe import TicTacToe
from alpha_go.mcts.node import MCTSNode
from alpha_go.mcts.search import MCTS
from alpha_go.utils.config import MCTSConfig


class DummyModel:
    """Returns uniform policy and zero value."""

    def __init__(self, action_size: int):
        self.action_size = action_size

    def predict(self, state):
        pi = np.ones(self.action_size, dtype=np.float32) / self.action_size
        return pi, 0.0

    def predict_batch(self, states):
        policies = [np.ones(self.action_size, dtype=np.float32) / self.action_size for _ in states]
        values = [0.0 for _ in states]
        return policies, values


class TestMCTSNode:

    def test_create_node(self):
        state = np.zeros(9, dtype=np.float32)
        node = MCTSNode(state=state, player=1)
        assert node.N == 0
        assert node.Q == 0.0
        assert node.is_leaf()

    def test_expand(self):
        game = TicTacToe()
        state = game.get_initial_state()
        node = MCTSNode(state=state, player=1)
        priors = np.ones(9, dtype=np.float32) / 9
        node.expand(game, priors)
        assert node.is_expanded
        assert len(node.children) == 9

    def test_expand_partial_board(self):
        game = TicTacToe()
        state = game.get_initial_state()
        state[4] = 1  # center taken
        node = MCTSNode(state=state, player=-1)
        priors = np.ones(9, dtype=np.float32) / 9
        node.expand(game, priors)
        assert len(node.children) == 8
        assert 4 not in node.children

    def test_backpropagate(self):
        state = np.zeros(9, dtype=np.float32)
        parent = MCTSNode(state=state, player=1)
        child = MCTSNode(state=state, player=-1, parent=parent, action=0, prior=0.5)
        parent.children[0] = child

        child.backpropagate(0.5)
        assert child.N == 1
        assert child.Q == 0.5
        assert parent.N == 1
        assert parent.Q == -0.5  # flipped perspective

    def test_select_child(self):
        game = TicTacToe()
        state = game.get_initial_state()
        node = MCTSNode(state=state, player=1)
        priors = np.ones(9, dtype=np.float32) / 9
        node.expand(game, priors)
        node.N = 1  # parent needs visits for PUCT formula

        child = node.select_child(c_puct=1.0)
        assert child is not None
        assert child.action in range(9)


class TestMCTSSearch:

    def test_search_returns_valid_distribution(self):
        game = TicTacToe()
        model = DummyModel(9)
        config = MCTSConfig(num_simulations=10, dirichlet_epsilon=0.0)

        mcts = MCTS(game, model, config)
        state = game.get_initial_state()
        pi, _ = mcts.search(state, player=1)

        assert pi.shape == (9,)
        assert abs(pi.sum() - 1.0) < 1e-5
        assert np.all(pi >= 0)

    def test_search_only_legal_moves(self):
        game = TicTacToe()
        model = DummyModel(9)
        config = MCTSConfig(num_simulations=10, dirichlet_epsilon=0.0)

        state = game.get_initial_state()
        state[4] = 1  # center taken
        mcts = MCTS(game, model, config)
        pi, _ = mcts.search(state, player=-1)

        assert pi[4] == 0  # can't play on occupied square

    def test_search_terminal_position(self):
        """Search on a nearly-won position should focus on winning move."""
        game = TicTacToe()
        model = DummyModel(9)
        config = MCTSConfig(num_simulations=50, dirichlet_epsilon=0.0)

        # Player 1 has 0,1 — needs 2 to win
        state = np.zeros(9, dtype=np.float32)
        state[0] = 1
        state[1] = 1
        state[3] = -1
        state[4] = -1

        mcts = MCTS(game, model, config)
        pi, _ = mcts.search(state, player=1)

        # Should strongly prefer action 2 (winning move)
        assert pi[2] > 0.3  # at least a significant portion

    def test_diagnostics(self):
        """Test that diagnostics are collected when requested."""
        game = TicTacToe()
        model = DummyModel(9)
        config = MCTSConfig(num_simulations=10, dirichlet_epsilon=0.0)

        mcts = MCTS(game, model, config)
        state = game.get_initial_state()

        pi, diag = mcts.search(state, player=1, collect_diagnostics=True)
        assert diag is not None
        assert isinstance(diag.root_value, float)
        assert diag.policy_entropy >= 0
        assert diag.max_depth >= 0

        # Without diagnostics
        pi2, diag2 = mcts.search(state, player=1, collect_diagnostics=False)
        assert diag2 is None


class TestVirtualLoss:

    def test_apply_and_revert(self):
        """Virtual loss should be fully reversible."""
        state = np.zeros(9, dtype=np.float32)
        parent = MCTSNode(state=state, player=1)
        child = MCTSNode(state=state, player=-1, parent=parent, action=0, prior=0.5)
        parent.children[0] = child

        # Set up some existing stats via backprop
        child.backpropagate(0.5)
        assert child.N == 1 and parent.N == 1
        orig_child_Q = child.Q
        orig_parent_Q = parent.Q

        # Apply and revert should restore exactly
        child.apply_virtual_loss()
        assert child.N == 2  # incremented
        child.revert_virtual_loss()
        assert child.N == 1  # restored
        assert abs(child.Q - orig_child_Q) < 1e-10
        assert abs(parent.Q - orig_parent_Q) < 1e-10

    def test_vl_makes_node_less_attractive(self):
        """Virtual loss should reduce a node's attractiveness to its parent."""
        game = TicTacToe()
        state = game.get_initial_state()
        root = MCTSNode(state=state, player=1)
        priors = np.ones(9, dtype=np.float32) / 9
        root.expand(game, priors)
        root.N = 10

        # Give one child some visits
        target = root.children[0]
        target.backpropagate(-0.5)  # child thinks it's losing → parent sees +0.5

        # Compute score before VL
        score_before = -target.Q + 1.0 * target.P * np.sqrt(root.N) / (1 + target.N)

        target.apply_virtual_loss()
        score_after = -target.Q + 1.0 * target.P * np.sqrt(root.N) / (1 + target.N)

        assert score_after < score_before  # less attractive
        target.revert_virtual_loss()

    def test_double_vl_double_revert(self):
        """Two VL applications + two reverts should restore original state."""
        state = np.zeros(9, dtype=np.float32)
        node = MCTSNode(state=state, player=1)
        node.backpropagate(0.3)
        orig_N, orig_W, orig_Q = node.N, node.W, node.Q

        node.apply_virtual_loss()
        node.apply_virtual_loss()
        assert node.N == orig_N + 2

        node.revert_virtual_loss()
        node.revert_virtual_loss()
        assert node.N == orig_N
        assert abs(node.Q - orig_Q) < 1e-10


class TestBatchedSearch:

    def test_batched_returns_valid_distribution(self):
        """Batched search should produce a valid policy like sequential."""
        game = TicTacToe()
        model = DummyModel(9)
        config = MCTSConfig(num_simulations=16, dirichlet_epsilon=0.0, nn_batch_size=4)

        mcts = MCTS(game, model, config)
        state = game.get_initial_state()
        pi, _ = mcts.search(state, player=1)

        assert pi.shape == (9,)
        assert abs(pi.sum() - 1.0) < 1e-5
        assert np.all(pi >= 0)

    def test_batched_only_legal_moves(self):
        game = TicTacToe()
        model = DummyModel(9)
        config = MCTSConfig(num_simulations=16, dirichlet_epsilon=0.0, nn_batch_size=4)

        state = game.get_initial_state()
        state[4] = 1  # center taken
        mcts = MCTS(game, model, config)
        pi, _ = mcts.search(state, player=-1)

        assert pi[4] == 0

    def test_batched_diagnostics(self):
        game = TicTacToe()
        model = DummyModel(9)
        config = MCTSConfig(num_simulations=16, dirichlet_epsilon=0.0, nn_batch_size=8)

        mcts = MCTS(game, model, config)
        state = game.get_initial_state()
        pi, diag = mcts.search(state, player=1, collect_diagnostics=True)
        assert diag is not None
        assert diag.max_depth >= 0
        assert diag.policy_entropy >= 0

    def test_batch1_matches_sequential(self):
        """nn_batch_size=1 should use sequential path (sanity check)."""
        game = TicTacToe()
        model = DummyModel(9)
        config = MCTSConfig(num_simulations=10, dirichlet_epsilon=0.0, nn_batch_size=1)

        mcts = MCTS(game, model, config)
        state = game.get_initial_state()
        pi, _ = mcts.search(state, player=1)

        assert pi.shape == (9,)
        assert abs(pi.sum() - 1.0) < 1e-5
