"""Tests for the neural network."""

import numpy as np
import os
import tempfile

import pytest

from alpha_go.neural_net.conv_net import ConvNetWrapper
from alpha_go.neural_net.othello_net import OthelloNetWrapper
from alpha_go.neural_net.simple_net import SimpleNetWrapper
from alpha_go.neural_net import create_model
from alpha_go.utils.config import NetworkConfig


class TestSimpleNet:

    def setup_method(self):
        self.config = NetworkConfig(hidden_size=32, num_layers=2)
        self.model = SimpleNetWrapper(
            board_size=9, action_size=9, config=self.config, lr=0.001
        )

    def test_predict_shape(self):
        state = np.zeros(9, dtype=np.float32)
        pi, v = self.model.predict(state)
        assert pi.shape == (9,)
        assert abs(pi.sum() - 1.0) < 1e-5
        assert -1 <= v <= 1

    def test_predict_nonzero_input(self):
        state = np.array([1, 0, -1, 0, 1, 0, -1, 0, 0], dtype=np.float32)
        pi, v = self.model.predict(state)
        assert pi.shape == (9,)
        assert abs(pi.sum() - 1.0) < 1e-5

    def test_train_step(self):
        states = np.random.randn(16, 9).astype(np.float32)
        pis = np.ones((16, 9), dtype=np.float32) / 9
        vs = np.random.randn(16).astype(np.float32).clip(-1, 1)

        losses = self.model.train_step(states, pis, vs)
        assert 'total_loss' in losses
        assert 'policy_loss' in losses
        assert 'value_loss' in losses
        assert losses['total_loss'] > 0

    def test_train_reduces_loss(self):
        """Training on consistent data should reduce loss."""
        state = np.zeros((1, 9), dtype=np.float32)
        pi = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.float32)  # always center
        v = np.array([0.5], dtype=np.float32)

        # Repeat to make a batch
        states = np.tile(state, (32, 1))
        pis = np.tile(pi, (32, 1))
        vs = np.tile(v, 32)

        loss1 = self.model.train_step(states, pis, vs)['total_loss']
        for _ in range(50):
            self.model.train_step(states, pis, vs)
        loss2 = self.model.train_step(states, pis, vs)['total_loss']

        assert loss2 < loss1

    def test_save_load(self):
        state = np.zeros(9, dtype=np.float32)
        pi1, v1 = self.model.predict(state)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            self.model.save(path)
            new_model = SimpleNetWrapper(
                board_size=9, action_size=9, config=self.config, lr=0.001
            )
            new_model.load(path)
            pi2, v2 = new_model.predict(state)
            np.testing.assert_allclose(pi1, pi2, atol=1e-6)
            assert abs(v1 - v2) < 1e-6
        finally:
            os.unlink(path)

    def test_clone(self):
        state = np.zeros(9, dtype=np.float32)
        pi1, v1 = self.model.predict(state)

        cloned = self.model.clone()
        pi2, v2 = cloned.predict(state)

        np.testing.assert_allclose(pi1, pi2, atol=1e-6)
        assert abs(v1 - v2) < 1e-6

        # Verify they're independent (modifying clone doesn't affect original)
        dummy_states = np.random.randn(16, 9).astype(np.float32)
        dummy_pis = np.ones((16, 9), dtype=np.float32) / 9
        dummy_vs = np.zeros(16, dtype=np.float32)
        cloned.train_step(dummy_states, dummy_pis, dummy_vs)

        pi3, v3 = self.model.predict(state)
        np.testing.assert_allclose(pi1, pi3, atol=1e-6)


class TestConvNet:
    """Tests for ConvNet — mirrors TestSimpleNet but with CNN architecture."""

    def setup_method(self):
        # Square board (Othello 6x6)
        self.config = NetworkConfig(network_type='cnn', num_filters=16, num_res_blocks=2)
        self.board_size = 36
        self.action_size = 37  # 36 + pass
        self.board_shape = (6, 6)
        self.model = ConvNetWrapper(
            board_size=self.board_size, action_size=self.action_size,
            config=self.config, lr=0.001, board_shape=self.board_shape,
        )

    def test_predict_shape(self):
        state = np.zeros(self.board_size, dtype=np.float32)
        pi, v = self.model.predict(state)
        assert pi.shape == (self.action_size,)
        assert abs(pi.sum() - 1.0) < 1e-5
        assert -1 <= v <= 1

    def test_predict_nonzero_input(self):
        state = np.random.choice([-1, 0, 1], size=self.board_size).astype(np.float32)
        pi, v = self.model.predict(state)
        assert pi.shape == (self.action_size,)
        assert abs(pi.sum() - 1.0) < 1e-5

    def test_train_step(self):
        states = np.random.randn(16, self.board_size).astype(np.float32)
        pis = np.ones((16, self.action_size), dtype=np.float32) / self.action_size
        vs = np.random.randn(16).astype(np.float32).clip(-1, 1)

        losses = self.model.train_step(states, pis, vs)
        assert 'total_loss' in losses
        assert 'policy_loss' in losses
        assert 'value_loss' in losses
        assert losses['total_loss'] > 0

    def test_train_reduces_loss(self):
        """Training on consistent data should reduce loss."""
        state = np.zeros((1, self.board_size), dtype=np.float32)
        pi = np.zeros((1, self.action_size), dtype=np.float32)
        pi[0, 0] = 1.0  # always action 0
        v = np.array([0.5], dtype=np.float32)

        states = np.tile(state, (32, 1))
        pis = np.tile(pi, (32, 1))
        vs = np.tile(v, 32)

        loss1 = self.model.train_step(states, pis, vs)['total_loss']
        for _ in range(50):
            self.model.train_step(states, pis, vs)
        loss2 = self.model.train_step(states, pis, vs)['total_loss']

        assert loss2 < loss1

    def test_save_load(self):
        state = np.zeros(self.board_size, dtype=np.float32)
        pi1, v1 = self.model.predict(state)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            self.model.save(path)
            new_model = ConvNetWrapper(
                board_size=self.board_size, action_size=self.action_size,
                config=self.config, lr=0.001, board_shape=self.board_shape,
            )
            new_model.load(path)
            pi2, v2 = new_model.predict(state)
            np.testing.assert_allclose(pi1, pi2, atol=1e-6)
            assert abs(v1 - v2) < 1e-6
        finally:
            os.unlink(path)

    def test_clone(self):
        state = np.zeros(self.board_size, dtype=np.float32)
        pi1, v1 = self.model.predict(state)

        cloned = self.model.clone()
        pi2, v2 = cloned.predict(state)

        np.testing.assert_allclose(pi1, pi2, atol=1e-6)
        assert abs(v1 - v2) < 1e-6

        # Verify independence
        dummy_states = np.random.randn(16, self.board_size).astype(np.float32)
        dummy_pis = np.ones((16, self.action_size), dtype=np.float32) / self.action_size
        dummy_vs = np.zeros(16, dtype=np.float32)
        cloned.train_step(dummy_states, dummy_pis, dummy_vs)

        pi3, v3 = self.model.predict(state)
        np.testing.assert_allclose(pi1, pi3, atol=1e-6)

    def test_nonsquare_board(self):
        """CNN works with non-square boards (Connect4: 6x7)."""
        config = NetworkConfig(network_type='cnn', num_filters=16, num_res_blocks=2)
        model = ConvNetWrapper(
            board_size=42, action_size=7,
            config=config, lr=0.001, board_shape=(6, 7),
        )
        state = np.zeros(42, dtype=np.float32)
        pi, v = model.predict(state)
        assert pi.shape == (7,)
        assert abs(pi.sum() - 1.0) < 1e-5
        assert -1 <= v <= 1

        # Train step also works
        states = np.random.randn(8, 42).astype(np.float32)
        pis = np.ones((8, 7), dtype=np.float32) / 7
        vs = np.zeros(8, dtype=np.float32)
        losses = model.train_step(states, pis, vs)
        assert losses['total_loss'] > 0


class TestOthelloNet:
    """Tests for OthelloNet — mirrors TestConvNet but with reference architecture."""

    def setup_method(self):
        self.config = NetworkConfig(network_type='othellonet', num_filters=16, dropout=0.3)
        self.board_size = 36
        self.action_size = 37  # 36 + pass
        self.board_shape = (6, 6)
        self.model = OthelloNetWrapper(
            board_size=self.board_size, action_size=self.action_size,
            config=self.config, lr=0.001, board_shape=self.board_shape,
        )

    def test_predict_shape(self):
        state = np.zeros(self.board_size, dtype=np.float32)
        pi, v = self.model.predict(state)
        assert pi.shape == (self.action_size,)
        assert abs(pi.sum() - 1.0) < 1e-5
        assert -1 <= v <= 1

    def test_train_step(self):
        states = np.random.randn(16, self.board_size).astype(np.float32)
        pis = np.ones((16, self.action_size), dtype=np.float32) / self.action_size
        vs = np.random.randn(16).astype(np.float32).clip(-1, 1)

        losses = self.model.train_step(states, pis, vs)
        assert 'total_loss' in losses
        assert 'policy_loss' in losses
        assert 'value_loss' in losses
        assert losses['total_loss'] > 0

    def test_train_reduces_loss(self):
        """Training on consistent data should reduce loss."""
        state = np.zeros((1, self.board_size), dtype=np.float32)
        pi = np.zeros((1, self.action_size), dtype=np.float32)
        pi[0, 0] = 1.0
        v = np.array([0.5], dtype=np.float32)

        states = np.tile(state, (32, 1))
        pis = np.tile(pi, (32, 1))
        vs = np.tile(v, 32)

        loss1 = self.model.train_step(states, pis, vs)['total_loss']
        for _ in range(50):
            self.model.train_step(states, pis, vs)
        loss2 = self.model.train_step(states, pis, vs)['total_loss']

        assert loss2 < loss1

    def test_save_load(self):
        state = np.zeros(self.board_size, dtype=np.float32)
        pi1, v1 = self.model.predict(state)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            self.model.save(path)
            new_model = OthelloNetWrapper(
                board_size=self.board_size, action_size=self.action_size,
                config=self.config, lr=0.001, board_shape=self.board_shape,
            )
            new_model.load(path)
            pi2, v2 = new_model.predict(state)
            np.testing.assert_allclose(pi1, pi2, atol=1e-6)
            assert abs(v1 - v2) < 1e-6
        finally:
            os.unlink(path)

    def test_clone(self):
        state = np.zeros(self.board_size, dtype=np.float32)
        pi1, v1 = self.model.predict(state)

        cloned = self.model.clone()
        pi2, v2 = cloned.predict(state)

        np.testing.assert_allclose(pi1, pi2, atol=1e-6)
        assert abs(v1 - v2) < 1e-6

        # Verify independence
        dummy_states = np.random.randn(16, self.board_size).astype(np.float32)
        dummy_pis = np.ones((16, self.action_size), dtype=np.float32) / self.action_size
        dummy_vs = np.zeros(16, dtype=np.float32)
        cloned.train_step(dummy_states, dummy_pis, dummy_vs)

        pi3, v3 = self.model.predict(state)
        np.testing.assert_allclose(pi1, pi3, atol=1e-6)


class TestCreateModel:
    """Tests for the create_model factory function."""

    class _FakeGame:
        def __init__(self, board_size, action_size, board_shape):
            self._board_size = board_size
            self._action_size = action_size
            self._board_shape = board_shape

        def get_board_size(self):
            return self._board_size

        def get_action_size(self):
            return self._action_size

        def get_board_shape(self):
            return self._board_shape

    def test_creates_mlp(self):
        game = self._FakeGame(9, 9, (3, 3))
        config = NetworkConfig(network_type='mlp', hidden_size=32, num_layers=2)
        model = create_model(game, config)
        assert isinstance(model, SimpleNetWrapper)

    def test_creates_cnn(self):
        game = self._FakeGame(36, 37, (6, 6))
        config = NetworkConfig(network_type='cnn', num_filters=16, num_res_blocks=2)
        model = create_model(game, config)
        assert isinstance(model, ConvNetWrapper)

    def test_creates_othellonet(self):
        game = self._FakeGame(36, 37, (6, 6))
        config = NetworkConfig(network_type='othellonet', num_filters=16, dropout=0.3)
        model = create_model(game, config)
        assert isinstance(model, OthelloNetWrapper)

    def test_unknown_type_raises(self):
        game = self._FakeGame(9, 9, (3, 3))
        config = NetworkConfig(network_type='transformer')
        with pytest.raises(ValueError, match="Unknown network_type"):
            create_model(game, config)
