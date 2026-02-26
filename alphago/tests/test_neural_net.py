"""Tests for the neural network."""

import numpy as np
import os
import tempfile

from alpha_go.neural_net.simple_net import SimpleNetWrapper
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
