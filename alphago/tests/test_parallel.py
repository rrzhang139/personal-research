"""Tests for the parallel multiprocessing module."""

import numpy as np

from alpha_go.neural_net.simple_net import SimpleNetWrapper
from alpha_go.training.parallel import (
    resolve_num_workers,
    serialize_model_state,
    _reconstruct_model,
    create_pool,
)
from alpha_go.utils.config import NetworkConfig


def _make_model():
    config = NetworkConfig(hidden_size=32, num_layers=2)
    return SimpleNetWrapper(board_size=9, action_size=9, config=config, lr=0.001)


class TestResolveNumWorkers:

    def test_zero_returns_auto(self):
        result = resolve_num_workers(0)
        assert result >= 1

    def test_positive_returns_as_is(self):
        assert resolve_num_workers(4) == 4
        assert resolve_num_workers(1) == 1

    def test_one_returns_one(self):
        assert resolve_num_workers(1) == 1


class TestSerializeRoundtrip:

    def test_mlp_roundtrip(self):
        model = _make_model()
        state = np.random.randn(9).astype(np.float32)
        pi_before, v_before = model.predict(state)

        weight_bytes, info = serialize_model_state(model)
        assert isinstance(weight_bytes, bytes)
        assert len(weight_bytes) > 0
        assert info['board_size'] == 9
        assert info['action_size'] == 9

        restored = _reconstruct_model(weight_bytes, info)
        pi_after, v_after = restored.predict(state)

        np.testing.assert_array_almost_equal(pi_before, pi_after, decimal=5)
        assert abs(v_before - v_after) < 1e-5

    def test_info_has_required_fields(self):
        model = _make_model()
        _, info = serialize_model_state(model)
        assert 'board_size' in info
        assert 'action_size' in info
        assert 'config' in info
        assert 'lr' in info


class TestCreatePool:

    def test_create_and_close(self):
        model = _make_model()
        pool = create_pool(model, num_workers=2)
        # Pool should be usable
        assert pool is not None
        pool.close()
        pool.join()
