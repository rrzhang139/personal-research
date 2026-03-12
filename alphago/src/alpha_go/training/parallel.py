"""Multiprocessing helpers for parallel self-play and arena games.

Uses Pool + worker initializer pattern: serialize model weights once,
each worker reconstructs the model on init, then plays games using
a process-global model reference. This avoids re-serializing per game.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import sys

import torch


def resolve_num_workers(n: int) -> int:
    """Resolve worker count: 0 -> auto (cpu_count - 1), else as-is."""
    if n == 0:
        return max(1, (os.cpu_count() or 2) - 1)
    return n


def serialize_model_state(model) -> tuple[bytes, dict]:
    """Serialize model weights to bytes + metadata for reconstruction.

    Returns:
        (weight_bytes, info): info contains board_size, action_size, config, lr,
        and optionally board_shape for CNN models.
    """
    buf = io.BytesIO()
    torch.save(model.net.state_dict(), buf)
    weight_bytes = buf.getvalue()

    info = {
        'board_size': model.board_size,
        'action_size': model.action_size,
        'config': model.config,
        'lr': model.lr,
    }
    if hasattr(model, 'board_shape'):
        info['board_shape'] = model.board_shape

    return weight_bytes, info


def _reconstruct_model(weight_bytes: bytes, info: dict):
    """Rebuild a model wrapper from serialized weights + info.

    Always reconstructs on CPU — MCTS self-play is CPU-bound (single-sample
    forward passes), and GPU transfer overhead makes it slower. Training
    uses the main process model on GPU separately.
    """
    from ..neural_net.conv_net import ConvNetWrapper
    from ..neural_net.simple_net import SimpleNetWrapper

    # Force CPU device for worker models to avoid CUDA issues in subprocesses
    _orig_cuda_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        config = info['config']
        if config.network_type == 'cnn':
            model = ConvNetWrapper(
                info['board_size'], info['action_size'], config,
                lr=info['lr'], board_shape=info['board_shape'],
            )
        else:
            model = SimpleNetWrapper(
                info['board_size'], info['action_size'], config, lr=info['lr'],
            )
    finally:
        torch.cuda.is_available = _orig_cuda_available

    state_dict = torch.load(io.BytesIO(weight_bytes), map_location='cpu', weights_only=True)
    model.net.load_state_dict(state_dict)
    model.net.eval()
    return model


# Process-global model references (set by pool initializers)
_worker_model = None
_worker_model1 = None
_worker_model2 = None


def _worker_init(weight_bytes: bytes, info: dict, threads_per_worker: int = 1):
    """Pool initializer: load model into process-global _worker_model."""
    global _worker_model
    # Limit CPU threads per worker to avoid contention (default PyTorch uses ALL cores)
    t = str(threads_per_worker)
    os.environ['OMP_NUM_THREADS'] = t
    os.environ['MKL_NUM_THREADS'] = t
    import torch
    torch.set_num_threads(threads_per_worker)
    _worker_model = _reconstruct_model(weight_bytes, info)
    # Seed with pid for game diversity
    import numpy as np
    np.random.seed(os.getpid() % (2**31))


def _worker_init_two_models(
    weight_bytes1: bytes, info1: dict,
    weight_bytes2: bytes, info2: dict,
):
    """Pool initializer for arena: load two models."""
    global _worker_model1, _worker_model2
    _worker_model1 = _reconstruct_model(weight_bytes1, info1)
    _worker_model2 = _reconstruct_model(weight_bytes2, info2)
    import numpy as np
    np.random.seed(os.getpid() % (2**31))


def _get_mp_context():
    """Get multiprocessing context.

    'fork' on Linux without CUDA (fast — no re-import overhead).
    'spawn' on macOS or when CUDA is initialized (CUDA can't survive fork).
    """
    if sys.platform == 'linux' and not torch.cuda.is_initialized():
        return mp.get_context('fork')
    return mp.get_context('spawn')


def create_pool(model, num_workers: int) -> mp.pool.Pool:
    """Create a process pool with model pre-loaded in each worker."""
    weight_bytes, info = serialize_model_state(model)
    ctx = _get_mp_context()
    # Allocate CPU threads evenly: total cores / workers (min 1)
    threads_per_worker = max(1, (os.cpu_count() or 1) // num_workers)
    return ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(weight_bytes, info, threads_per_worker),
    )


def create_arena_pool(model1, model2, num_workers: int) -> mp.pool.Pool:
    """Create a process pool with two models pre-loaded (for arena)."""
    wb1, info1 = serialize_model_state(model1)
    wb2, info2 = serialize_model_state(model2)
    ctx = _get_mp_context()
    return ctx.Pool(
        processes=num_workers,
        initializer=_worker_init_two_models,
        initargs=(wb1, info1, wb2, info2),
    )
