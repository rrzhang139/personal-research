"""Multiprocessing helpers for parallel self-play and arena games.

Two modes:
1. CPU-parallel (original): each worker has its own model copy on CPU.
   Good for small models where CPU inference is fast.

2. GPU-parallel (new): workers do game logic on CPU, send inference
   requests to a centralized GPU server in the main process via queues.
   Much faster for CNN models where GPU batch inference >> CPU inference.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import queue as stdlib_queue
import sys
import threading

import numpy as np
import torch


def resolve_num_workers(n: int) -> int:
    """Resolve worker count: 0 -> auto (cpu_count - 1), else as-is."""
    if n == 0:
        return max(1, (os.cpu_count() or 2) - 1)
    return n


# ---------------------------------------------------------------------------
# Model serialization (used by CPU-parallel mode)
# ---------------------------------------------------------------------------

def serialize_model_state(model) -> tuple[bytes, dict]:
    """Serialize model weights to bytes + metadata for reconstruction."""
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
    """Rebuild a model wrapper from serialized weights + info (CPU only)."""
    from ..neural_net.conv_net import ConvNetWrapper
    from ..neural_net.simple_net import SimpleNetWrapper

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


# ---------------------------------------------------------------------------
# CPU-parallel mode (original Pool-based approach)
# ---------------------------------------------------------------------------

_worker_model = None
_worker_model1 = None
_worker_model2 = None


def _worker_init(weight_bytes: bytes, info: dict):
    """Pool initializer: load model into process-global _worker_model."""
    global _worker_model
    _worker_model = _reconstruct_model(weight_bytes, info)
    np.random.seed(os.getpid() % (2**31))


def _worker_init_two_models(
    weight_bytes1: bytes, info1: dict,
    weight_bytes2: bytes, info2: dict,
):
    """Pool initializer for arena: load two models."""
    global _worker_model1, _worker_model2
    _worker_model1 = _reconstruct_model(weight_bytes1, info1)
    _worker_model2 = _reconstruct_model(weight_bytes2, info2)
    np.random.seed(os.getpid() % (2**31))


def _get_mp_context():
    """Get multiprocessing context."""
    if sys.platform == 'linux' and not torch.cuda.is_initialized():
        return mp.get_context('fork')
    return mp.get_context('spawn')


def create_pool(model, num_workers: int) -> mp.pool.Pool:
    """Create a process pool with model pre-loaded in each worker."""
    weight_bytes, info = serialize_model_state(model)
    ctx = _get_mp_context()
    return ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(weight_bytes, info),
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


# ---------------------------------------------------------------------------
# GPU-parallel mode: workers do game logic, GPU server does inference
# ---------------------------------------------------------------------------

class ModelProxy:
    """Proxy model that sends inference requests to GPU server via queues.

    Implements predict/predict_batch so MCTS search can use it transparently.
    """

    def __init__(self, request_queue, response_queue, worker_id: int):
        self.req_q = request_queue
        self.resp_q = response_queue
        self.worker_id = worker_id

    def predict(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        self.req_q.put((self.worker_id, [state]))
        policies, values = self.resp_q.get()
        return policies[0], values[0]

    def predict_batch(self, states: list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
        self.req_q.put((self.worker_id, states))
        return self.resp_q.get()


def _gpu_worker_main(worker_id: int, game_name: str, mcts_config,
                     num_games: int, request_queue, response_queue,
                     result_queue):
    """Worker process: play self-play games, send inference to GPU server."""
    # No torch needed in workers — only game logic + numpy
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    np.random.seed(os.getpid() % (2**31))

    # Import game and self-play here to avoid importing torch
    from ..games import get_game
    from .self_play import self_play_game

    game = get_game(game_name)
    proxy = ModelProxy(request_queue, response_queue, worker_id)

    for _ in range(num_games):
        examples, outcome, diag = self_play_game(
            game, proxy, mcts_config, collect_diagnostics=True
        )
        result_queue.put((examples, outcome, diag))

    # Signal done to GPU server
    request_queue.put(None)


def _gpu_inference_loop(model, request_queue, response_queues, num_workers):
    """Run GPU inference server: batch requests from workers, evaluate on GPU.

    Runs until all workers signal done (by putting None in request_queue).
    Batches across multiple workers for better GPU utilization.
    """
    workers_done = 0

    while workers_done < num_workers:
        # Collect first request (blocking with timeout)
        pending = []
        try:
            req = request_queue.get(timeout=0.005)
            if req is None:
                workers_done += 1
                continue
            pending.append(req)
        except stdlib_queue.Empty:
            continue

        # Drain more requests (non-blocking) for batching across workers
        drain_limit = max(num_workers * 2, 64)
        while len(pending) < drain_limit:
            try:
                req = request_queue.get_nowait()
                if req is None:
                    workers_done += 1
                    continue
                pending.append(req)
            except stdlib_queue.Empty:
                break

        if not pending:
            continue

        # Flatten all states into one mega-batch
        all_states = []
        meta = []  # (worker_id, start_idx, count)
        for worker_id, states in pending:
            start = len(all_states)
            all_states.extend(states)
            meta.append((worker_id, start, len(states)))

        # GPU batch inference
        policies, values = model.predict_batch(all_states)

        # Dispatch results back to workers
        for worker_id, start, count in meta:
            resp = (policies[start:start + count], values[start:start + count])
            response_queues[worker_id].put(resp)


def generate_gpu_parallel_self_play(game, model, mcts_config, num_games: int,
                                     num_workers: int, game_name: str,
                                     augment: bool = True):
    """Generate self-play data: workers on CPU, inference on GPU.

    Workers run MCTS game logic on CPU and send inference requests to a
    GPU server running in the main process. This keeps the GPU busy with
    large batches while parallelizing game logic across CPU cores.

    Returns:
        (examples, stats): same format as generate_self_play_data.
    """
    from .self_play import SelfPlayStats

    # Use fork on Linux for fast worker startup (workers don't use CUDA)
    ctx = mp.get_context('fork') if sys.platform == 'linux' else mp.get_context('spawn')

    # Create queues
    request_queue = ctx.Queue()
    response_queues = [ctx.Queue() for _ in range(num_workers)]
    result_queue = ctx.Queue()

    # Distribute games evenly
    base = num_games // num_workers
    remainder = num_games % num_workers
    games_per_worker = [base + (1 if i < remainder else 0) for i in range(num_workers)]

    # Start worker processes
    workers = []
    for i in range(num_workers):
        if games_per_worker[i] == 0:
            # Signal done immediately for empty workers
            request_queue.put(None)
            continue
        p = ctx.Process(
            target=_gpu_worker_main,
            args=(i, game_name, mcts_config, games_per_worker[i],
                  request_queue, response_queues[i], result_queue),
        )
        p.start()
        workers.append(p)

    # Run GPU inference loop in main thread (blocks until all workers done)
    _gpu_inference_loop(model, request_queue, response_queues, num_workers)

    # Wait for workers to finish
    for p in workers:
        p.join()

    # Collect results
    all_examples = []
    stats = SelfPlayStats()
    game_lengths = []
    root_values = []
    policy_entropies = []
    search_depths = []

    while not result_queue.empty():
        examples, outcome, diag = result_queue.get_nowait()

        if outcome == 1:
            stats.p1_wins += 1
        elif outcome == -1:
            stats.p2_wins += 1
        else:
            stats.draws += 1

        if diag:
            game_lengths.append(diag['game_length'])
            root_values.append(diag['mean_root_value'])
            policy_entropies.append(diag['mean_policy_entropy'])
            search_depths.append(diag['mean_search_depth'])

        if augment:
            for state, pi, v in examples:
                for sym_state, sym_pi in game.get_symmetries(state, pi):
                    all_examples.append((sym_state, sym_pi, v))
        else:
            all_examples.extend(examples)

    if game_lengths:
        stats.mean_game_length = float(np.mean(game_lengths))
        stats.mean_root_value = float(np.mean(root_values))
        stats.mean_policy_entropy = float(np.mean(policy_entropies))
        stats.mean_search_depth = float(np.mean(search_depths))

    return all_examples, stats
