"""Parallel self-play helpers.

GPU-parallel mode (primary):
  Worker threads do MCTS game logic. A background thread batches their
  NN requests and runs inference on GPU. GPU ops release the GIL, so
  multiple threads overlap their game logic with GPU inference.

CPU-parallel mode (legacy, for arena):
  Each subprocess has its own model copy on CPU.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch


def resolve_num_workers(n: int) -> int:
    """Resolve worker count: 0 -> auto (cpu_count - 1), else as-is."""
    if n == 0:
        return max(1, (os.cpu_count() or 2) - 1)
    return n


# ---------------------------------------------------------------------------
# GPU-parallel: threaded workers + batched GPU inference
# ---------------------------------------------------------------------------

class BatchInferenceModel:
    """Wraps a model to batch inference requests from multiple threads.

    Worker threads call predict/predict_batch which enqueue requests.
    A background thread collects them, forms GPU batches, and dispatches results.
    """

    def __init__(self, model):
        self._model = model
        self._request_queue = queue.Queue()

    def predict(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Single-state inference. Blocks until GPU processes the request."""
        event = threading.Event()
        holder = [None]  # mutable container for result
        self._request_queue.put(([state], event, holder))
        event.wait()
        policies, values = holder[0]
        return policies[0], values[0]

    def predict_batch(self, states: list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
        """Batch inference. Blocks until GPU processes the request."""
        event = threading.Event()
        holder = [None]
        self._request_queue.put((states, event, holder))
        event.wait()
        return holder[0]

    def run_inference_loop(self, stop_event: threading.Event):
        """Background thread: collect requests, batch-process on GPU."""
        while not stop_event.is_set():
            pending = []
            try:
                req = self._request_queue.get(timeout=0.002)
                pending.append(req)
            except queue.Empty:
                continue

            # Drain more requests for cross-thread batching
            while len(pending) < 64:
                try:
                    req = self._request_queue.get_nowait()
                    pending.append(req)
                except queue.Empty:
                    break

            # Flatten all states into one mega-batch
            all_states = []
            meta = []  # (event, holder, start, count)
            for states, event, holder in pending:
                start = len(all_states)
                all_states.extend(states)
                meta.append((event, holder, start, len(states)))

            # GPU batch inference (releases GIL during CUDA ops)
            policies, values = self._model.predict_batch(all_states)

            # Dispatch results and signal waiting threads
            for event, holder, start, count in meta:
                holder[0] = (policies[start:start + count], values[start:start + count])
                event.set()


def _threaded_self_play_worker(game_name: str, batch_model: BatchInferenceModel,
                                mcts_config, num_games: int):
    """Thread worker: play self-play games using batched GPU model."""
    from ..games import get_game
    from .self_play import self_play_game

    game = get_game(game_name)
    results = []

    for _ in range(num_games):
        examples, outcome, diag = self_play_game(
            game, batch_model, mcts_config, collect_diagnostics=True
        )
        results.append((examples, outcome, diag))

    return results


def generate_gpu_parallel_self_play(game, model, mcts_config, num_games: int,
                                     num_workers: int, game_name: str,
                                     augment: bool = True):
    """Generate self-play data: threaded workers + batched GPU inference.

    Workers run MCTS game logic in threads. A background thread batches
    their NN requests and runs inference on GPU. GPU ops release the GIL,
    allowing game logic and inference to overlap.
    """
    from .self_play import SelfPlayStats

    # Wrap model for batched cross-thread inference
    batch_model = BatchInferenceModel(model)

    # Start GPU inference thread
    stop_event = threading.Event()
    gpu_thread = threading.Thread(
        target=batch_model.run_inference_loop,
        args=(stop_event,),
        daemon=True,
    )
    gpu_thread.start()

    # Distribute games across workers
    base = num_games // num_workers
    remainder = num_games % num_workers
    games_per_worker = [base + (1 if i < remainder else 0) for i in range(num_workers)]

    # Run workers in thread pool
    all_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            if games_per_worker[i] > 0:
                f = executor.submit(
                    _threaded_self_play_worker,
                    game_name, batch_model, mcts_config, games_per_worker[i],
                )
                futures.append(f)

        for f in as_completed(futures):
            all_results.extend(f.result())

    # Stop GPU inference thread
    stop_event.set()
    gpu_thread.join(timeout=5)

    # Aggregate results
    all_examples = []
    stats = SelfPlayStats()
    game_lengths = []
    root_values = []
    policy_entropies = []
    search_depths = []

    for examples, outcome, diag in all_results:
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


# ---------------------------------------------------------------------------
# C++-parallel mode (C++ MCTS engine with true multi-threading)
# ---------------------------------------------------------------------------

def generate_cpp_parallel_self_play(game, model, mcts_config, num_games: int,
                                     num_threads: int, augment: bool = True):
    """Generate self-play data using C++ MCTS engine.

    C++ worker threads run game logic + MCTS without the GIL. Only NN inference
    callbacks acquire the GIL momentarily. This pushes GPU utilization to 60-80%.
    """
    from mcts_cpp import generate_self_play_data as cpp_generate
    return cpp_generate(game, model, mcts_config, num_games,
                        num_threads=num_threads, augment=augment)


# ---------------------------------------------------------------------------
# CPU-parallel mode (for arena — workers have own model copy on CPU)
# ---------------------------------------------------------------------------

_worker_model = None
_worker_model1 = None
_worker_model2 = None


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
