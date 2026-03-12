"""Python wrapper for the C++ MCTS engine.

Adapts the C++ API to match the existing Python training pipeline interface.
"""

from __future__ import annotations

import numpy as np

from alpha_go.training.self_play import SelfPlayStats
from alpha_go.utils.config import MCTSConfig as PyMCTSConfig


def _import_cpp():
    """Import the compiled C++ module."""
    try:
        from mcts_cpp._mcts_cpp import (
            GoGame,
            MCTSConfig as CppMCTSConfig,
            GameStats,
            Example,
            generate_self_play_data as _cpp_generate,
        )
        return GoGame, CppMCTSConfig, GameStats, Example, _cpp_generate
    except ImportError as e:
        raise ImportError(
            f"C++ MCTS module not found. Build it first:\n"
            f"  cmake -S alphago -B alphago/build -DCMAKE_BUILD_TYPE=Release\n"
            f"  cmake --build alphago/build --parallel\n"
            f"  cp alphago/build/_mcts_cpp*.so alphago/src/mcts_cpp/\n"
            f"Original error: {e}"
        ) from e


def _convert_config(py_config: PyMCTSConfig):
    """Convert Python MCTSConfig to C++ MCTSConfig."""
    _, CppMCTSConfig, _, _, _ = _import_cpp()
    cfg = CppMCTSConfig()
    cfg.num_simulations = py_config.num_simulations
    cfg.c_puct = py_config.c_puct
    cfg.dirichlet_alpha = py_config.dirichlet_alpha
    cfg.dirichlet_epsilon = py_config.dirichlet_epsilon
    cfg.temperature = py_config.temperature
    cfg.temp_threshold = py_config.temp_threshold
    cfg.temp_decay_halflife = getattr(py_config, 'temp_decay_halflife', 0)
    cfg.nn_batch_size = getattr(py_config, 'nn_batch_size', 1)
    cfg.playout_cap_prob = getattr(py_config, 'playout_cap_prob', 1.0)
    cfg.playout_cap_cheap_fraction = getattr(py_config, 'playout_cap_cheap_fraction', 0.25)
    cfg.fpu_reduction = getattr(py_config, 'fpu_reduction', 0.0)
    cfg.root_fpu_reduction = getattr(py_config, 'root_fpu_reduction', -1.0)
    cfg.c_puct_base = getattr(py_config, 'c_puct_base', 0.0)
    return cfg


def generate_self_play_data(game, model, mcts_config: PyMCTSConfig,
                            num_games: int, num_threads: int = 4,
                            augment: bool = True):
    """Generate self-play data using C++ MCTS engine.

    Args:
        game: Go game instance (for board_size and symmetries).
        model: PyTorch model wrapper with predict_batch method.
        mcts_config: Python MCTSConfig.
        num_games: Number of games to generate.
        num_threads: Number of C++ worker threads.
        augment: Whether to apply symmetry augmentation.

    Returns:
        (examples, stats): Same format as Python generate_self_play_data.
    """
    _, _, _, _, _cpp_generate = _import_cpp()

    board_size = game.size
    cpp_config = _convert_config(mcts_config)

    # Create predict_fn that C++ can call
    # Import torch here to avoid import at module level
    import torch

    device = model.net.device

    def predict_fn(states_arr):
        """Predict function called from C++ with GIL acquired.

        Args:
            states_arr: numpy array of shape (batch_size, nn_input_size)

        Returns:
            (policies, values): numpy float32 arrays
        """
        states_np = np.asarray(states_arr, dtype=np.float32)
        if states_np.ndim == 1:
            states_np = states_np.reshape(1, -1)

        # Direct torch conversion — skip list roundtrip through predict_batch
        if model.net.training:
            model.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(states_np).to(device)
            log_pi, v = model.net(x)
            policies_arr = torch.exp(log_pi).cpu().numpy()
            values_arr = v.squeeze(-1).cpu().numpy()

        return policies_arr, values_arr

    # Call C++ engine
    examples_cpp, cpp_stats = _cpp_generate(
        board_size, num_games, cpp_config, predict_fn, num_threads
    )

    # Convert C++ Examples to Python tuples and apply augmentation
    all_examples = []
    for ex in examples_cpp:
        state = np.array(ex.get_state(), dtype=np.float32)
        policy = np.array(ex.get_policy(), dtype=np.float32)
        value = ex.value

        if augment:
            for sym_state, sym_pi in game.get_symmetries(state, policy):
                all_examples.append((sym_state, sym_pi, value))
        else:
            all_examples.append((state, policy, value))

    # Convert stats
    stats = SelfPlayStats()
    stats.p1_wins = cpp_stats.p1_wins
    stats.p2_wins = cpp_stats.p2_wins
    stats.draws = cpp_stats.draws
    stats.mean_game_length = cpp_stats.mean_game_length
    stats.mean_root_value = cpp_stats.mean_root_value
    stats.mean_policy_entropy = cpp_stats.mean_policy_entropy
    stats.mean_search_depth = cpp_stats.mean_search_depth

    return all_examples, stats
