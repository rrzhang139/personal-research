"""Benchmark C++ Go engine vs Python Go engine."""

import time
import numpy as np

def bench_get_next_state(n_calls=10000):
    """Benchmark get_next_state: 10K calls."""
    from alpha_go.games.go import Go as PyGo
    from mcts_cpp._mcts_cpp import GoGame as CppGo

    py_game = PyGo(9)
    cpp_game = CppGo(9)

    # Setup: initial state
    py_state = py_game.get_initial_state()
    cpp_state = np.asarray(cpp_game.get_initial_state())

    rng = np.random.RandomState(42)

    # Python benchmark
    t0 = time.perf_counter()
    state = py_state.copy()
    player = 1
    for i in range(n_calls):
        valid = py_game.get_valid_moves(state, player)
        actions = np.where(valid > 0)[0]
        action = rng.choice(actions)
        state = py_game.get_next_state(state, action, player)
        is_term, _ = py_game.check_terminal(state, action, player)
        if is_term:
            state = py_game.get_initial_state()
            player = 1
        else:
            player = -player
    py_time = time.perf_counter() - t0

    # C++ benchmark
    rng2 = np.random.RandomState(42)
    t0 = time.perf_counter()
    state = cpp_state.copy()
    player = 1
    for i in range(n_calls):
        valid = np.asarray(cpp_game.get_valid_moves(state, player))
        actions = np.where(valid > 0)[0]
        action = rng2.choice(actions)
        state = np.asarray(cpp_game.get_next_state(state, int(action), player))
        is_term, _ = cpp_game.check_terminal(state, int(action), player)
        if is_term:
            state = np.asarray(cpp_game.get_initial_state())
            player = 1
        else:
            player = -player
    cpp_time = time.perf_counter() - t0

    print(f"get_next_state + get_valid_moves + check_terminal ({n_calls} calls):")
    print(f"  Python: {py_time:.3f}s ({n_calls/py_time:.0f} calls/s)")
    print(f"  C++:    {cpp_time:.3f}s ({n_calls/cpp_time:.0f} calls/s)")
    print(f"  Speedup: {py_time/cpp_time:.1f}x")

def bench_random_games(n_games=100):
    """Benchmark playing random games."""
    from alpha_go.games.go import Go as PyGo
    from mcts_cpp._mcts_cpp import GoGame as CppGo

    py_game = PyGo(9)
    cpp_game = CppGo(9)

    # Python
    rng = np.random.RandomState(42)
    t0 = time.perf_counter()
    for _ in range(n_games):
        state = py_game.get_initial_state()
        player = 1
        for move in range(300):
            valid = py_game.get_valid_moves(state, player)
            actions = np.where(valid > 0)[0]
            action = rng.choice(actions)
            state = py_game.get_next_state(state, action, player)
            is_term, _ = py_game.check_terminal(state, action, player)
            if is_term:
                break
            player = -player
    py_time = time.perf_counter() - t0

    # C++
    rng2 = np.random.RandomState(42)
    t0 = time.perf_counter()
    for _ in range(n_games):
        state = np.asarray(cpp_game.get_initial_state())
        player = 1
        for move in range(300):
            valid = np.asarray(cpp_game.get_valid_moves(state, player))
            actions = np.where(valid > 0)[0]
            action = rng2.choice(actions)
            state = np.asarray(cpp_game.get_next_state(state, int(action), player))
            is_term, _ = cpp_game.check_terminal(state, int(action), player)
            if is_term:
                break
            player = -player
    cpp_time = time.perf_counter() - t0

    print(f"\nRandom games ({n_games} games):")
    print(f"  Python: {py_time:.3f}s ({n_games/py_time:.1f} games/s)")
    print(f"  C++:    {cpp_time:.3f}s ({n_games/cpp_time:.1f} games/s)")
    print(f"  Speedup: {py_time/cpp_time:.1f}x")


if __name__ == "__main__":
    bench_get_next_state()
    bench_random_games()
