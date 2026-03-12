#!/usr/bin/env python3
"""Benchmark: C++ MCTS vs Python MCTS on GPU.

Runs 5 training iterations with each method, measuring:
- Self-play time per iteration
- Training time per iteration
- Total iteration time
- GPU utilization (sampled)
- Training stability (loss, vs_random)

After benchmarking, runs 20 full iterations with C++ to verify training stability.
"""
import json
import os
import sys
import time
import subprocess
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games.go import Go
from alpha_go.neural_net import create_model
from alpha_go.utils.config import (
    AlphaZeroConfig, MCTSConfig, NetworkConfig, TrainingConfig, ArenaConfig,
)
from alpha_go.training.self_play import generate_self_play_data, SelfPlayStats
from alpha_go.training.trainer import train_on_examples

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


def sample_gpu_util(stop_event, samples):
    """Background thread: sample GPU utilization every 0.5s."""
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                text=True, timeout=2
            )
            samples.append(float(out.strip().split('\n')[0]))
        except Exception:
            pass
        stop_event.wait(0.5)


def run_benchmark_iterations(game, model, mcts_config, training_config, num_iters,
                             num_games, num_workers, use_cpp, label):
    """Run N training iterations and collect timing + metrics."""
    from collections import deque

    replay_buffer = deque(maxlen=training_config.max_buffer_size)
    results = {
        'label': label,
        'self_play_times': [],
        'train_times': [],
        'iter_times': [],
        'gpu_utils': [],
        'losses': [],
        'examples_per_iter': [],
    }

    best_model = model

    for i in range(1, num_iters + 1):
        t_iter = time.time()

        # Self-play with GPU sampling
        gpu_samples = []
        stop_event = threading.Event()
        gpu_thread = threading.Thread(target=sample_gpu_util, args=(stop_event, gpu_samples), daemon=True)
        gpu_thread.start()

        t0 = time.time()
        new_examples, sp_stats = generate_self_play_data(
            game=game,
            model=best_model,
            mcts_config=mcts_config,
            num_games=num_games,
            augment=True,
            num_workers=num_workers,
            game_name="go9",
            use_cpp=use_cpp,
        )
        t_self_play = time.time() - t0

        stop_event.set()
        gpu_thread.join(timeout=2)
        avg_gpu = np.mean(gpu_samples) if gpu_samples else 0.0

        # Training
        replay_buffer.extend(new_examples)
        training_examples = list(replay_buffer)

        t0 = time.time()
        new_model = best_model.clone()
        losses = train_on_examples(
            model=new_model,
            examples=training_examples,
            batch_size=training_config.batch_size,
            epochs=training_config.epochs_per_iteration,
        )
        t_train = time.time() - t0

        best_model = new_model
        t_total = time.time() - t_iter

        results['self_play_times'].append(t_self_play)
        results['train_times'].append(t_train)
        results['iter_times'].append(t_total)
        results['gpu_utils'].append(avg_gpu)
        results['losses'].append(losses)
        results['examples_per_iter'].append(len(new_examples))

        print(f"  [{label}] iter {i}/{num_iters}: "
              f"self_play={t_self_play:.1f}s, train={t_train:.1f}s, total={t_total:.1f}s, "
              f"GPU={avg_gpu:.0f}%, loss={losses['total_loss']:.4f}, "
              f"examples={len(new_examples)}, "
              f"games={sp_stats.p1_wins+sp_stats.p2_wins+sp_stats.draws} "
              f"(B{sp_stats.p1_wins}/W{sp_stats.p2_wins}/D{sp_stats.draws})")

    return results, best_model


def print_comparison(py_results, cpp_results):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    def avg(lst):
        return np.mean(lst) if lst else 0

    py_sp = avg(py_results['self_play_times'])
    cpp_sp = avg(cpp_results['self_play_times'])
    py_tr = avg(py_results['train_times'])
    cpp_tr = avg(cpp_results['train_times'])
    py_tot = avg(py_results['iter_times'])
    cpp_tot = avg(cpp_results['iter_times'])
    py_gpu = avg(py_results['gpu_utils'])
    cpp_gpu = avg(cpp_results['gpu_utils'])
    py_ex = avg(py_results['examples_per_iter'])
    cpp_ex = avg(cpp_results['examples_per_iter'])

    print(f"\n{'Metric':<25} {'Python':>12} {'C++':>12} {'Speedup':>10}")
    print("-" * 60)
    print(f"{'Self-play (s/iter)':<25} {py_sp:>11.1f}s {cpp_sp:>11.1f}s {py_sp/max(cpp_sp,0.01):>9.1f}x")
    print(f"{'Training (s/iter)':<25} {py_tr:>11.1f}s {cpp_tr:>11.1f}s {py_tr/max(cpp_tr,0.01):>9.1f}x")
    print(f"{'Total (s/iter)':<25} {py_tot:>11.1f}s {cpp_tot:>11.1f}s {py_tot/max(cpp_tot,0.01):>9.1f}x")
    print(f"{'GPU utilization (%)':<25} {py_gpu:>11.0f}% {cpp_gpu:>11.0f}%")
    print(f"{'Examples/iter':<25} {py_ex:>12.0f} {cpp_ex:>12.0f}")

    # Loss comparison
    py_loss = [l['total_loss'] for l in py_results['losses']]
    cpp_loss = [l['total_loss'] for l in cpp_results['losses']]
    print(f"\n{'Loss (final)':<25} {py_loss[-1]:>12.4f} {cpp_loss[-1]:>12.4f}")
    print(f"{'Loss (mean)':<25} {np.mean(py_loss):>12.4f} {np.mean(cpp_loss):>12.4f}")

    print("\nSelf-play per iteration:")
    for i, (ps, cs) in enumerate(zip(py_results['self_play_times'], cpp_results['self_play_times'])):
        print(f"  iter {i+1}: Python={ps:.1f}s  C++={cs:.1f}s  speedup={ps/max(cs,0.01):.1f}x")

    return {
        'python_self_play_mean': py_sp,
        'cpp_self_play_mean': cpp_sp,
        'self_play_speedup': py_sp / max(cpp_sp, 0.01),
        'python_total_mean': py_tot,
        'cpp_total_mean': cpp_tot,
        'total_speedup': py_tot / max(cpp_tot, 0.01),
        'python_gpu_util': py_gpu,
        'cpp_gpu_util': cpp_gpu,
    }


def main():
    game = Go(size=9)

    mcts_config = MCTSConfig(
        num_simulations=200,
        c_puct=1.0,
        dirichlet_alpha=0.12,
        dirichlet_epsilon=0.25,
        temp_threshold=30,
        temp_decay_halflife=19,
        nn_batch_size=64,
        playout_cap_prob=0.125,
        playout_cap_cheap_fraction=0.15,
        fpu_reduction=0.2,
        root_fpu_reduction=0.1,
    )

    training_config = TrainingConfig(
        lr=0.002,
        weight_decay=1e-4,
        batch_size=256,
        epochs_per_iteration=10,
        max_buffer_size=200000,
        buffer_strategy="fifo",
    )

    # Smaller benchmark: fewer games for quick comparison
    NUM_GAMES = 50    # games per iteration (50 instead of 500 for speed)
    BENCH_ITERS = 5   # iterations to benchmark
    STABILITY_ITERS = 20  # extra iterations to verify C++ stability
    NUM_WORKERS_CPP = 1   # C++ worker threads (1 = best; multi-thread slower due to GIL)

    config = AlphaZeroConfig(
        game="go9",
        network=NetworkConfig(
            network_type="cnn",
            num_filters=128,
            num_res_blocks=4,
        ),
        training=training_config,
    )

    # Create fresh model for fair comparison
    model = create_model(game, config.network, lr=training_config.lr,
                         weight_decay=training_config.weight_decay)
    device = str(model.net.device)
    print(f"Device: {device}")
    print(f"Benchmark: {BENCH_ITERS} iters × {NUM_GAMES} games, 200 sims")
    print(f"C++ threads: {NUM_WORKERS_CPP}")
    print()

    # --- Phase 1: Python baseline (sequential) ---
    print("=" * 50)
    print("Phase 1: Python sequential self-play")
    print("=" * 50)
    py_model = model.clone()
    py_results, py_final_model = run_benchmark_iterations(
        game, py_model, mcts_config, training_config,
        num_iters=BENCH_ITERS, num_games=NUM_GAMES,
        num_workers=1, use_cpp=False, label="Python",
    )

    # --- Phase 2: C++ self-play ---
    print()
    print("=" * 50)
    print(f"Phase 2: C++ MCTS ({NUM_WORKERS_CPP} threads)")
    print("=" * 50)
    cpp_model = model.clone()
    cpp_results, cpp_final_model = run_benchmark_iterations(
        game, cpp_model, mcts_config, training_config,
        num_iters=BENCH_ITERS, num_games=NUM_GAMES,
        num_workers=NUM_WORKERS_CPP, use_cpp=True, label="C++",
    )

    # --- Comparison ---
    comparison = print_comparison(py_results, cpp_results)

    # --- Phase 3: C++ stability test (20 more iterations) ---
    print()
    print("=" * 50)
    print(f"Phase 3: C++ stability test ({STABILITY_ITERS} iters × {NUM_GAMES} games)")
    print("=" * 50)
    stability_results, stable_model = run_benchmark_iterations(
        game, cpp_final_model, mcts_config, training_config,
        num_iters=STABILITY_ITERS, num_games=NUM_GAMES,
        num_workers=NUM_WORKERS_CPP, use_cpp=True, label="C++-stable",
    )

    # Check stability: loss should be decreasing or stable
    all_losses = [l['total_loss'] for l in cpp_results['losses']] + \
                 [l['total_loss'] for l in stability_results['losses']]
    first_half = np.mean(all_losses[:len(all_losses)//2])
    second_half = np.mean(all_losses[len(all_losses)//2:])
    stable = second_half <= first_half * 1.1  # allow 10% increase

    print(f"\nStability check: loss first half={first_half:.4f}, second half={second_half:.4f}")
    print(f"Training {'STABLE' if stable else 'UNSTABLE'}")

    # Save results
    results = {
        'comparison': comparison,
        'python_self_play_times': py_results['self_play_times'],
        'cpp_self_play_times': cpp_results['self_play_times'],
        'python_losses': [l['total_loss'] for l in py_results['losses']],
        'cpp_losses': [l['total_loss'] for l in cpp_results['losses']],
        'cpp_stability_losses': [l['total_loss'] for l in stability_results['losses']],
        'stable': stable,
    }
    out_path = os.path.join(EXPERIMENT_DIR, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
