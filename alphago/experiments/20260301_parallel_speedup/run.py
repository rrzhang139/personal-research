#!/usr/bin/env python3
"""Experiment: Parallel self-play speedup on Othello (6x6).

Compares wall time for sequential vs parallel self-play/arena/eval
using 1, 2, 4, and 8 workers on Othello with 200 MCTS simulations.

Othello is chosen because games are ~0.8s each (vs 0.03s for tic-tac-toe),
making the per-game cost large enough to amortize pool creation overhead.
"""

import json
import os
import sys
import time

import numpy as np

# Ensure alphago package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games import get_game
from alpha_go.neural_net import create_model
from alpha_go.training.self_play import generate_self_play_data
from alpha_go.training.arena import arena_compare, play_vs_random
from alpha_go.training.trainer import train_on_examples
from alpha_go.training.parallel import resolve_num_workers
from alpha_go.utils.config import MCTSConfig, NetworkConfig

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_DIR, 'data')
FIG_DIR = os.path.join(EXP_DIR, 'figures')

# Experiment parameters
GAME_NAME = 'othello'
NUM_SIMS = 200
NUM_ITERATIONS = 3
GAMES_PER_ITER = 50
ARENA_GAMES = 20
EVAL_GAMES = 30
WORKER_CONFIGS = [1, 2, 4, 8]

MCTS_CONFIG = MCTSConfig(
    num_simulations=NUM_SIMS,
    c_puct=1.0,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
)

NET_CONFIG = NetworkConfig(hidden_size=128, num_layers=4)


def run_iteration(game, model, mcts_config, num_workers, game_name):
    """Run one full iteration and return phase timings."""
    timings = {}

    # Self-play
    t = time.time()
    examples, sp_stats = generate_self_play_data(
        game=game, model=model, mcts_config=mcts_config,
        num_games=GAMES_PER_ITER, augment=True,
        num_workers=num_workers, game_name=game_name,
    )
    timings['self_play'] = time.time() - t

    # Train
    t = time.time()
    new_model = model.clone()
    losses = train_on_examples(
        model=new_model, examples=examples,
        batch_size=64, epochs=5,
    )
    timings['train'] = time.time() - t

    # Arena
    t = time.time()
    win_rate, arena_stats = arena_compare(
        game=game, new_model=new_model, old_model=model,
        mcts_config=mcts_config, num_games=ARENA_GAMES,
        num_workers=num_workers, game_name=game_name,
    )
    timings['arena'] = time.time() - t

    # Eval vs random
    t = time.time()
    vs_random = play_vs_random(
        game, new_model, mcts_config, num_games=EVAL_GAMES,
        num_workers=num_workers, game_name=game_name,
    )
    timings['eval'] = time.time() - t

    timings['total'] = sum(timings.values())

    return timings, losses, sp_stats, vs_random


def main():
    print(f"\n  Parallel Speedup Experiment — {GAME_NAME}")
    print(f"  {NUM_SIMS} sims, {GAMES_PER_ITER} games/iter, {NUM_ITERATIONS} iters")
    print(f"  Workers to test: {WORKER_CONFIGS}")
    print(f"  CPU count: {os.cpu_count()}")
    print()

    game = get_game(GAME_NAME)
    all_results = {}

    for nw in WORKER_CONFIGS:
        resolved = resolve_num_workers(nw)
        label = f"{resolved}w" if nw > 0 else f"auto({resolved}w)"
        print(f"  === {label} {'='*50}")

        # Fresh model for each config (fair comparison)
        np.random.seed(42)
        model = create_model(game, NET_CONFIG, lr=0.001)

        iter_timings = []
        for it in range(1, NUM_ITERATIONS + 1):
            timings, losses, sp_stats, vs_random = run_iteration(
                game, model, MCTS_CONFIG, resolved, GAME_NAME,
            )
            iter_timings.append(timings)

            print(f"    iter {it}/{NUM_ITERATIONS}: "
                  f"self_play={timings['self_play']:.1f}s  "
                  f"train={timings['train']:.1f}s  "
                  f"arena={timings['arena']:.1f}s  "
                  f"eval={timings['eval']:.1f}s  "
                  f"total={timings['total']:.1f}s  "
                  f"vsRand={vs_random:.0%}")

        # Average timings across iterations
        avg = {}
        for key in iter_timings[0]:
            avg[key] = np.mean([t[key] for t in iter_timings])

        total_wall = sum(t['total'] for t in iter_timings)
        print(f"    Avg: self_play={avg['self_play']:.1f}s  "
              f"arena={avg['arena']:.1f}s  eval={avg['eval']:.1f}s  "
              f"total={avg['total']:.1f}s  (wall: {total_wall:.1f}s)")
        print()

        all_results[resolved] = {
            'num_workers': resolved,
            'iterations': iter_timings,
            'avg': {k: float(v) for k, v in avg.items()},
            'total_wall': float(total_wall),
        }

    # Save results
    with open(os.path.join(DATA_DIR, 'metrics.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save config
    config = {
        'game': GAME_NAME,
        'num_simulations': NUM_SIMS,
        'num_iterations': NUM_ITERATIONS,
        'games_per_iteration': GAMES_PER_ITER,
        'arena_games': ARENA_GAMES,
        'eval_games': EVAL_GAMES,
        'worker_configs': WORKER_CONFIGS,
        'network': {'hidden_size': 128, 'num_layers': 4, 'type': 'mlp'},
        'cpu_count': os.cpu_count(),
    }
    with open(os.path.join(EXP_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Generate plots
    generate_plots(all_results)

    # Print summary
    baseline = all_results[1]['avg']['total']
    print(f"\n  Summary (avg iter time):")
    print(f"  {'Workers':>8}  {'Self-Play':>10}  {'Arena':>8}  {'Eval':>8}  {'Total':>8}  {'Speedup':>8}")
    print(f"  {'─'*60}")
    for nw in sorted(all_results.keys()):
        r = all_results[nw]
        speedup = baseline / r['avg']['total']
        print(f"  {nw:>8}  {r['avg']['self_play']:>9.1f}s  "
              f"{r['avg']['arena']:>7.1f}s  {r['avg']['eval']:>7.1f}s  "
              f"{r['avg']['total']:>7.1f}s  {speedup:>7.2f}x")
    print()


def generate_plots(results):
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    workers = sorted(results.keys())

    # --- Plot 1: Stacked bar chart of phase times ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Stacked bars
    ax = axes[0]
    phases = ['self_play', 'train', 'arena', 'eval']
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
    labels = ['Self-Play', 'Train', 'Arena', 'Eval']

    x = range(len(workers))
    bottoms = [0] * len(workers)
    for phase, color, label in zip(phases, colors, labels):
        vals = [results[w]['avg'][phase] for w in workers]
        ax.bar(x, vals, bottom=bottoms, color=color, label=label, alpha=0.8)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in workers])
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Avg Iteration Time by Phase')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Speedup curve
    ax = axes[1]
    baseline_total = results[1]['avg']['total']
    speedups = [baseline_total / results[w]['avg']['total'] for w in workers]
    ax.plot(workers, speedups, 'b-o', linewidth=2, markersize=8, label='Actual')
    ax.plot(workers, workers, 'k--', alpha=0.3, label='Ideal (linear)')
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Total Iteration Speedup')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(workers)

    # Self-play speedup only
    ax = axes[2]
    baseline_sp = results[1]['avg']['self_play']
    sp_speedups = [baseline_sp / results[w]['avg']['self_play'] for w in workers]
    ax.plot(workers, sp_speedups, 'r-o', linewidth=2, markersize=8, label='Self-play')

    baseline_arena = results[1]['avg']['arena']
    arena_speedups = [baseline_arena / results[w]['avg']['arena'] for w in workers]
    ax.plot(workers, arena_speedups, 'g-s', linewidth=2, markersize=8, label='Arena')

    baseline_eval = results[1]['avg']['eval']
    eval_speedups = [baseline_eval / results[w]['avg']['eval'] for w in workers]
    ax.plot(workers, eval_speedups, 'm-^', linewidth=2, markersize=8, label='Eval')

    ax.plot(workers, workers, 'k--', alpha=0.3, label='Ideal')
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Per-Phase Speedup')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(workers)

    fig.suptitle(f'Parallel Self-Play Speedup — Othello 6x6, {NUM_SIMS} sims',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_DIR, 'speedup.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Plot 2: Per-iteration timing breakdown ---
    fig, axes = plt.subplots(1, len(workers), figsize=(4 * len(workers), 4), sharey=True)
    if len(workers) == 1:
        axes = [axes]

    for ax, w in zip(axes, workers):
        iters = range(1, NUM_ITERATIONS + 1)
        for phase, color, label in zip(phases, colors, labels):
            vals = [results[w]['iterations'][i][phase] for i in range(NUM_ITERATIONS)]
            ax.plot(iters, vals, color=color, marker='o', markersize=4, label=label)
        ax.set_xlabel('Iteration')
        ax.set_title(f'{w} Worker{"s" if w > 1 else ""}')
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel('Time (seconds)')
            ax.legend(fontsize=7)

    fig.suptitle('Phase Timing per Iteration', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, 'per_iteration.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Plots saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
