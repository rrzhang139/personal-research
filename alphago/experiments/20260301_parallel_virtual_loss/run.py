#!/usr/bin/env python3
"""Experiment: Game parallelism vs Virtual Loss vs Both.

Compares three parallelism strategies on Othello 6x6 (200 sims):
1. Baseline: sequential games, sequential MCTS (1 worker, batch=1)
2. Game parallel: 4 workers, sequential MCTS (batch=1)
3. Game parallel + VL: 4 workers, batched MCTS (batch=8)

Measures wall time per phase and total, plus play quality (vs random).
"""

import json
import os
import sys
import time

import numpy as np

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
NET_CONFIG = NetworkConfig(hidden_size=128, num_layers=4)

# Configs to test: (label, num_workers, nn_batch_size)
CONFIGS = [
    ('baseline',       1, 1),
    ('4w_batch1',      4, 1),
    ('4w_batch8',      4, 8),
]


def make_mcts_config(nn_batch_size):
    return MCTSConfig(
        num_simulations=NUM_SIMS,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        nn_batch_size=nn_batch_size,
    )


def run_iteration(game, model, mcts_config, num_workers, game_name):
    """Run one full iteration and return phase timings + metrics."""
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
    print(f"\n  Parallel + Virtual Loss Experiment — {GAME_NAME}")
    print(f"  {NUM_SIMS} sims, {GAMES_PER_ITER} games/iter, {NUM_ITERATIONS} iters")
    print(f"  CPU count: {os.cpu_count()}")
    print()

    game = get_game(GAME_NAME)
    all_results = {}

    for label, nw, nn_batch in CONFIGS:
        mcts_config = make_mcts_config(nn_batch)
        resolved_nw = resolve_num_workers(nw)
        print(f"  === {label} (workers={resolved_nw}, nn_batch={nn_batch}) {'='*30}")

        np.random.seed(42)
        model = create_model(game, NET_CONFIG, lr=0.001)

        iter_timings = []
        vs_randoms = []
        for it in range(1, NUM_ITERATIONS + 1):
            timings, losses, sp_stats, vs_random = run_iteration(
                game, model, mcts_config, resolved_nw, GAME_NAME,
            )
            iter_timings.append(timings)
            vs_randoms.append(vs_random)

            print(f"    iter {it}/{NUM_ITERATIONS}: "
                  f"self_play={timings['self_play']:.1f}s  "
                  f"train={timings['train']:.1f}s  "
                  f"arena={timings['arena']:.1f}s  "
                  f"eval={timings['eval']:.1f}s  "
                  f"total={timings['total']:.1f}s  "
                  f"vsRand={vs_random:.0%}")

        avg = {}
        for key in iter_timings[0]:
            avg[key] = float(np.mean([t[key] for t in iter_timings]))

        total_wall = sum(t['total'] for t in iter_timings)
        print(f"    Avg: self_play={avg['self_play']:.1f}s  "
              f"arena={avg['arena']:.1f}s  eval={avg['eval']:.1f}s  "
              f"total={avg['total']:.1f}s  (wall: {total_wall:.1f}s)")
        print()

        all_results[label] = {
            'num_workers': resolved_nw,
            'nn_batch_size': nn_batch,
            'iterations': iter_timings,
            'avg': avg,
            'total_wall': float(total_wall),
            'vs_random': [float(v) for v in vs_randoms],
        }

    # Save results
    with open(os.path.join(DATA_DIR, 'metrics.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    config = {
        'game': GAME_NAME,
        'num_simulations': NUM_SIMS,
        'num_iterations': NUM_ITERATIONS,
        'games_per_iteration': GAMES_PER_ITER,
        'arena_games': ARENA_GAMES,
        'eval_games': EVAL_GAMES,
        'configs': {label: {'workers': nw, 'nn_batch': nb} for label, nw, nb in CONFIGS},
        'network': {'hidden_size': 128, 'num_layers': 4, 'type': 'mlp'},
        'cpu_count': os.cpu_count(),
    }
    with open(os.path.join(EXP_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    generate_plots(all_results)
    print_summary(all_results)


def print_summary(results):
    baseline = results['baseline']['avg']['total']
    print(f"\n  Summary (avg iter time):")
    print(f"  {'Config':>18}  {'Workers':>7}  {'Batch':>5}  {'Self-Play':>10}  {'Arena':>8}  {'Eval':>8}  {'Total':>8}  {'Speedup':>8}")
    print(f"  {'─'*80}")
    for label in ['baseline', '4w_batch1', '4w_batch8']:
        r = results[label]
        speedup = baseline / r['avg']['total']
        print(f"  {label:>18}  {r['num_workers']:>7}  {r['nn_batch_size']:>5}  "
              f"{r['avg']['self_play']:>9.1f}s  {r['avg']['arena']:>7.1f}s  "
              f"{r['avg']['eval']:>7.1f}s  {r['avg']['total']:>7.1f}s  {speedup:>7.2f}x")
    print()


def generate_plots(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = ['baseline', '4w_batch1', '4w_batch8']
    display_labels = ['Sequential\n(1w, batch=1)', 'Game Parallel\n(4w, batch=1)', 'Game + VL\n(4w, batch=8)']

    # --- Plot 1: Stacked bars + speedup ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Stacked bars
    ax = axes[0]
    phases = ['self_play', 'train', 'arena', 'eval']
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
    phase_labels = ['Self-Play', 'Train', 'Arena', 'Eval']

    x = range(len(labels))
    bottoms = [0] * len(labels)
    for phase, color, plabel in zip(phases, colors, phase_labels):
        vals = [results[l]['avg'][phase] for l in labels]
        ax.bar(x, vals, bottom=bottoms, color=color, label=plabel, alpha=0.8)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Avg Iteration Time by Phase')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Speedup
    ax = axes[1]
    baseline_total = results['baseline']['avg']['total']
    speedups = [baseline_total / results[l]['avg']['total'] for l in labels]
    bars = ax.bar(x, speedups, color=['#95a5a6', '#3498db', '#e74c3c'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Total Speedup vs Baseline')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{s:.2f}x', ha='center', fontsize=11, fontweight='bold')

    # Self-play specific speedup
    ax = axes[2]
    baseline_sp = results['baseline']['avg']['self_play']
    sp_speedups = [baseline_sp / results[l]['avg']['self_play'] for l in labels]
    bars = ax.bar(x, sp_speedups, color=['#95a5a6', '#3498db', '#e74c3c'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Self-Play Phase Speedup')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, s in zip(bars, sp_speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{s:.2f}x', ha='center', fontsize=11, fontweight='bold')

    fig.suptitle('Game Parallelism + Virtual Loss Batching — Othello 6x6, 200 sims',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, 'speedup.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Plot 2: Per-phase breakdown detailed ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (phase, plabel) in zip(axes, zip(phases, phase_labels)):
        vals = [results[l]['avg'][phase] for l in labels]
        bars = ax.bar(range(len(labels)), vals,
                      color=['#95a5a6', '#3498db', '#e74c3c'], alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(display_labels, fontsize=8)
        ax.set_ylabel('Time (s)')
        ax.set_title(plabel)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{v:.1f}s', ha='center', fontsize=9)

    fig.suptitle('Per-Phase Time Breakdown', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(FIG_DIR, 'phase_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Plots saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
