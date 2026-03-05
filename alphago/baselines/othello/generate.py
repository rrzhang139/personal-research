#!/usr/bin/env python3
"""Generate Othello 6x6 baselines: MLP and OthelloNNet.

Runs both architectures with shared training config, saves results to
mlp/ and othellonet/ subdirectories, and generates a comparison plot.

Usage:
    python baselines/othello/generate.py                    # full run (25 iters each)
    python baselines/othello/generate.py --num-iterations 2 # smoke test
"""

import argparse
import json
import os
import shutil
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games import get_game
from alpha_go.neural_net import create_model
from alpha_go.training.pipeline import run_pipeline, save_training_plots
from alpha_go.utils.config import (
    AlphaZeroConfig, ArenaConfig, MCTSConfig, NetworkConfig, TrainingConfig,
)

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))

# Shared training parameters (same as other baselines)
SHARED = dict(
    num_simulations=50,
    c_puct=1.0,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    lr=0.001,
    batch_size=64,
    epochs_per_iteration=10,
    num_iterations=25,
    games_per_iteration=100,
    arena_games=40,
    update_threshold=0.55,
    max_buffer_size=50_000,
    seed=42,
)

# Architecture-specific configs
CONFIGS = {
    'mlp': {
        'network': NetworkConfig(
            network_type='mlp',
            hidden_size=128,
            num_layers=4,
        ),
        'mcts_extra': {},
        'training_extra': {},
        'args_json': {
            'game': 'othello',
            'network_type': 'mlp',
            'hidden_size': 128,
            'num_layers': 4,
            **{k: v for k, v in SHARED.items()},
        },
    },
    'othellonet': {
        'network': NetworkConfig(
            network_type='othellonet',
            num_filters=512,
            dropout=0.3,
        ),
        'mcts_extra': {'nn_batch_size': 8},
        'training_extra': {
            'buffer_strategy': 'window',
            'buffer_window': 20,
        },
        'args_json': {
            'game': 'othello',
            'network_type': 'othellonet',
            'num_filters': 512,
            'dropout': 0.3,
            'nn_batch_size': 8,
            'buffer_strategy': 'window',
            'buffer_window': 20,
            **{k: v for k, v in SHARED.items()},
        },
    },
}


def run_baseline(name: str, num_iterations: int = None) -> dict:
    """Run a single baseline and save results to its subdirectory."""
    cfg = CONFIGS[name]
    out_dir = os.path.join(BASELINE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    iters = num_iterations or SHARED['num_iterations']

    config = AlphaZeroConfig(
        mcts=MCTSConfig(
            num_simulations=SHARED['num_simulations'],
            c_puct=SHARED['c_puct'],
            dirichlet_alpha=SHARED['dirichlet_alpha'],
            dirichlet_epsilon=SHARED['dirichlet_epsilon'],
            **cfg['mcts_extra'],
        ),
        network=cfg['network'],
        training=TrainingConfig(
            lr=SHARED['lr'],
            batch_size=SHARED['batch_size'],
            epochs_per_iteration=SHARED['epochs_per_iteration'],
            num_iterations=iters,
            games_per_iteration=SHARED['games_per_iteration'],
            max_buffer_size=SHARED['max_buffer_size'],
            checkpoint_dir=out_dir,
            **cfg['training_extra'],
        ),
        arena=ArenaConfig(
            arena_games=SHARED['arena_games'],
            update_threshold=SHARED['update_threshold'],
        ),
        game='othello',
        seed=SHARED['seed'],
        num_workers=1,
    )

    # Save args.json
    args = dict(cfg['args_json'])
    args['num_iterations'] = iters
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(args, f, indent=2)

    game = get_game('othello')
    model = create_model(game, config.network, lr=config.training.lr)

    print(f"\n{'#' * 70}")
    print(f"#  Othello 6x6 baseline: {name}")
    print(f"#  {iters} iterations, {SHARED['num_simulations']} sims, "
          f"{SHARED['games_per_iteration']} games/iter")
    print(f"{'#' * 70}")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    elapsed = time.time() - t0

    history['wall_time'] = elapsed

    # Pipeline already saves best.pt, history.json, and training_curves.png
    # to checkpoint_dir (= out_dir). Move training_curves from figures/ subdir.
    fig_src = os.path.join(out_dir, 'figures', 'training_curves.png')
    fig_dst = os.path.join(out_dir, 'training_curves.png')
    if os.path.exists(fig_src):
        shutil.copy2(fig_src, fig_dst)

    # Summary
    final_wr = history['vs_random_win_rate'][-1]
    best_wr = max(history['vs_random_win_rate'])
    accepted = sum(history['model_accepted'])
    total = len(history['model_accepted'])
    time_str = f"{elapsed:.0f}s" if elapsed < 120 else f"{elapsed / 60:.1f}m"

    print(f"\n  {name}: {final_wr:.0%} final, {best_wr:.0%} peak, "
          f"{accepted}/{total} accepted, {time_str}")

    return history


def generate_comparison_plot(results: dict):
    """Generate side-by-side comparison plot of all architectures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping comparison plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {'mlp': '#e74c3c', 'othellonet': '#2980b9'}
    labels = {'mlp': 'MLP (4x128)', 'othellonet': 'OthelloNNet (512f)'}

    # vs Random win rate
    ax = axes[0, 0]
    for name, history in results.items():
        iters = history['iteration']
        wr = [v * 100 for v in history['vs_random_win_rate']]
        ax.plot(iters, wr, color=colors[name], marker='o', markersize=3,
                linewidth=1.5, label=labels[name])
    ax.axhline(y=95, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Win Rate %')
    ax.set_xlabel('Iteration')
    ax.set_title('vs Random Player')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Total loss
    ax = axes[0, 1]
    for name, history in results.items():
        iters = history['iteration']
        ax.plot(iters, history['total_loss'], color=colors[name],
                linewidth=1.5, label=labels[name])
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Policy entropy
    ax = axes[1, 0]
    for name, history in results.items():
        iters = history['iteration']
        ax.plot(iters, history['policy_entropy'], color=colors[name],
                marker='o', markersize=3, linewidth=1.5, label=labels[name])
    ax.set_ylabel('Entropy (nats)')
    ax.set_xlabel('Iteration')
    ax.set_title('MCTS Policy Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Arena acceptance rate (bar chart)
    ax = axes[1, 1]
    bar_width = 0.35
    for i, (name, history) in enumerate(results.items()):
        iters = history['iteration']
        accepted = [int(a) for a in history['model_accepted']]
        offset = (i - 0.5) * bar_width
        x = [it + offset for it in iters]
        ax.bar(x, accepted, width=bar_width, color=colors[name],
               alpha=0.7, label=labels[name])
    ax.set_ylabel('Accepted (1/0)')
    ax.set_xlabel('Iteration')
    ax.set_title('Model Acceptance')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Othello 6x6 — MLP vs OthelloNNet', fontsize=14,
                 fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = os.path.join(BASELINE_DIR, 'comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Comparison plot saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Othello baselines')
    parser.add_argument('--num-iterations', type=int, default=None,
                        help='Override number of iterations (default: 25)')
    parser.add_argument('--only', choices=['mlp', 'othellonet'],
                        help='Run only one architecture')
    args = parser.parse_args()

    archs = [args.only] if args.only else ['mlp', 'othellonet']
    results = {}

    for arch in archs:
        results[arch] = run_baseline(arch, num_iterations=args.num_iterations)

    if len(results) > 1:
        generate_comparison_plot(results)

    # Print summary table
    print(f"\n{'=' * 60}")
    print(f"  Othello 6x6 Baselines Summary")
    print(f"{'=' * 60}")
    print(f"  {'Arch':<14} {'Final':>6} {'Peak':>6} {'Accepted':>10} {'Time':>8}")
    print(f"  {'-' * 50}")
    for name, h in results.items():
        final = h['vs_random_win_rate'][-1]
        peak = max(h['vs_random_win_rate'])
        acc = f"{sum(h['model_accepted'])}/{len(h['model_accepted'])}"
        t = h['wall_time']
        time_str = f"{t:.0f}s" if t < 120 else f"{t / 60:.1f}m"
        print(f"  {name:<14} {final:>5.0%} {peak:>5.0%} {acc:>10} {time_str:>8}")
    print()


if __name__ == '__main__':
    main()
