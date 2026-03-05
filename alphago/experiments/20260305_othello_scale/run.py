#!/usr/bin/env python3
"""Experiment: CNN vs MLP at scale — sim sweep on 6x6 + training on 10x10.

Part 1: Sim sweep on 6x6 (eval only, uses existing weights)
  - Load best MLP and OthelloNNet from 20260304_othello_cnn_vs_mlp experiment
  - Head-to-head at sims = [50, 100, 200], 40 games each
  - Tests if more sims help CNN close the gap

Part 2: Train + head-to-head on 10x10 (needs GPU)
  - Train both architectures on 10x10 Othello (200 sims, 25 iters)
  - MLP: 4x256 (bigger capacity for 100-cell board)
  - OthelloNNet: 512 filters, dropout=0.3, window buffer
  - Head-to-head after training

Usage:
    python experiments/20260305_othello_scale/run.py                    # full
    python experiments/20260305_othello_scale/run.py --part1-only       # sim sweep only (CPU ok)
    python experiments/20260305_othello_scale/run.py --part2-only       # 10x10 only (GPU)
    python experiments/20260305_othello_scale/run.py --quick            # smoke test
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games.othello import Othello
from alpha_go.neural_net import create_model
from alpha_go.training.arena import arena_compare, play_vs_random
from alpha_go.training.pipeline import run_pipeline
from alpha_go.utils.config import (
    AlphaZeroConfig, ArenaConfig, MCTSConfig, NetworkConfig, TrainingConfig,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXPERIMENT_DIR, 'data')
FIG_DIR = os.path.join(EXPERIMENT_DIR, 'figures')
PREV_EXPERIMENT = os.path.join(os.path.dirname(EXPERIMENT_DIR),
                               '20260304_othello_cnn_vs_mlp', 'data')


# ─── Part 1: Sim sweep on 6x6 using existing weights ───────────────────────

def run_sim_sweep(sim_counts, num_h2h_games=40):
    """Head-to-head CNN vs MLP at different sim counts using existing 6x6 weights."""
    game = Othello(size=6)

    # Load existing weights
    mlp_config = NetworkConfig(network_type='mlp', hidden_size=128, num_layers=4)
    cnn_config = NetworkConfig(network_type='othellonet', num_filters=512, dropout=0.3)

    mlp_model = create_model(game, mlp_config)
    cnn_model = create_model(game, cnn_config)

    mlp_path = os.path.join(PREV_EXPERIMENT, 'mlp', 'best.pt')
    cnn_path = os.path.join(PREV_EXPERIMENT, 'othellonet', 'best.pt')

    if not os.path.exists(mlp_path) or not os.path.exists(cnn_path):
        print(f"  ERROR: Missing weights from previous experiment")
        print(f"    MLP: {mlp_path} {'(found)' if os.path.exists(mlp_path) else '(MISSING)'}")
        print(f"    CNN: {cnn_path} {'(found)' if os.path.exists(cnn_path) else '(MISSING)'}")
        return None

    mlp_model.load(mlp_path)
    cnn_model.load(cnn_path)

    print(f"\n{'=' * 60}")
    print(f"  Part 1: Sim sweep on 6x6 Othello")
    print(f"  Sim counts: {sim_counts}")
    print(f"  {num_h2h_games} games per matchup")
    print(f"{'=' * 60}")

    sweep_results = {}
    for sims in sim_counts:
        mcts_config = MCTSConfig(num_simulations=sims)
        print(f"\n  --- {sims} sims ---")

        t0 = time.time()
        h2h_wr, h2h_stats = arena_compare(
            game, cnn_model, mlp_model, mcts_config, num_games=num_h2h_games,
        )
        elapsed = time.time() - t0

        result = {
            'sims': sims,
            'cnn_wins': h2h_stats['new_wins'],
            'mlp_wins': h2h_stats['old_wins'],
            'draws': h2h_stats['draws'],
            'cnn_win_rate': h2h_wr,
            'time': elapsed,
        }
        sweep_results[sims] = result

        print(f"  CNN: {h2h_stats['new_wins']}W / {h2h_stats['draws']}D / "
              f"{h2h_stats['old_wins']}L  (CNN wr: {h2h_wr:.0%})  [{elapsed:.1f}s]")

    # Save results
    out_path = os.path.join(DATA_DIR, 'sim_sweep_6x6.json')
    with open(out_path, 'w') as f:
        json.dump(sweep_results, f, indent=2, default=str)
    print(f"\n  Sim sweep results saved to {out_path}")

    return sweep_results


# ─── Part 2: Train on 10x10 ────────────────────────────────────────────────

def make_10x10_config(arch, num_iterations, games_per_iteration, num_simulations):
    """Build config for 10x10 Othello training."""
    if arch == 'mlp':
        return AlphaZeroConfig(
            mcts=MCTSConfig(num_simulations=num_simulations, nn_batch_size=8),
            network=NetworkConfig(
                network_type='mlp', hidden_size=256, num_layers=4,
            ),
            training=TrainingConfig(
                num_iterations=num_iterations,
                games_per_iteration=games_per_iteration,
                checkpoint_dir=os.path.join(DATA_DIR, '10x10_mlp'),
            ),
            arena=ArenaConfig(arena_games=40, update_threshold=0.55),
            game='othello10', seed=42,
        )
    else:
        return AlphaZeroConfig(
            mcts=MCTSConfig(num_simulations=num_simulations, nn_batch_size=8),
            network=NetworkConfig(
                network_type='othellonet', num_filters=512, dropout=0.3,
            ),
            training=TrainingConfig(
                num_iterations=num_iterations,
                games_per_iteration=games_per_iteration,
                buffer_strategy='window', buffer_window=20,
                checkpoint_dir=os.path.join(DATA_DIR, '10x10_othellonet'),
            ),
            arena=ArenaConfig(arena_games=40, update_threshold=0.55),
            game='othello10', seed=42,
        )


def train_10x10(arch, config):
    """Train one architecture on 10x10 Othello."""
    game = Othello(size=10)
    model = create_model(game, config.network, lr=config.training.lr)

    print(f"\n{'#' * 70}")
    print(f"#  Training: {arch} on 10x10 Othello")
    print(f"#  {config.training.num_iterations} iters, "
          f"{config.mcts.num_simulations} sims, "
          f"{config.training.games_per_iteration} games/iter")
    print(f"{'#' * 70}")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    elapsed = time.time() - t0
    history['wall_time'] = elapsed

    # Save history
    with open(os.path.join(DATA_DIR, f'10x10_{arch}_history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # Load best model
    best_model = create_model(game, config.network, lr=config.training.lr)
    best_path = os.path.join(config.training.checkpoint_dir, 'best.pt')
    if os.path.exists(best_path):
        best_model.load(best_path)

    final = history['vs_random_win_rate'][-1]
    peak = max(history['vs_random_win_rate'])
    acc = sum(history['model_accepted'])
    total = len(history['model_accepted'])
    time_str = f"{elapsed:.0f}s" if elapsed < 120 else f"{elapsed / 60:.1f}m"
    print(f"\n  {arch}: {final:.0%} final, {peak:.0%} peak, "
          f"{acc}/{total} accepted, {time_str}")

    return history, best_model


def run_10x10_experiment(num_iterations, games_per_iteration, num_simulations):
    """Train both architectures on 10x10 and head-to-head."""
    print(f"\n{'=' * 60}")
    print(f"  Part 2: CNN vs MLP on 10x10 Othello")
    print(f"  {num_iterations} iters, {num_simulations} sims, "
          f"{games_per_iteration} games/iter")
    print(f"{'=' * 60}")

    results = {}
    models = {}
    for arch in ['mlp', 'othellonet']:
        config = make_10x10_config(arch, num_iterations, games_per_iteration,
                                   num_simulations)
        results[arch], models[arch] = train_10x10(arch, config)

    # Head-to-head
    game = Othello(size=10)
    h2h_mcts = MCTSConfig(num_simulations=num_simulations)

    print(f"\n{'=' * 60}")
    print(f"  Head-to-head: OthelloNNet vs MLP on 10x10 (40 games)")
    print(f"{'=' * 60}")

    h2h_wr, h2h_stats = arena_compare(
        game, models['othellonet'], models['mlp'],
        h2h_mcts, num_games=40,
    )
    print(f"  CNN: {h2h_stats['new_wins']}W / {h2h_stats['draws']}D / "
          f"{h2h_stats['old_wins']}L  (CNN wr: {h2h_wr:.0%})")

    # Save
    with open(os.path.join(DATA_DIR, '10x10_head_to_head.json'), 'w') as f:
        json.dump({
            'cnn_wins': h2h_stats['new_wins'],
            'mlp_wins': h2h_stats['old_wins'],
            'draws': h2h_stats['draws'],
            'cnn_win_rate': h2h_wr,
        }, f, indent=2)

    return results, h2h_wr, h2h_stats


# ─── Plotting ──────────────────────────────────────────────────────────────

def plot_sim_sweep(sweep_results):
    """Bar chart of CNN vs MLP wins at each sim count."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available")
        return

    sims = sorted(sweep_results.keys(), key=int)
    cnn_wins = [sweep_results[s]['cnn_wins'] for s in sims]
    mlp_wins = [sweep_results[s]['mlp_wins'] for s in sims]
    draws = [sweep_results[s]['draws'] for s in sims]

    x = range(len(sims))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width for i in x], cnn_wins, width, label='CNN wins',
           color='#2980b9', alpha=0.8)
    ax.bar(list(x), draws, width, label='Draws', color='#95a5a6', alpha=0.8)
    ax.bar([i + width for i in x], mlp_wins, width, label='MLP wins',
           color='#e74c3c', alpha=0.8)

    ax.set_xlabel('MCTS Simulations')
    ax.set_ylabel('Games (out of 40)')
    ax.set_title('6x6 Othello: CNN vs MLP at Different Sim Counts')
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in sims])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    path = os.path.join(FIG_DIR, 'sim_sweep_6x6.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Sim sweep plot saved to {path}")


def plot_10x10_comparison(results, h2h_stats):
    """2x3 comparison plots for 10x10 training."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available")
        return

    colors = {'mlp': '#e74c3c', 'othellonet': '#2980b9'}
    labels = {'mlp': 'MLP (4x256)', 'othellonet': 'OthelloNNet (512f)'}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. vs Random
    ax = axes[0, 0]
    for name, h in results.items():
        wr = [v * 100 for v in h['vs_random_win_rate']]
        ax.plot(h['iteration'], wr, color=colors[name], marker='o',
                markersize=3, linewidth=1.5, label=labels[name])
    ax.axhline(y=95, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Win Rate %')
    ax.set_xlabel('Iteration')
    ax.set_title('vs Random Player')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Total loss
    ax = axes[0, 1]
    for name, h in results.items():
        ax.plot(h['iteration'], h['total_loss'], color=colors[name],
                linewidth=1.5, label=labels[name])
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.set_title('Total Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Value loss
    ax = axes[0, 2]
    for name, h in results.items():
        ax.plot(h['iteration'], h['value_loss'], color=colors[name],
                linewidth=1.5, label=labels[name])
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.set_title('Value Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Policy entropy
    ax = axes[1, 0]
    for name, h in results.items():
        ax.plot(h['iteration'], h['policy_entropy'], color=colors[name],
                marker='o', markersize=3, linewidth=1.5, label=labels[name])
    ax.set_ylabel('Entropy (nats)')
    ax.set_xlabel('Iteration')
    ax.set_title('MCTS Policy Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Search depth
    ax = axes[1, 1]
    for name, h in results.items():
        ax.plot(h['iteration'], h['mean_search_depth'], color=colors[name],
                marker='s', markersize=3, linewidth=1.5, label=labels[name])
    ax.set_ylabel('Depth')
    ax.set_xlabel('Iteration')
    ax.set_title('Mean Search Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Head-to-head
    ax = axes[1, 2]
    cats = ['CNN wins', 'Draws', 'MLP wins']
    vals = [h2h_stats['new_wins'], h2h_stats['draws'], h2h_stats['old_wins']]
    bar_colors = [colors['othellonet'], '#95a5a6', colors['mlp']]
    bars = ax.bar(cats, vals, color=bar_colors, alpha=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha='center', fontweight='bold')
    ax.set_ylabel('Games')
    ax.set_title('Head-to-Head (40 games)')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Othello 10x10: CNN vs MLP', fontsize=14,
                 fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = os.path.join(FIG_DIR, '10x10_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  10x10 comparison plot saved to {path}")


# ─── Report ────────────────────────────────────────────────────────────────

def write_report(sweep_results, results_10x10, h2h_wr_10x10, h2h_stats_10x10,
                 total_time):
    """Write experiment report."""
    report = f"""# Experiment: CNN vs MLP at Scale — Sim Sweep + 10x10 Othello

## Hypothesis

On 6x6, MLP won head-to-head 40-0 despite CNN having lower loss. Two possible
explanations: (1) MLP produces sharper priors at low sim counts, giving deeper
MCTS search. (2) 6x6 is too small for spatial patterns to matter.

This experiment tests both by:
1. **Sim sweep on 6x6**: head-to-head at 50/100/200 sims (existing weights)
2. **10x10 Othello**: train both architectures on a larger board where spatial
   patterns (corners, edges) should matter more

## Part 1: Sim Sweep on 6x6

Using weights from 20260304_othello_cnn_vs_mlp experiment.

| Sims | CNN Wins | Draws | MLP Wins | CNN Win Rate |
|------|----------|-------|----------|--------------|
"""

    if sweep_results:
        for s in sorted(sweep_results.keys(), key=int):
            r = sweep_results[s]
            report += (f"| {r['sims']} | {r['cnn_wins']} | {r['draws']} | "
                       f"{r['mlp_wins']} | {r['cnn_win_rate']:.0%} |\n")

    report += "\n"

    if results_10x10:
        mlp = results_10x10['mlp']
        cnn = results_10x10['othellonet']

        mlp_final = mlp['vs_random_win_rate'][-1]
        mlp_peak = max(mlp['vs_random_win_rate'])
        mlp_acc = sum(mlp['model_accepted'])
        mlp_total = len(mlp['model_accepted'])

        cnn_final = cnn['vs_random_win_rate'][-1]
        cnn_peak = max(cnn['vs_random_win_rate'])
        cnn_acc = sum(cnn['model_accepted'])
        cnn_total = len(cnn['model_accepted'])

        report += f"""## Part 2: 10x10 Othello Training

### Config
| Parameter | MLP | OthelloNNet |
|-----------|-----|-------------|
| network_type | mlp | othellonet |
| hidden_size / num_filters | 256 | 512 |
| num_layers | 4 | — |
| dropout | 0.0 | 0.3 |
| nn_batch_size | 8 | 8 |
| buffer_strategy | fifo | window |
| num_simulations | 200 | 200 |

### vs Random Win Rate
| Metric | MLP | OthelloNNet |
|--------|-----|-------------|
| Final vs Random | {mlp_final:.0%} | {cnn_final:.0%} |
| Peak vs Random | {mlp_peak:.0%} | {cnn_peak:.0%} |
| Models Accepted | {mlp_acc}/{mlp_total} | {cnn_acc}/{cnn_total} |
| Training Time | {mlp['wall_time']:.0f}s | {cnn['wall_time']:.0f}s |

### Head-to-Head (40 games on 10x10)
| | Wins | Draws | Losses |
|---|------|-------|--------|
| **OthelloNNet** | {h2h_stats_10x10['new_wins']} | {h2h_stats_10x10['draws']} | {h2h_stats_10x10['old_wins']} |
| **MLP** | {h2h_stats_10x10['old_wins']} | {h2h_stats_10x10['draws']} | {h2h_stats_10x10['new_wins']} |

OthelloNNet win rate: {h2h_wr_10x10:.0%}

### Training Dynamics
| Metric | MLP (final) | OthelloNNet (final) |
|--------|-------------|---------------------|
| Total Loss | {mlp['total_loss'][-1]:.3f} | {cnn['total_loss'][-1]:.3f} |
| Policy Loss | {mlp['policy_loss'][-1]:.3f} | {cnn['policy_loss'][-1]:.3f} |
| Value Loss | {mlp['value_loss'][-1]:.3f} | {cnn['value_loss'][-1]:.3f} |
| Policy Entropy | {mlp['policy_entropy'][-1]:.3f} | {cnn['policy_entropy'][-1]:.3f} |
| Search Depth | {mlp['mean_search_depth'][-1]:.1f} | {cnn['mean_search_depth'][-1]:.1f} |
"""

    report += f"""
## Analysis

TODO: fill in after results are collected.

Total experiment time: {total_time / 60:.1f} minutes.
"""

    path = os.path.join(EXPERIMENT_DIR, 'report.md')
    with open(path, 'w') as f:
        f.write(report)
    print(f"  Report saved to {path}")


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Smoke test (2 iters, 10 games, 25 sims)')
    parser.add_argument('--part1-only', action='store_true',
                        help='Run sim sweep on 6x6 only (uses existing weights)')
    parser.add_argument('--part2-only', action='store_true',
                        help='Run 10x10 training only (needs GPU)')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    if args.quick:
        sim_counts = [25, 50]
        n_iters, n_games, n_sims = 2, 10, 25
        h2h_games = 10
    else:
        sim_counts = [50, 100, 200]
        n_iters, n_games, n_sims = 25, 100, 200
        h2h_games = 40

    # Save config
    with open(os.path.join(EXPERIMENT_DIR, 'config.json'), 'w') as f:
        json.dump({
            'experiment': 'CNN vs MLP at scale: sim sweep + 10x10',
            'quick_mode': args.quick,
            'part1': {
                'sim_counts': sim_counts,
                'h2h_games': h2h_games,
                'weights_from': '20260304_othello_cnn_vs_mlp',
            },
            'part2': {
                'board_size': 10,
                'num_iterations': n_iters,
                'games_per_iteration': n_games,
                'num_simulations': n_sims,
                'mlp': {'hidden_size': 256, 'num_layers': 4},
                'othellonet': {
                    'num_filters': 512, 'dropout': 0.3,
                    'nn_batch_size': 8, 'buffer_strategy': 'window',
                },
            },
        }, f, indent=2)

    t_total = time.time()

    sweep_results = None
    results_10x10 = None
    h2h_wr_10x10 = None
    h2h_stats_10x10 = None

    # Part 1: Sim sweep on 6x6
    if not args.part2_only:
        sweep_results = run_sim_sweep(sim_counts, num_h2h_games=h2h_games)
        if sweep_results:
            plot_sim_sweep(sweep_results)

    # Part 2: 10x10 training
    if not args.part1_only:
        results_10x10, h2h_wr_10x10, h2h_stats_10x10 = run_10x10_experiment(
            n_iters, n_games, n_sims)
        if results_10x10:
            plot_10x10_comparison(results_10x10, h2h_stats_10x10)

    total_time = time.time() - t_total

    # Report
    write_report(sweep_results, results_10x10, h2h_wr_10x10,
                 h2h_stats_10x10, total_time)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")
    if sweep_results:
        print(f"\n  Part 1: Sim sweep on 6x6")
        for s in sorted(sweep_results.keys(), key=int):
            r = sweep_results[s]
            print(f"    {r['sims']:3d} sims: CNN {r['cnn_wins']}W / "
                  f"{r['draws']}D / {r['mlp_wins']}L  ({r['cnn_win_rate']:.0%})")
    if results_10x10:
        print(f"\n  Part 2: 10x10 Othello")
        for name, h in results_10x10.items():
            final = h['vs_random_win_rate'][-1]
            peak = max(h['vs_random_win_rate'])
            t = h['wall_time']
            ts = f"{t:.0f}s" if t < 120 else f"{t / 60:.1f}m"
            print(f"    {name:<14} final={final:.0%}  peak={peak:.0%}  time={ts}")
        print(f"\n    H2H: CNN {h2h_stats_10x10['new_wins']}W / "
              f"{h2h_stats_10x10['draws']}D / {h2h_stats_10x10['old_wins']}L "
              f"(CNN wr: {h2h_wr_10x10:.0%})")
    print(f"\n  Total time: {total_time / 60:.1f}m")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
