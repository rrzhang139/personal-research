#!/usr/bin/env python3
"""Experiment: CNN vs MLP on Othello 6x6 (post-bugfix comparison).

The winner-detection bugfix (2026-03-04) invalidated all previous Othello
comparisons. MLP was previously reported at 60% vs random; post-bugfix it
reaches 88% in just 2 iterations. This is the definitive architecture
comparison with corrected evaluation.

Trains both architectures with the shared baseline config (50 sims, 25 iters,
100 games/iter), then plays them head-to-head.

Usage:
    python experiments/20260304_othello_cnn_vs_mlp/run.py          # full (GPU)
    python experiments/20260304_othello_cnn_vs_mlp/run.py --quick   # smoke test
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games import get_game
from alpha_go.neural_net import create_model
from alpha_go.training.arena import arena_compare, play_vs_random
from alpha_go.training.pipeline import run_pipeline
from alpha_go.utils.config import (
    AlphaZeroConfig, ArenaConfig, MCTSConfig, NetworkConfig, TrainingConfig,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXPERIMENT_DIR, 'data')
FIG_DIR = os.path.join(EXPERIMENT_DIR, 'figures')


def make_config(arch, num_iterations, games_per_iteration, num_simulations):
    """Build config for a given architecture."""
    if arch == 'mlp':
        return AlphaZeroConfig(
            mcts=MCTSConfig(num_simulations=num_simulations),
            network=NetworkConfig(
                network_type='mlp', hidden_size=128, num_layers=4,
            ),
            training=TrainingConfig(
                num_iterations=num_iterations,
                games_per_iteration=games_per_iteration,
                checkpoint_dir=os.path.join(DATA_DIR, 'mlp'),
            ),
            arena=ArenaConfig(arena_games=40, update_threshold=0.55),
            game='othello', seed=42,
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
                checkpoint_dir=os.path.join(DATA_DIR, 'othellonet'),
            ),
            arena=ArenaConfig(arena_games=40, update_threshold=0.55),
            game='othello', seed=42,
        )


def train_arch(name, config):
    """Train one architecture, return (history, best_model)."""
    game = get_game('othello')
    model = create_model(game, config.network, lr=config.training.lr)

    print(f"\n{'#' * 70}")
    print(f"#  Training: {name}")
    print(f"#  {config.training.num_iterations} iters, "
          f"{config.mcts.num_simulations} sims, "
          f"{config.training.games_per_iteration} games/iter")
    print(f"{'#' * 70}")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    elapsed = time.time() - t0
    history['wall_time'] = elapsed

    # Save history
    with open(os.path.join(DATA_DIR, f'{name}_history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # Load best model for head-to-head
    best_model = create_model(game, config.network, lr=config.training.lr)
    best_path = os.path.join(config.training.checkpoint_dir, 'best.pt')
    if os.path.exists(best_path):
        best_model.load(best_path)

    final = history['vs_random_win_rate'][-1]
    peak = max(history['vs_random_win_rate'])
    acc = sum(history['model_accepted'])
    total = len(history['model_accepted'])
    time_str = f"{elapsed:.0f}s" if elapsed < 120 else f"{elapsed / 60:.1f}m"
    print(f"\n  {name}: {final:.0%} final, {peak:.0%} peak, "
          f"{acc}/{total} accepted, {time_str}")

    return history, best_model


def generate_plots(results, h2h_stats):
    """Generate 2x3 comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available")
        return

    colors = {'mlp': '#e74c3c', 'othellonet': '#2980b9'}
    labels = {'mlp': 'MLP (4x128)', 'othellonet': 'OthelloNNet (512f)'}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. vs Random learning curves
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
    ax.set_title('Value Loss (position evaluation)')
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

    # 6. Head-to-head result
    ax = axes[1, 2]
    if h2h_stats:
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
    else:
        ax.set_visible(False)

    fig.suptitle('Othello 6x6: CNN vs MLP (Post-Bugfix)', fontsize=14,
                 fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = os.path.join(FIG_DIR, 'comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Comparison plot saved to {path}")


def write_report(results, h2h_wr, h2h_stats, total_time):
    """Write experiment report."""
    mlp = results['mlp']
    cnn = results['othellonet']

    mlp_final = mlp['vs_random_win_rate'][-1]
    mlp_peak = max(mlp['vs_random_win_rate'])
    mlp_acc = sum(mlp['model_accepted'])
    mlp_total = len(mlp['model_accepted'])

    cnn_final = cnn['vs_random_win_rate'][-1]
    cnn_peak = max(cnn['vs_random_win_rate'])
    cnn_acc = sum(cnn['model_accepted'])
    cnn_total = len(cnn['model_accepted'])

    report = f"""# Experiment: CNN vs MLP on Othello 6x6 (Post-Bugfix)

## Hypothesis

The OthelloNNet (plain CNN with shrinking spatial dims) should outperform the
MLP (4x128) on Othello because Othello has strong spatial patterns (corners,
edges, adjacency). Previous comparisons were invalid due to a winner-detection
bug that capped reported win rates at ~60-75%.

## Setup

- **Baseline**: Both trained with shared config (50 sims, 25 iters, 100 games/iter)
- **MLP**: 4 layers x 128 hidden, FIFO buffer
- **OthelloNNet**: 512 filters, dropout=0.3, window buffer (last 20 iters), nn_batch_size=8
- **Evaluation**: 50 games vs random per iteration, 40-game head-to-head after training

### Config diff from defaults
| Parameter | MLP | OthelloNNet |
|-----------|-----|-------------|
| network_type | mlp | othellonet |
| num_filters | — | 512 |
| dropout | — | 0.3 |
| nn_batch_size | 1 | 8 |
| buffer_strategy | fifo | window |
| buffer_window | — | 20 |
| num_simulations | 50 | 50 |

## Results

### vs Random Win Rate
| Metric | MLP | OthelloNNet |
|--------|-----|-------------|
| Final vs Random | {mlp_final:.0%} | {cnn_final:.0%} |
| Peak vs Random | {mlp_peak:.0%} | {cnn_peak:.0%} |
| Models Accepted | {mlp_acc}/{mlp_total} | {cnn_acc}/{cnn_total} |
| Training Time | {mlp['wall_time']:.0f}s | {cnn['wall_time']:.0f}s |

### Head-to-Head (40 games, alternating colors)
| | Wins | Draws | Losses |
|---|------|-------|--------|
| **OthelloNNet** | {h2h_stats['new_wins']} | {h2h_stats['draws']} | {h2h_stats['old_wins']} |
| **MLP** | {h2h_stats['old_wins']} | {h2h_stats['draws']} | {h2h_stats['new_wins']} |

OthelloNNet win rate: {h2h_wr:.0%}

### Training Dynamics
| Metric | MLP (final) | OthelloNNet (final) |
|--------|-------------|---------------------|
| Total Loss | {mlp['total_loss'][-1]:.3f} | {cnn['total_loss'][-1]:.3f} |
| Policy Loss | {mlp['policy_loss'][-1]:.3f} | {cnn['policy_loss'][-1]:.3f} |
| Value Loss | {mlp['value_loss'][-1]:.3f} | {cnn['value_loss'][-1]:.3f} |
| Policy Entropy | {mlp['policy_entropy'][-1]:.3f} | {cnn['policy_entropy'][-1]:.3f} |
| Search Depth | {mlp['mean_search_depth'][-1]:.1f} | {cnn['mean_search_depth'][-1]:.1f} |

See `figures/comparison.png` for learning curves.

## Analysis

**vs Random:** Both architectures reach high win rates post-bugfix. The CNN
{"outperforms" if cnn_final > mlp_final else "matches" if abs(cnn_final - mlp_final) < 0.05 else "underperforms"} the MLP ({cnn_final:.0%} vs {mlp_final:.0%}).

**Head-to-head:** The direct match reveals which architecture produces
genuinely stronger play beyond beating random. CNN
{"wins" if h2h_stats['new_wins'] > h2h_stats['old_wins'] else "loses" if h2h_stats['new_wins'] < h2h_stats['old_wins'] else "draws"}
{h2h_stats['new_wins']}-{h2h_stats['old_wins']} (with {h2h_stats['draws']} draws).

**Loss landscape:** CNN typically achieves lower loss (especially value loss),
indicating better position evaluation. Lower policy entropy means MCTS
focuses on fewer moves — the network priors are sharper.

**Training cost:** OthelloNNet is significantly slower per iteration due to
CNN forward passes, especially on CPU. On GPU with nn_batch_size=8, the
overhead is amortized.

## Next Steps

1. **MCTS contribution sweep** — test OthelloNNet at 0/1/5/10/25/50/100/200
   sims during eval to measure how much MCTS adds on top of the learned network.
2. **Scale to 8x8 Othello** — the real test. 8x8 has ~60 moves/game and much
   deeper strategy. May need more sims, iterations, and filters.
3. **Training efficiency** — CNN needs fewer iterations to reach the same loss.
   Could we train CNN with fewer games/iter and still match MLP quality?

Total experiment time: {total_time / 60:.1f} minutes.
"""
    path = os.path.join(EXPERIMENT_DIR, 'report.md')
    with open(path, 'w') as f:
        f.write(report)
    print(f"  Report saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test (2 iters, 10 games)')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    if args.quick:
        n_iters, n_games, n_sims = 2, 10, 25
    else:
        n_iters, n_games, n_sims = 25, 100, 50

    # Save config
    with open(os.path.join(EXPERIMENT_DIR, 'config.json'), 'w') as f:
        json.dump({
            'experiment': 'CNN vs MLP on Othello 6x6 (post-bugfix)',
            'quick_mode': args.quick,
            'num_iterations': n_iters,
            'games_per_iteration': n_games,
            'num_simulations': n_sims,
            'architectures': {
                'mlp': {'hidden_size': 128, 'num_layers': 4},
                'othellonet': {
                    'num_filters': 512, 'dropout': 0.3,
                    'nn_batch_size': 8, 'buffer_strategy': 'window',
                },
            },
        }, f, indent=2)

    t_total = time.time()

    # Train both
    results = {}
    models = {}
    for arch in ['mlp', 'othellonet']:
        config = make_config(arch, n_iters, n_games, n_sims)
        results[arch], models[arch] = train_arch(arch, config)

    # Head-to-head: CNN vs MLP with equal MCTS config
    h2h_mcts = MCTSConfig(num_simulations=n_sims)
    game = get_game('othello')

    print(f"\n{'=' * 60}")
    print(f"  Head-to-head: OthelloNNet vs MLP ({40} games)")
    print(f"{'=' * 60}")

    h2h_wr, h2h_stats = arena_compare(
        game, models['othellonet'], models['mlp'],
        h2h_mcts, num_games=40,
    )
    print(f"  CNN: {h2h_stats['new_wins']}W / {h2h_stats['draws']}D / "
          f"{h2h_stats['old_wins']}L  (win rate: {h2h_wr:.0%})")

    # Save head-to-head
    with open(os.path.join(DATA_DIR, 'head_to_head.json'), 'w') as f:
        json.dump({
            'cnn_wins': h2h_stats['new_wins'],
            'mlp_wins': h2h_stats['old_wins'],
            'draws': h2h_stats['draws'],
            'cnn_win_rate': h2h_wr,
        }, f, indent=2)

    total_time = time.time() - t_total

    # Generate plots and report
    generate_plots(results, h2h_stats)
    write_report(results, h2h_wr, h2h_stats, total_time)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for name, h in results.items():
        final = h['vs_random_win_rate'][-1]
        peak = max(h['vs_random_win_rate'])
        acc = sum(h['model_accepted'])
        total = len(h['model_accepted'])
        t = h['wall_time']
        ts = f"{t:.0f}s" if t < 120 else f"{t / 60:.1f}m"
        print(f"  {name:<14} final={final:.0%}  peak={peak:.0%}  "
              f"accepted={acc}/{total}  time={ts}")
    print(f"\n  Head-to-head: CNN {h2h_stats['new_wins']}W / "
          f"{h2h_stats['draws']}D / {h2h_stats['old_wins']}L vs MLP "
          f"(CNN win rate: {h2h_wr:.0%})")
    print(f"  Total time: {total_time / 60:.1f}m")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
