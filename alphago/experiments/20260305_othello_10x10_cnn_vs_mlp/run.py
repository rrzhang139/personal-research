#!/usr/bin/env python3
"""Experiment: CNN vs MLP on 10x10 Othello.

Same comparison structure as the 6x6 experiment (20260304_othello_cnn_vs_mlp)
but on a larger board where spatial patterns should matter more.

Config mirrors 6x6 experiment except:
- Board: 10x10 (100 cells, 101 actions) vs 6x6 (36 cells, 37 actions)
- Sims: 100 (up from 50, bigger board needs more search)
- MLP: 4x256 (up from 4x128, more capacity for larger input)
- Both use nn_batch_size=8 for GPU batching

Usage:
    python experiments/20260305_othello_10x10_cnn_vs_mlp/run.py          # full (GPU)
    python experiments/20260305_othello_10x10_cnn_vs_mlp/run.py --quick  # smoke test
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

BOARD_SIZE = 10


def make_config(arch, num_iterations, games_per_iteration, num_simulations):
    """Build config for a given architecture on 10x10 Othello."""
    if arch == 'mlp':
        return AlphaZeroConfig(
            mcts=MCTSConfig(num_simulations=num_simulations, nn_batch_size=8),
            network=NetworkConfig(
                network_type='mlp', hidden_size=256, num_layers=4,
            ),
            training=TrainingConfig(
                num_iterations=num_iterations,
                games_per_iteration=games_per_iteration,
                checkpoint_dir=os.path.join(DATA_DIR, 'mlp'),
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
                checkpoint_dir=os.path.join(DATA_DIR, 'othellonet'),
            ),
            arena=ArenaConfig(arena_games=40, update_threshold=0.55),
            game='othello10', seed=42,
        )


def train_arch(name, config):
    """Train one architecture, return (history, best_model)."""
    game = Othello(size=BOARD_SIZE)
    model = create_model(game, config.network, lr=config.training.lr)

    print(f"\n{'#' * 70}")
    print(f"#  Training: {name} on {BOARD_SIZE}x{BOARD_SIZE} Othello")
    print(f"#  {config.training.num_iterations} iters, "
          f"{config.mcts.num_simulations} sims, "
          f"{config.training.games_per_iteration} games/iter")
    print(f"{'#' * 70}")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    elapsed = time.time() - t0
    history['wall_time'] = elapsed

    # Save history
    hist_path = os.path.join(DATA_DIR, f'{name}_history.json')
    with open(hist_path, 'w') as f:
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
        print("  matplotlib not available, skipping plots")
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
    if h2h_stats:
        cats = ['CNN wins', 'Draws', 'MLP wins']
        vals = [h2h_stats['new_wins'], h2h_stats['draws'], h2h_stats['old_wins']]
        bar_colors = [colors['othellonet'], '#95a5a6', colors['mlp']]
        bars = ax.bar(cats, vals, color=bar_colors, alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha='center', fontweight='bold')
        ax.set_ylabel('Games')
        ax.set_title(f'Head-to-Head ({sum(vals)} games)')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.set_visible(False)

    fig.suptitle(f'Othello {BOARD_SIZE}x{BOARD_SIZE}: CNN vs MLP', fontsize=14,
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

    report = f"""# Experiment: CNN vs MLP on {BOARD_SIZE}x{BOARD_SIZE} Othello

## Hypothesis

On 6x6, MLP won head-to-head 40-0 despite CNN having lower loss. MLP produces
sharper priors (entropy 0.42 vs 0.61), leading to deeper MCTS search (7.1 vs 4.8).

On {BOARD_SIZE}x{BOARD_SIZE}, CNN's spatial inductive bias should become advantageous because:
1. Larger board ({BOARD_SIZE*BOARD_SIZE} cells) has richer spatial patterns (corners, edges, diagonals)
2. MLP with flat {BOARD_SIZE*BOARD_SIZE}-dim input struggles to learn spatial relationships
3. {BOARD_SIZE*BOARD_SIZE+1} actions make sharp priors harder without spatial structure

## Setup

- **Board**: {BOARD_SIZE}x{BOARD_SIZE} Othello ({BOARD_SIZE*BOARD_SIZE} cells, {BOARD_SIZE*BOARD_SIZE+1} actions)
- **Reference**: 6x6 experiment (20260304_othello_cnn_vs_mlp)

### Config
| Parameter | MLP | OthelloNNet |
|-----------|-----|-------------|
| network_type | mlp | othellonet |
| hidden_size / num_filters | 256 | 512 |
| num_layers | 4 | — |
| dropout | 0.0 | 0.3 |
| nn_batch_size | 8 | 8 |
| buffer_strategy | fifo | window (20) |
| num_simulations | {mlp.get('num_sims', 100)} | {cnn.get('num_sims', 100)} |
| num_iterations | {mlp_total} | {cnn_total} |
| games_per_iteration | 100 | 100 |

## Results

### vs Random Win Rate
| Metric | MLP | OthelloNNet |
|--------|-----|-------------|
| Final vs Random | {mlp_final:.0%} | {cnn_final:.0%} |
| Peak vs Random | {mlp_peak:.0%} | {cnn_peak:.0%} |
| Models Accepted | {mlp_acc}/{mlp_total} | {cnn_acc}/{cnn_total} |
| Training Time | {mlp['wall_time']:.0f}s ({mlp['wall_time']/60:.1f}m) | {cnn['wall_time']:.0f}s ({cnn['wall_time']/60:.1f}m) |

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

### Comparison with 6x6 Results
| Metric | 6x6 MLP | 6x6 CNN | {BOARD_SIZE}x{BOARD_SIZE} MLP | {BOARD_SIZE}x{BOARD_SIZE} CNN |
|--------|---------|---------|------------|------------|
| vs Random | 100% | 100% | {mlp_final:.0%} | {cnn_final:.0%} |
| H2H Winner | MLP 40-0 | — | {'CNN' if h2h_stats['new_wins'] > h2h_stats['old_wins'] else 'MLP' if h2h_stats['old_wins'] > h2h_stats['new_wins'] else 'Draw'} {max(h2h_stats['new_wins'], h2h_stats['old_wins'])}-{min(h2h_stats['new_wins'], h2h_stats['old_wins'])} | — |
| Entropy | 0.42 | 0.61 | {mlp['policy_entropy'][-1]:.2f} | {cnn['policy_entropy'][-1]:.2f} |
| Depth | 7.1 | 4.8 | {mlp['mean_search_depth'][-1]:.1f} | {cnn['mean_search_depth'][-1]:.1f} |

See `figures/comparison.png` for learning curves.

## Analysis

{"CNN wins head-to-head — spatial inductive bias helps on larger board as hypothesized." if h2h_stats['new_wins'] > h2h_stats['old_wins'] else "MLP still wins head-to-head — spatial inductive bias not sufficient advantage even on larger board." if h2h_stats['old_wins'] > h2h_stats['new_wins'] else "Even match — board size partially closes the gap."}

## Next Steps

1. Scale to standard 8x8 Othello for direct comparison with literature
2. Try higher sim counts (200, 400) to see if CNN benefits more from deeper search
3. Auxiliary training targets (ownership prediction) may help CNN leverage spatial structure

Total experiment time: {total_time / 60:.1f} minutes ({total_time / 3600:.1f} hours).
"""
    path = os.path.join(EXPERIMENT_DIR, 'report.md')
    with open(path, 'w') as f:
        f.write(report)
    print(f"  Report saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test (2 iters, 10 games, 25 sims)')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    if args.quick:
        n_iters, n_games, n_sims = 2, 10, 25
    else:
        n_iters, n_games, n_sims = 25, 100, 100

    t_total = time.time()

    # Train both architectures
    results = {}
    models = {}
    for arch in ['mlp', 'othellonet']:
        config = make_config(arch, n_iters, n_games, n_sims)
        results[arch], models[arch] = train_arch(arch, config)

    # Head-to-head
    game = Othello(size=BOARD_SIZE)
    h2h_mcts = MCTSConfig(num_simulations=n_sims, nn_batch_size=8)
    h2h_games = 40 if not args.quick else 10

    print(f"\n{'=' * 60}")
    print(f"  Head-to-head: OthelloNNet vs MLP ({h2h_games} games)")
    print(f"{'=' * 60}")

    h2h_wr, h2h_stats = arena_compare(
        game, models['othellonet'], models['mlp'],
        h2h_mcts, num_games=h2h_games,
    )
    print(f"  CNN: {h2h_stats['new_wins']}W / {h2h_stats['draws']}D / "
          f"{h2h_stats['old_wins']}L  (CNN wr: {h2h_wr:.0%})")

    # Save head-to-head
    with open(os.path.join(DATA_DIR, 'head_to_head.json'), 'w') as f:
        json.dump({
            'cnn_wins': h2h_stats['new_wins'],
            'mlp_wins': h2h_stats['old_wins'],
            'draws': h2h_stats['draws'],
            'cnn_win_rate': h2h_wr,
        }, f, indent=2)

    total_time = time.time() - t_total

    # Plots and report
    generate_plots(results, h2h_stats)
    write_report(results, h2h_wr, h2h_stats, total_time)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  RESULTS SUMMARY — {BOARD_SIZE}x{BOARD_SIZE} Othello")
    print(f"{'=' * 60}")
    for name, h in results.items():
        final = h['vs_random_win_rate'][-1]
        peak = max(h['vs_random_win_rate'])
        acc = sum(h['model_accepted'])
        total = len(h['model_accepted'])
        t = h['wall_time']
        ts = f"{t:.0f}s" if t < 120 else f"{t / 60:.1f}m"
        depth = h['mean_search_depth'][-1]
        ent = h['policy_entropy'][-1]
        print(f"  {name:<14} final={final:.0%}  peak={peak:.0%}  "
              f"accepted={acc}/{total}  depth={depth:.1f}  "
              f"entropy={ent:.2f}  time={ts}")
    print(f"\n  Head-to-head: CNN {h2h_stats['new_wins']}W / "
          f"{h2h_stats['draws']}D / {h2h_stats['old_wins']}L vs MLP "
          f"(CNN win rate: {h2h_wr:.0%})")
    print(f"  Total time: {total_time / 60:.1f}m ({total_time / 3600:.1f}h)")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
