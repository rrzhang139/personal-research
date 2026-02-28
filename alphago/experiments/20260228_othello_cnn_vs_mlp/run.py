#!/usr/bin/env python3
"""Experiment: CNN vs MLP on Othello (6x6).

Train Othello with ResNet CNN and compare against the existing MLP baseline.
Both use identical hyperparameters (the baseline config: 50 sims, 25 iters,
100 games/iter, lr=0.001).

Hypothesis: The MLP baseline struggles on Othello (60-74% vs random, 5/25
models accepted). Othello has strong spatial structure — flanking, corners,
edges — that convolutions should exploit. A ResNet CNN should:
1. Learn faster (more models accepted)
2. Reach higher win rates vs random
3. Produce lower policy loss (better understanding of move quality)
4. Show deeper search (sharper priors let MCTS explore further)
"""

import json
import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games import get_game
from alpha_go.neural_net import create_model
from alpha_go.training.pipeline import run_pipeline
from alpha_go.utils.config import (
    AlphaZeroConfig, ArenaConfig, MCTSConfig, NetworkConfig, TrainingConfig,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXPERIMENT_DIR, 'data')
FIG_DIR = os.path.join(EXPERIMENT_DIR, 'figures')

# Baseline config — identical for both MLP and CNN
MCTS = MCTSConfig(num_simulations=50)
TRAINING = TrainingConfig(
    num_iterations=25,
    games_per_iteration=100,
    epochs_per_iteration=10,
    batch_size=64,
    lr=0.001,
)
ARENA = ArenaConfig(arena_games=40, update_threshold=0.55)

# MLP config: 4x128 (same as baseline)
MLP_NETWORK = NetworkConfig(
    network_type="mlp",
    hidden_size=128,
    num_layers=4,
)

# CNN config: 4 res blocks, 64 filters (same depth/width spirit)
CNN_NETWORK = NetworkConfig(
    network_type="cnn",
    num_filters=64,
    num_res_blocks=4,
)


def run_single(network_config: NetworkConfig, tag: str, seed: int = 42) -> dict:
    """Run one training session with the given network config."""
    config = AlphaZeroConfig(
        mcts=MCTS,
        network=network_config,
        training=TrainingConfig(
            num_iterations=TRAINING.num_iterations,
            games_per_iteration=TRAINING.games_per_iteration,
            epochs_per_iteration=TRAINING.epochs_per_iteration,
            batch_size=TRAINING.batch_size,
            lr=TRAINING.lr,
            checkpoint_dir=os.path.join(DATA_DIR, tag),
        ),
        arena=ARENA,
        game='othello',
        seed=seed,
    )

    game = get_game('othello')
    model = create_model(game, config.network, lr=config.training.lr)

    t0 = time.time()
    history = run_pipeline(game, model, config)
    elapsed = time.time() - t0

    history['wall_time'] = elapsed
    history['network_type'] = tag
    return history


def main():
    all_results = {}

    # --- Run MLP ---
    print(f"\n{'#'*70}")
    print(f"#  MLP: 4x128 (baseline reproduction)")
    print(f"{'#'*70}")
    mlp_history = run_single(MLP_NETWORK, 'mlp')
    all_results['mlp'] = mlp_history

    path = os.path.join(DATA_DIR, 'mlp.json')
    with open(path, 'w') as f:
        json.dump(mlp_history, f, indent=2, default=str)

    # --- Run CNN ---
    print(f"\n{'#'*70}")
    print(f"#  CNN: 4 res blocks, 64 filters")
    print(f"{'#'*70}")
    cnn_history = run_single(CNN_NETWORK, 'cnn')
    all_results['cnn'] = cnn_history

    path = os.path.join(DATA_DIR, 'cnn.json')
    with open(path, 'w') as f:
        json.dump(cnn_history, f, indent=2, default=str)

    # Generate comparison plots
    plot_comparison(all_results)

    # Print summary
    print_summary(all_results)

    # Save config
    config_path = os.path.join(EXPERIMENT_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'experiment': 'CNN vs MLP on Othello (6x6)',
            'game': 'othello',
            'configs': {
                'mlp': {
                    'network_type': 'mlp',
                    'hidden_size': 128,
                    'num_layers': 4,
                },
                'cnn': {
                    'network_type': 'cnn',
                    'num_filters': 64,
                    'num_res_blocks': 4,
                },
            },
            'shared': {
                'num_simulations': MCTS.num_simulations,
                'num_iterations': TRAINING.num_iterations,
                'games_per_iteration': TRAINING.games_per_iteration,
                'epochs_per_iteration': TRAINING.epochs_per_iteration,
                'batch_size': TRAINING.batch_size,
                'lr': TRAINING.lr,
                'arena_games': ARENA.arena_games,
                'update_threshold': ARENA.update_threshold,
            },
            'seed': 42,
        }, f, indent=2)


def plot_comparison(all_results: dict):
    """Generate side-by-side comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    mlp = all_results['mlp']
    cnn = all_results['cnn']
    iters = mlp['iteration']

    styles = {
        'mlp': {'color': '#e74c3c', 'label': 'MLP (4×128)', 'ls': '-', 'marker': 'o'},
        'cnn': {'color': '#2980b9', 'label': 'CNN (4 res, 64f)', 'ls': '-', 'marker': 's'},
    }

    # --- vs Random win rate ---
    ax = axes[0, 0]
    for key, hist in all_results.items():
        s = styles[key]
        wr = [v * 100 for v in hist['vs_random_win_rate']]
        ax.plot(iters, wr, ls=s['ls'], marker=s['marker'], color=s['color'],
                markersize=3, linewidth=1.5, label=s['label'])
    ax.axhline(y=95, color='gray', linestyle=':', alpha=0.4, label='95% target')
    ax.set_ylabel('Win Rate %')
    ax.set_xlabel('Iteration')
    ax.set_title('vs Random Player')
    ax.set_ylim(30, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Total loss ---
    ax = axes[0, 1]
    for key, hist in all_results.items():
        s = styles[key]
        ax.plot(iters, hist['total_loss'], ls=s['ls'], color=s['color'],
                linewidth=1.5, label=s['label'])
    ax.set_ylabel('Total Loss')
    ax.set_xlabel('Iteration')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Policy loss ---
    ax = axes[0, 2]
    for key, hist in all_results.items():
        s = styles[key]
        ax.plot(iters, hist['policy_loss'], ls=s['ls'], color=s['color'],
                linewidth=1.5, label=s['label'] + ' (policy)')
        ax.plot(iters, hist['value_loss'], ls='--', color=s['color'],
                linewidth=1.0, alpha=0.6, label=s['label'] + ' (value)')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.set_title('Policy vs Value Loss')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Policy entropy ---
    ax = axes[1, 0]
    for key, hist in all_results.items():
        s = styles[key]
        if hist.get('policy_entropy'):
            ax.plot(iters, hist['policy_entropy'], ls=s['ls'], marker=s['marker'],
                    color=s['color'], markersize=3, linewidth=1.5, label=s['label'])
    ax.set_ylabel('Entropy (nats)')
    ax.set_xlabel('Iteration')
    ax.set_title('MCTS Policy Entropy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Search depth ---
    ax = axes[1, 1]
    for key, hist in all_results.items():
        s = styles[key]
        if hist.get('mean_search_depth'):
            ax.plot(iters, hist['mean_search_depth'], ls=s['ls'], marker=s['marker'],
                    color=s['color'], markersize=3, linewidth=1.5, label=s['label'])
    ax.set_ylabel('Mean Depth')
    ax.set_xlabel('Iteration')
    ax.set_title('MCTS Search Depth')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Model acceptance cumulative ---
    ax = axes[1, 2]
    for key, hist in all_results.items():
        s = styles[key]
        cum_accepted = np.cumsum(hist['model_accepted'])
        ax.plot(iters, cum_accepted, ls=s['ls'], marker=s['marker'], color=s['color'],
                markersize=3, linewidth=1.5, label=s['label'])
    ax.set_ylabel('Cumulative Models Accepted')
    ax.set_xlabel('Iteration')
    ax.set_title('Learning Progress (Arena Acceptances)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Othello (6×6): CNN vs MLP — Same Hyperparameters',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(FIG_DIR, 'cnn_vs_mlp.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlots saved to {path}")


def print_summary(all_results: dict):
    """Print a summary table."""
    print(f"\n{'='*90}")
    print("SUMMARY: CNN vs MLP on Othello (6x6)")
    print(f"{'='*90}")
    print(f"{'Network':>12}  {'Final vsRand':>12}  {'Best vsRand':>12}  {'Final Loss':>10}  "
          f"{'Accepted':>8}  {'Final Depth':>11}  {'Time':>8}")
    print("─" * 90)

    for key in ['mlp', 'cnn']:
        h = all_results[key]
        final_wr = h['vs_random_win_rate'][-1]
        best_wr = max(h['vs_random_win_rate'])
        final_loss = h['total_loss'][-1]
        accepted = sum(h['model_accepted'])
        total = len(h['model_accepted'])
        final_depth = h['mean_search_depth'][-1] if h.get('mean_search_depth') else 0
        wall = h['wall_time']

        label = 'MLP 4×128' if key == 'mlp' else 'CNN 4res/64f'
        time_str = f"{wall:.0f}s" if wall < 120 else f"{wall/60:.1f}m"

        print(f"{label:>12}  {final_wr:>11.0%}  {best_wr:>11.0%}  {final_loss:>10.3f}  "
              f"{accepted:>3}/{total:<3}  {final_depth:>11.1f}  {time_str:>8}")

    print()


if __name__ == '__main__':
    main()
