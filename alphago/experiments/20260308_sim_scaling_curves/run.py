#!/usr/bin/env python3
"""Experiment: Sim scaling curves — how much does MCTS search help each architecture?

For both 6x6 and 10x10, take the trained CNN and MLP models and play them
head-to-head at varying sim counts: [0, 1, 5, 25, 50, 100, 200].

Zero-sim (0) uses the raw network policy with no MCTS — pure learned knowledge.
This reveals whether CNN's advantage comes from better representations or from
enabling deeper search.

All eval-only — no training. Uses existing weights. CPU is fine.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games.othello import Othello
from alpha_go.neural_net import create_model
from alpha_go.training.arena import arena_compare, play_vs_random
from alpha_go.utils.config import MCTSConfig, NetworkConfig

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXPERIMENT_DIR, 'data')
FIG_DIR = os.path.join(EXPERIMENT_DIR, 'figures')

# Weight paths
WEIGHTS = {
    '6x6': {
        'mlp': os.path.join(os.path.dirname(EXPERIMENT_DIR),
                            '20260304_othello_cnn_vs_mlp', 'data', 'mlp', 'best.pt'),
        'cnn': os.path.join(os.path.dirname(EXPERIMENT_DIR),
                            '20260304_othello_cnn_vs_mlp', 'data', 'othellonet', 'best.pt'),
    },
    '10x10': {
        'mlp': os.path.join(os.path.dirname(EXPERIMENT_DIR),
                            '20260305_othello_10x10_cnn_vs_mlp', 'data', 'mlp', 'best.pt'),
        'cnn': os.path.join(os.path.dirname(EXPERIMENT_DIR),
                            '20260306_othello_10x10_cnn', 'cnn_data', 'othellonet', 'best.pt'),
    },
}


def load_models(board_size):
    """Load CNN and MLP models for a given board size."""
    game = Othello(size=board_size)

    if board_size == 6:
        mlp_cfg = NetworkConfig(network_type='mlp', hidden_size=128, num_layers=4)
        cnn_cfg = NetworkConfig(network_type='othellonet', num_filters=512, dropout=0.3)
    else:
        mlp_cfg = NetworkConfig(network_type='mlp', hidden_size=256, num_layers=4)
        cnn_cfg = NetworkConfig(network_type='othellonet', num_filters=512, dropout=0.3)

    key = f'{board_size}x{board_size}'
    mlp = create_model(game, mlp_cfg)
    mlp.load(WEIGHTS[key]['mlp'])
    cnn = create_model(game, cnn_cfg)
    cnn.load(WEIGHTS[key]['cnn'])

    return game, mlp, cnn


def play_zero_sim_game(game, model1, model2, board_size):
    """Play a single game using raw network policy (no MCTS). Greedy argmax."""
    state = game.get_initial_state()
    player = 1
    models = {1: model1, -1: model2}

    for _ in range(board_size * board_size * 2):
        canonical = game.get_canonical_state(state, player)
        policy, _ = models[player].predict(canonical)

        valid = game.get_valid_moves(state, player)
        policy = policy * valid
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            policy = valid / valid.sum()

        action = np.argmax(policy)
        state = game.get_next_state(state, action, player)
        is_terminal, value = game.check_terminal(state, action, player)
        if is_terminal:
            # value is from perspective of player who just moved
            if value > 0:
                return player
            elif value < 0:
                return -player
            else:
                return 0
        player = -player
    return 0  # shouldn't happen


def zero_sim_compare(game, model_new, model_old, num_games, board_size):
    """Head-to-head with zero sims (raw network policy, greedy)."""
    new_wins = 0
    old_wins = 0
    draws = 0

    for i in range(num_games):
        # Alternate who plays first
        if i % 2 == 0:
            result = play_zero_sim_game(game, model_new, model_old, board_size)
            if result == 1:
                new_wins += 1
            elif result == -1:
                old_wins += 1
            else:
                draws += 1
        else:
            result = play_zero_sim_game(game, model_old, model_new, board_size)
            if result == 1:
                old_wins += 1
            elif result == -1:
                new_wins += 1
            else:
                draws += 1

    total = new_wins + draws * 0.5
    wr = total / num_games if num_games > 0 else 0
    return wr, {'new_wins': new_wins, 'old_wins': old_wins, 'draws': draws}


def run_scaling(board_size, sim_counts, num_games=40):
    """Run head-to-head at each sim count for a given board size."""
    print(f"\n{'=' * 60}")
    print(f"  Sim Scaling: {board_size}x{board_size} Othello")
    print(f"  Sims: {sim_counts}, {num_games} games each")
    print(f"{'=' * 60}")

    game, mlp, cnn = load_models(board_size)
    results = {}

    for sims in sim_counts:
        t0 = time.time()
        if sims == 0:
            # Zero-sim: raw network policy
            wr, stats = zero_sim_compare(game, cnn, mlp, num_games, board_size)
        else:
            mcts_cfg = MCTSConfig(num_simulations=sims, dirichlet_epsilon=0.0)
            wr, stats = arena_compare(game, cnn, mlp, mcts_cfg, num_games=num_games)
        elapsed = time.time() - t0

        results[sims] = {
            'sims': sims,
            'cnn_wins': stats['new_wins'],
            'mlp_wins': stats['old_wins'],
            'draws': stats['draws'],
            'cnn_win_rate': wr,
            'time': elapsed,
        }

        print(f"  {sims:>4d} sims: CNN {stats['new_wins']:2d}W / "
              f"{stats['draws']:2d}D / {stats['old_wins']:2d}L  "
              f"(CNN wr: {wr:.0%})  [{elapsed:.1f}s]")

    return results


def plot_scaling(results_6x6, results_10x10):
    """Plot sim scaling curves for both board sizes."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (title, results, board) in zip(axes[:2], [
        ('6x6 Othello', results_6x6, 6),
        ('10x10 Othello', results_10x10, 10),
    ]):
        sims = sorted(results.keys(), key=int)
        cnn_wr = [results[s]['cnn_win_rate'] * 100 for s in sims]
        mlp_wr = [(1 - results[s]['cnn_win_rate']) * 100 for s in sims]

        # Use log scale but handle 0 sims
        labels = [str(s) for s in sims]

        ax.plot(range(len(sims)), cnn_wr, 'o-', color='#2980b9', linewidth=2,
                markersize=8, label='CNN', zorder=3)
        ax.plot(range(len(sims)), mlp_wr, 's-', color='#e74c3c', linewidth=2,
                markersize=8, label='MLP', zorder=3)
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
        ax.fill_between(range(len(sims)), 50, cnn_wr, alpha=0.1, color='#2980b9')
        ax.fill_between(range(len(sims)), mlp_wr, 50, alpha=0.1, color='#e74c3c')

        ax.set_xticks(range(len(sims)))
        ax.set_xticklabels(labels)
        ax.set_xlabel('MCTS Simulations')
        ax.set_ylabel('Win Rate %')
        ax.set_title(f'{title}: CNN vs MLP by Sim Count')
        ax.set_ylim(-5, 105)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    # Panel 3: Stacked bar showing wins/draws/losses at each sim count for 10x10
    ax = axes[2]
    results = results_10x10
    sims = sorted(results.keys(), key=int)
    cnn_w = [results[s]['cnn_wins'] for s in sims]
    draws = [results[s]['draws'] for s in sims]
    mlp_w = [results[s]['mlp_wins'] for s in sims]

    x = range(len(sims))
    ax.bar(x, cnn_w, color='#2980b9', alpha=0.8, label='CNN wins')
    ax.bar(x, draws, bottom=cnn_w, color='#95a5a6', alpha=0.8, label='Draws')
    ax.bar(x, mlp_w, bottom=[c + d for c, d in zip(cnn_w, draws)],
           color='#e74c3c', alpha=0.8, label='MLP wins')

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in sims])
    ax.set_xlabel('MCTS Simulations')
    ax.set_ylabel('Games (out of 40)')
    ax.set_title('10x10: Game Outcomes by Sim Count')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'sim_scaling.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved to {path}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    sim_counts = [0, 1, 5, 25, 50, 100, 200]
    num_games = 40

    # Check weights exist
    for board_key, paths in WEIGHTS.items():
        for arch, path in paths.items():
            if not os.path.exists(path):
                print(f"  ERROR: Missing {board_key} {arch} weights: {path}")
                return

    # Save config
    with open(os.path.join(EXPERIMENT_DIR, 'config.json'), 'w') as f:
        json.dump({
            'experiment': 'Sim scaling curves: CNN vs MLP at varying search depth',
            'sim_counts': sim_counts,
            'num_games': num_games,
            'boards': ['6x6', '10x10'],
            'weights': {k: {a: os.path.basename(p) for a, p in v.items()}
                        for k, v in WEIGHTS.items()},
        }, f, indent=2)

    t_total = time.time()

    # Run both board sizes
    results_6x6 = run_scaling(6, sim_counts, num_games)
    results_10x10 = run_scaling(10, sim_counts, num_games)

    total_time = time.time() - t_total

    # Save results
    all_results = {
        '6x6': {str(k): v for k, v in results_6x6.items()},
        '10x10': {str(k): v for k, v in results_10x10.items()},
    }
    with open(os.path.join(DATA_DIR, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Plot
    plot_scaling(results_6x6, results_10x10)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — Sim Scaling Curves")
    print(f"{'=' * 60}")
    for board, results in [('6x6', results_6x6), ('10x10', results_10x10)]:
        print(f"\n  {board}:")
        for s in sorted(results.keys(), key=int):
            r = results[s]
            print(f"    {s:>4d} sims: CNN {r['cnn_wins']:2d}W / "
                  f"{r['draws']:2d}D / {r['mlp_wins']:2d}L  ({r['cnn_win_rate']:.0%})")
    print(f"\n  Total time: {total_time / 60:.1f}m")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
