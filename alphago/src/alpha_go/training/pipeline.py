"""Full AlphaZero training pipeline: self-play -> train -> arena -> accept/reject.

This is the main loop. Each iteration:
1. Generate self-play games with the current best model
2. Train a copy of the model on the accumulated replay buffer
3. Pit the new model against the old in an arena
4. If new model wins enough, it becomes the new best model
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import asdict

import numpy as np

from ..games.base_game import Game
from ..utils.config import AlphaZeroConfig
from .arena import arena_compare, play_vs_random
from .parallel import resolve_num_workers
from .self_play import generate_self_play_data
from .trainer import train_on_examples


def run_pipeline(game: Game, model, config: AlphaZeroConfig) -> dict:
    """Run the full AlphaZero training pipeline.

    Returns:
        History dict with per-iteration metrics.
    """
    np.random.seed(config.seed)
    num_workers = resolve_num_workers(config.num_workers)

    # Setup
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    fig_dir = os.path.join(config.training.checkpoint_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    replay_buffer = deque(maxlen=config.training.max_buffer_size)
    history = {
        'iteration': [],
        'total_loss': [],
        'policy_loss': [],
        'value_loss': [],
        'arena_win_rate': [],
        'vs_random_win_rate': [],
        'model_accepted': [],
        'buffer_size': [],
        'self_play_outcomes': [],
        # MCTS diagnostics
        'policy_entropy': [],
        'mean_root_value': [],
        'mean_search_depth': [],
        'mean_game_length': [],
    }

    # wandb setup
    run = None
    if config.use_wandb:
        import wandb
        run = wandb.init(
            project=config.wandb_project,
            config=_config_to_dict(config),
        )

    # Buffer setup: FIFO deque or sliding window of per-iteration lists
    use_window = config.training.buffer_strategy == "window"
    if use_window:
        iteration_history = []  # list of per-iteration example lists

    _print_header(config, num_workers)
    _print_table_header()

    best_model = model
    t_start = time.time()

    for iteration in range(1, config.training.num_iterations + 1):
        t_iter = time.time()

        # 1. Self-play (with optional progressive sims)
        t_phase = time.time()
        mcts_config = config.mcts
        if getattr(config.mcts, 'progressive_sims', False):
            from dataclasses import replace
            progress = (iteration - 1) / max(1, config.training.num_iterations - 1)
            min_s = getattr(config.mcts, 'min_sims', 50)
            current_sims = int(min_s + progress * (config.mcts.num_simulations - min_s))
            mcts_config = replace(config.mcts, num_simulations=current_sims)
        new_examples, sp_stats = generate_self_play_data(
            game=game,
            model=best_model,
            mcts_config=mcts_config,
            num_games=config.training.games_per_iteration,
            augment=True,
            num_workers=num_workers,
            game_name=config.game,
            use_cpp=getattr(config, 'use_cpp_mcts', False),
        )
        if use_window:
            iteration_history.append(new_examples)
            if len(iteration_history) > config.training.buffer_window:
                iteration_history.pop(0)
            training_examples = [ex for batch in iteration_history for ex in batch]
        else:
            replay_buffer.extend(new_examples)
            training_examples = list(replay_buffer)
        t_self_play = time.time() - t_phase

        # 2. Train
        t_phase = time.time()
        new_model = best_model.clone()

        # Apply learning rate schedule
        lr_schedule = getattr(config.training, 'lr_schedule', 'constant')
        if lr_schedule == 'cosine':
            import math
            lr_min = getattr(config.training, 'lr_min', 1e-5)
            progress = (iteration - 1) / max(1, config.training.num_iterations - 1)
            lr = lr_min + 0.5 * (config.training.lr - lr_min) * (1 + math.cos(math.pi * progress))
            for param_group in new_model.optimizer.param_groups:
                param_group['lr'] = lr

        losses = train_on_examples(
            model=new_model,
            examples=training_examples,
            batch_size=config.training.batch_size,
            epochs=config.training.epochs_per_iteration,
        )
        t_train = time.time() - t_phase

        # 3. Arena (skip if arena_games == 0 — always accept, like AlphaZero)
        t_phase = time.time()
        if config.arena.arena_games > 0:
            win_rate, arena_stats = arena_compare(
                game=game,
                new_model=new_model,
                old_model=best_model,
                mcts_config=config.mcts,
                num_games=config.arena.arena_games,
                num_workers=num_workers,
                game_name=config.game,
            )
            accepted = win_rate >= config.arena.update_threshold
        else:
            win_rate = 1.0
            arena_stats = {'new_wins': 0, 'old_wins': 0, 'draws': 0}
            accepted = True
        t_arena = time.time() - t_phase

        # 4. Accept or reject
        if accepted:
            best_model = new_model
            best_model.save(os.path.join(config.training.checkpoint_dir, 'best.pt'))

        # 5. Evaluate vs random
        t_phase = time.time()
        eval_games = getattr(config.arena, 'eval_games', 50)
        if eval_games > 0:
            vs_random = play_vs_random(
                game, best_model, config.mcts, num_games=eval_games,
                num_workers=num_workers, game_name=config.game,
            )
        else:
            vs_random = -1.0  # skipped
        t_eval = time.time() - t_phase

        iter_time = time.time() - t_iter

        # Record history
        history['iteration'].append(iteration)
        history['total_loss'].append(losses['total_loss'])
        history['policy_loss'].append(losses['policy_loss'])
        history['value_loss'].append(losses['value_loss'])
        history['arena_win_rate'].append(win_rate)
        history['vs_random_win_rate'].append(vs_random)
        history['model_accepted'].append(accepted)
        history['buffer_size'].append(len(training_examples))
        history['self_play_outcomes'].append(sp_stats.outcomes_tuple)
        history['policy_entropy'].append(sp_stats.mean_policy_entropy)
        history['mean_root_value'].append(sp_stats.mean_root_value)
        history['mean_search_depth'].append(sp_stats.mean_search_depth)
        history['mean_game_length'].append(sp_stats.mean_game_length)

        # Print iteration row
        _print_iter_row(
            iteration=iteration,
            total=config.training.num_iterations,
            losses=losses,
            arena_stats=arena_stats,
            accepted=accepted,
            vs_random=vs_random,
            buffer_size=len(training_examples),
            sp_stats=sp_stats,
            iter_time=iter_time,
        )

        # Log to wandb
        if run:
            import wandb
            wandb.log({
                # Training loss
                'loss/total': losses['total_loss'],
                'loss/policy': losses['policy_loss'],
                'loss/value': losses['value_loss'],
                # Evaluation
                'eval/vs_random': vs_random,
                'eval/arena_win_rate': win_rate,
                'eval/model_accepted': int(accepted),
                # Arena breakdown
                'arena/new_wins': arena_stats['new_wins'],
                'arena/old_wins': arena_stats['old_wins'],
                'arena/draws': arena_stats['draws'],
                # Self-play outcomes
                'self_play/p1_wins': sp_stats.p1_wins,
                'self_play/p2_wins': sp_stats.p2_wins,
                'self_play/draws': sp_stats.draws,
                'self_play/draw_rate': sp_stats.draws / max(1, sp_stats.p1_wins + sp_stats.p2_wins + sp_stats.draws),
                'self_play/mean_game_length': sp_stats.mean_game_length,
                # MCTS diagnostics
                'mcts/policy_entropy': sp_stats.mean_policy_entropy,
                'mcts/mean_root_value': sp_stats.mean_root_value,
                'mcts/mean_search_depth': sp_stats.mean_search_depth,
                # Infrastructure
                'buffer_size': len(training_examples),
                'iter_time': iter_time,
                # Phase timing
                'time/self_play': t_self_play,
                'time/train': t_train,
                'time/arena': t_arena,
                'time/eval': t_eval,
            })

    # Save final model
    best_model.save(os.path.join(config.training.checkpoint_dir, 'final.pt'))

    # Save history
    _save_history(history, config.training.checkpoint_dir)

    # Generate plots
    plot_path = save_training_plots(history, fig_dir)

    # Upload final plots to wandb
    if run:
        import wandb
        if plot_path and os.path.exists(plot_path):
            wandb.log({'training_curves': wandb.Image(plot_path)})
        run.finish()

    total_time = time.time() - t_start
    n_accepted = sum(history['model_accepted'])
    n_total = config.training.num_iterations

    _print_footer(history, total_time, n_accepted, n_total, plot_path)

    return history


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _print_header(config: AlphaZeroConfig, num_workers: int = 1):
    """Print a compact run header with config and diff from defaults."""
    print()
    print(f"  AlphaZero — {config.game}")
    net = f"{config.network.num_layers}x{config.network.hidden_size} MLP"
    mcts = f"{config.mcts.num_simulations} sims, c_puct={config.mcts.c_puct}"
    workers_str = f"  |  Workers: {num_workers}" if num_workers > 1 else ""
    print(f"  Network: {net}  |  MCTS: {mcts}  |  LR: {config.training.lr}{workers_str}")

    diff = _config_diff(config)
    if diff:
        print(f"  Changed from default:")
        for k, (default, actual) in diff.items():
            print(f"    {k}: {default} -> {actual}")
    print()


def _print_table_header():
    """Print the column headers for the iteration table."""
    header = (
        f"{'Iter':>4}  "
        f"{'Loss':>6} {'pi':>6} {'v':>6}  "
        f"{'Arena':>9}  "
        f"{'':>2}  "
        f"{'vsRand':>6}  "
        f"{'H(pi)':>5}  "
        f"{'Depth':>5}  "
        f"{'Time':>5}"
    )
    print(header)
    print("─" * len(header))


def _print_iter_row(
    iteration, total, losses, arena_stats, accepted,
    vs_random, buffer_size, sp_stats, iter_time,
):
    """Print one compact row of the iteration table."""
    check = "+" if accepted else "-"
    w, l, d = arena_stats['new_wins'], arena_stats['old_wins'], arena_stats['draws']

    row = (
        f"{iteration:>2}/{total:<2} "
        f"{losses['total_loss']:>6.3f} {losses['policy_loss']:>6.3f} {losses['value_loss']:>6.3f}  "
        f"{w:>2}W/{d}D/{l}L  "
        f"{check:>2}  "
        f"{vs_random:>5.0%}  "
        f"{sp_stats.mean_policy_entropy:>5.2f}  "
        f"{sp_stats.mean_search_depth:>5.1f}  "
        f"{iter_time:>4.1f}s"
    )
    print(row)


def _print_footer(history, total_time, n_accepted, n_total, plot_path):
    """Print the training summary after completion."""
    print()
    best_loss = min(history['total_loss'])
    best_vs_random = max(history['vs_random_win_rate'])
    final_vs_random = history['vs_random_win_rate'][-1]

    mins = total_time / 60
    time_str = f"{total_time:.0f}s" if total_time < 120 else f"{mins:.1f}m"

    print(f"  Done in {time_str} | "
          f"Models accepted: {n_accepted}/{n_total} | "
          f"Best loss: {best_loss:.3f}")
    print(f"  vs Random — final: {final_vs_random:.0%}, best: {best_vs_random:.0%}")
    if plot_path:
        print(f"  Plots saved to {plot_path}")
    print()


# ---------------------------------------------------------------------------
# Config diff
# ---------------------------------------------------------------------------

def _config_diff(config: AlphaZeroConfig) -> dict:
    """Compare config against defaults, return only changed fields."""
    defaults = AlphaZeroConfig()
    diff = {}

    sub_configs = [
        ('mcts', config.mcts, defaults.mcts),
        ('network', config.network, defaults.network),
        ('training', config.training, defaults.training),
        ('arena', config.arena, defaults.arena),
    ]

    for prefix, current, default in sub_configs:
        for field_name in vars(default):
            cur_val = getattr(current, field_name)
            def_val = getattr(default, field_name)
            if cur_val != def_val:
                diff[f"{prefix}.{field_name}"] = (def_val, cur_val)

    for field_name in ['game', 'seed']:
        cur_val = getattr(config, field_name)
        def_val = getattr(defaults, field_name)
        if cur_val != def_val:
            diff[field_name] = (def_val, cur_val)

    return diff


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_training_plots(history: dict, output_dir: str) -> str:
    """Generate and save training curves.

    Creates a 3x2 figure with 6 subplots covering losses, strength,
    arena decisions, MCTS diagnostics, and self-play dynamics.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    iters = history['iteration']
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # --- Loss curves ---
    ax = axes[0, 0]
    ax.plot(iters, history['total_loss'], 'k-', label='total', linewidth=1.5)
    ax.plot(iters, history['policy_loss'], 'b--', label='policy', alpha=0.7)
    ax.plot(iters, history['value_loss'], 'r--', label='value', alpha=0.7)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- vs Random win rate ---
    ax = axes[0, 1]
    ax.plot(iters, [v * 100 for v in history['vs_random_win_rate']], 'g-o',
            markersize=3, linewidth=1.5)
    ax.axhline(y=95, color='gray', linestyle=':', alpha=0.5, label='95%')
    ax.set_ylabel('Win Rate %')
    ax.set_xlabel('Iteration')
    ax.set_title('vs Random Player')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Arena win rate with accept/reject ---
    ax = axes[1, 0]
    arena_pct = [v * 100 for v in history['arena_win_rate']]
    colors = ['#2ecc71' if a else '#e74c3c' for a in history['model_accepted']]
    ax.bar(iters, arena_pct, color=colors, alpha=0.7, width=0.8)
    ax.axhline(y=55, color='gray', linestyle='--', alpha=0.5, label='threshold')
    ax.set_ylabel('Win Rate %')
    ax.set_xlabel('Iteration')
    ax.set_title('Arena (green=accepted, red=rejected)')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Policy entropy ---
    ax = axes[1, 1]
    if history.get('policy_entropy'):
        ax.plot(iters, history['policy_entropy'], 'purple', marker='o',
                markersize=3, linewidth=1.5)
        ax.set_ylabel('Entropy (nats)')
        ax.set_xlabel('Iteration')
        ax.set_title('MCTS Policy Entropy (lower = more confident)')
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    # --- Root value + search depth (dual y-axis) ---
    ax = axes[2, 0]
    if history.get('mean_root_value') and history.get('mean_search_depth'):
        color1 = '#2980b9'
        color2 = '#e67e22'
        ax.plot(iters, history['mean_root_value'], color=color1, marker='o',
                markersize=3, linewidth=1.5, label='root value')
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylabel('Root Value', color=color1)
        ax.set_xlabel('Iteration')
        ax.tick_params(axis='y', labelcolor=color1)
        ax.set_title('MCTS Root Value & Search Depth')

        ax2 = ax.twinx()
        ax2.plot(iters, history['mean_search_depth'], color=color2, marker='s',
                 markersize=3, linewidth=1.5, label='search depth')
        ax2.set_ylabel('Depth', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    # --- Self-play outcomes + game length ---
    ax = axes[2, 1]
    if history['self_play_outcomes']:
        sp_wins = [o[0] for o in history['self_play_outcomes']]
        sp_losses = [o[1] for o in history['self_play_outcomes']]
        sp_draws = [o[2] for o in history['self_play_outcomes']]
        ax.stackplot(iters, sp_wins, sp_draws, sp_losses,
                     labels=['P1 wins', 'Draws', 'P2 wins'],
                     colors=['#3498db', '#95a5a6', '#e67e22'], alpha=0.7)

        if history.get('mean_game_length'):
            ax2 = ax.twinx()
            ax2.plot(iters, history['mean_game_length'], 'k-', linewidth=1.5,
                     label='game length')
            ax2.set_ylabel('Avg Game Length')
            ax2.legend(fontsize=8, loc='upper left')

        ax.set_ylabel('Games')
        ax.set_xlabel('Iteration')
        ax.set_title('Self-Play Outcomes & Game Length')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle('AlphaZero Training', fontsize=13, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = os.path.join(output_dir, 'training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_history(history: dict, checkpoint_dir: str):
    """Save history as JSON for later analysis."""
    serializable = {}
    for k, v in history.items():
        if k == 'self_play_outcomes':
            serializable[k] = [list(o) for o in v]
        else:
            serializable[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]

    path = os.path.join(checkpoint_dir, 'history.json')
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)


def _config_to_dict(config: AlphaZeroConfig) -> dict:
    """Flatten config dataclasses into a single dict for logging."""
    d = asdict(config)
    flat = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for k2, v2 in value.items():
                flat[f"{key}/{k2}"] = v2
        else:
            flat[key] = value
    return flat
