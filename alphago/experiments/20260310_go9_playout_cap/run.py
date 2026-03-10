#!/usr/bin/env python3
"""Go 9x9 — playout cap randomization + no arena.

Two key changes from KataGo / AlphaZero papers:
1. Playout cap: 12.5% of moves get full 200-sim search (recorded for training),
   87.5% get cheap 30-sim search (not recorded). Games finish ~2x faster.
2. No arena: always use latest model (AlphaZero dropped arena; 20-game arena
   can't detect small improvements in Go).

Warm-start from 128f best.pt (94% peak vs random).

Estimated: ~5.5h on A4000, ~$0.94
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games.go import Go
from alpha_go.neural_net import create_model
from alpha_go.training.pipeline import run_pipeline
from alpha_go.utils.config import (
    AlphaZeroConfig, ArenaConfig, MCTSConfig, NetworkConfig, TrainingConfig,
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

WARM_START = os.path.join(
    os.path.dirname(__file__), '..', '20260309_go9_128f', 'data', 'checkpoints', 'best.pt'
)

config = AlphaZeroConfig(
    mcts=MCTSConfig(
        num_simulations=200,            # deep full search
        c_puct=1.5,                     # ELF OpenGo found 1.5 optimal
        dirichlet_alpha=0.03,
        dirichlet_epsilon=0.25,
        temp_threshold=20,              # greedy earlier (Go games ~80 moves)
        nn_batch_size=8,
        playout_cap_prob=0.125,         # KataGo: 12.5% full search
        playout_cap_cheap_fraction=0.15, # cheap = 200 * 0.15 = 30 sims
    ),
    network=NetworkConfig(
        network_type='cnn',
        num_filters=128,
        num_res_blocks=4,
        dropout=0.0,
    ),
    training=TrainingConfig(
        lr=0.001,
        batch_size=64,
        epochs_per_iteration=10,
        max_buffer_size=200000,
        buffer_strategy='fifo',
        num_iterations=12,
        games_per_iteration=100,        # 2x more games (playout cap makes them cheaper)
        checkpoint_dir=os.path.join(DATA_DIR, 'checkpoints'),
    ),
    arena=ArenaConfig(
        arena_games=0,                  # NO ARENA — always accept (like AlphaZero)
        update_threshold=0.55,
    ),
    game='go',
    seed=42,
    num_workers=1,
    use_wandb=True,
    wandb_project='alphazero',
)

with open(os.path.join(EXP_DIR, 'config.json'), 'w') as f:
    json.dump({
        'game': 'go9',
        'hypothesis': 'Playout cap + no arena: more games, deeper search, always latest model',
        'changes': ['playout_cap_prob=0.125', 'cheap=30 sims', 'full=200 sims',
                     'no arena', '100 games/iter', 'c_puct=1.5'],
        'warm_start': WARM_START,
        'mcts': {k: getattr(config.mcts, k) for k in config.mcts.__dataclass_fields__},
        'network': {k: getattr(config.network, k) for k in config.network.__dataclass_fields__},
        'training': {k: getattr(config.training, k) for k in config.training.__dataclass_fields__},
        'arena': {k: getattr(config.arena, k) for k in config.arena.__dataclass_fields__},
    }, f, indent=2, default=str)

game = Go(size=9)
model = create_model(game, config.network, lr=config.training.lr)

warm_path = os.path.abspath(WARM_START)
if os.path.exists(warm_path):
    model.load(warm_path)
    print(f"Loaded warm-start weights from {warm_path}")
else:
    print(f"WARNING: warm-start not found: {warm_path}")
    print("Training from scratch!")

total_params = sum(p.numel() for p in model.net.parameters())
cheap_sims = max(1, int(config.mcts.num_simulations * config.mcts.playout_cap_cheap_fraction))
print(f"\n=== Go 9x9 — Playout Cap + No Arena ===")
print(f"Network: CNN {config.network.num_filters}f {config.network.num_res_blocks}res ({total_params:,} params)")
print(f"MCTS: {config.mcts.num_simulations} full / {cheap_sims} cheap sims, "
      f"{config.mcts.playout_cap_prob:.1%} full prob")
print(f"Arena: DISABLED (always accept)")
print(f"Training: {config.training.num_iterations} iters x {config.training.games_per_iteration} games")
print(f"GPU: {__import__('torch').cuda.get_device_name(0) if __import__('torch').cuda.is_available() else 'CPU'}")

start = time.time()
history = run_pipeline(game, model, config)
elapsed = time.time() - start

with open(os.path.join(DATA_DIR, 'history.json'), 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n=== Done in {elapsed/60:.1f} min ===")

try:
    import wandb
    if wandb.run:
        best_path = os.path.join(DATA_DIR, 'checkpoints', 'best.pt')
        if os.path.exists(best_path):
            artifact = wandb.Artifact('go9-playout-cap', type='model')
            artifact.add_file(best_path, name='best.pt')
            wandb.log_artifact(artifact)
            print("Uploaded to wandb artifact 'go9-playout-cap'")
        wandb.finish()
except Exception as e:
    print(f"W&B upload failed: {e}")

print("\n=== Auto-pushing results ===")
try:
    repo_root = os.path.join(EXP_DIR, '..', '..', '..')
    subprocess.run(['git', 'add', '-f', 'alphago/experiments/20260310_go9_playout_cap/'],
                   cwd=repo_root, check=True)
    vs_rand = history['vs_random_win_rate'][-1] if history.get('vs_random_win_rate') else '?'
    best_vs = max(history['vs_random_win_rate']) if history.get('vs_random_win_rate') else '?'
    msg = f'Go 9x9 playout cap: {vs_rand:.0%} final, {best_vs:.0%} peak vs random, {elapsed/60:.0f}m\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>'
    subprocess.run(['git', 'commit', '-m', msg], cwd=repo_root, check=True)
    subprocess.run(['git', 'push'], cwd=repo_root, check=True, capture_output=True)
    print("Results pushed!")
except Exception as e:
    print(f"Git push failed: {e}")
