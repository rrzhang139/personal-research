#!/usr/bin/env python3
"""Go 9x9 — 100 sims + warm-start from 128f best.pt.

Hypothesis: 50 sims gives depth 2.6, not enough to find tactical captures.
100 sims should deepen search. Warm-start from 94% peak model so we don't
waste iterations re-learning basic Go.

Changes from 128f run:
- num_simulations: 50 -> 100 (2x search depth)
- num_iterations: 25 -> 15 (100 sims = 2x slower, fit in ~9h)
- Warm-start from experiments/20260309_go9_128f/data/checkpoints/best.pt

Estimated: ~9h on A4000, ~$1.53
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
        num_simulations=100,        # 2x from 128f run
        c_puct=1.0,
        dirichlet_alpha=0.03,
        dirichlet_epsilon=0.25,
        temp_threshold=30,
        nn_batch_size=8,
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
        max_buffer_size=150000,
        buffer_strategy='fifo',
        num_iterations=15,          # fewer iters (100 sims = 2x slower)
        games_per_iteration=50,
        checkpoint_dir=os.path.join(DATA_DIR, 'checkpoints'),
    ),
    arena=ArenaConfig(
        arena_games=20,
        update_threshold=0.55,
    ),
    game='go',
    seed=42,
    num_workers=1,
    use_wandb=True,
    wandb_project='alphazero',
)

# Save config
with open(os.path.join(EXP_DIR, 'config.json'), 'w') as f:
    json.dump({
        'game': 'go9',
        'board_size': 9,
        'hypothesis': '100 sims + warm-start: does deeper search push past 94%?',
        'warm_start': WARM_START,
        'mcts': {k: getattr(config.mcts, k) for k in config.mcts.__dataclass_fields__},
        'network': {k: getattr(config.network, k) for k in config.network.__dataclass_fields__},
        'training': {k: getattr(config.training, k) for k in config.training.__dataclass_fields__},
        'arena': {k: getattr(config.arena, k) for k in config.arena.__dataclass_fields__},
    }, f, indent=2, default=str)

game = Go(size=9)
model = create_model(game, config.network, lr=config.training.lr)

# Warm-start
warm_path = os.path.abspath(WARM_START)
if os.path.exists(warm_path):
    model.load(warm_path)
    print(f"Loaded warm-start weights from {warm_path}")
else:
    print(f"WARNING: warm-start file not found: {warm_path}")
    print("Training from scratch!")

total_params = sum(p.numel() for p in model.net.parameters())
print(f"\n=== Go 9x9 — 100 sims + warm-start ===")
print(f"Network: CNN {config.network.num_filters}f {config.network.num_res_blocks} res ({total_params:,} params)")
print(f"MCTS: {config.mcts.num_simulations} sims, batch={config.mcts.nn_batch_size}")
print(f"Training: {config.training.num_iterations} iters x {config.training.games_per_iteration} games")
print(f"GPU: {__import__('torch').cuda.get_device_name(0) if __import__('torch').cuda.is_available() else 'CPU'}")

start = time.time()
history = run_pipeline(game, model, config)
elapsed = time.time() - start

with open(os.path.join(DATA_DIR, 'history.json'), 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n=== Done in {elapsed/60:.1f} min ===")

# Upload to wandb
try:
    import wandb
    if wandb.run:
        best_path = os.path.join(DATA_DIR, 'checkpoints', 'best.pt')
        if os.path.exists(best_path):
            artifact = wandb.Artifact('go9-cnn-100sims', type='model')
            artifact.add_file(best_path, name='best.pt')
            wandb.log_artifact(artifact)
            print("Uploaded best.pt to wandb artifact 'go9-cnn-100sims'")
        wandb.finish()
except Exception as e:
    print(f"W&B artifact upload failed: {e}")

# Auto-push
print("\n=== Auto-pushing results ===")
try:
    repo_root = os.path.join(EXP_DIR, '..', '..', '..')
    subprocess.run(['git', 'add', '-f', 'alphago/experiments/20260310_go9_100sims/'],
                   cwd=repo_root, check=True)
    msg = f'Go 9x9 100sims results: {history[-1].get("vs_random", "?")}% final, {elapsed/60:.0f}m\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>'
    subprocess.run(['git', 'commit', '-m', msg], cwd=repo_root, check=True)
    subprocess.run(['git', 'push'], cwd=repo_root, check=True, capture_output=True)
    print("Results pushed to git!")
except Exception as e:
    print(f"Git push failed: {e}")
    print("Manual push needed.")
