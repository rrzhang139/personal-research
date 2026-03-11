#!/usr/bin/env python3
"""Profile self-play to find CPU hotspots.

Usage: python -u scripts/profile_selfplay.py
"""
import cProfile
import os
import pstats
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from alpha_go.games.go import Go
from alpha_go.neural_net import create_model
from alpha_go.utils.config import MCTSConfig, NetworkConfig
from alpha_go.training.self_play import generate_self_play_data

game = Go(size=9)
net_cfg = NetworkConfig(network_type="cnn", num_filters=128, num_res_blocks=4)
model = create_model(game, net_cfg, lr=0.001)

# Load warm-start if available
warm = 'experiments/20260310_go9_playout_cap/data/checkpoints/best.pt'
if os.path.exists(warm):
    model.load(warm)
    print(f"Warm-started from {warm}")

print(f"Device: {model.net.device}")

mcts_cfg = MCTSConfig(
    num_simulations=200,
    c_puct=1.0,
    dirichlet_alpha=0.12,
    dirichlet_epsilon=0.25,
    temp_threshold=30,
    temp_decay_halflife=19,
    nn_batch_size=64,
    playout_cap_prob=0.125,
    playout_cap_cheap_fraction=0.15,
    fpu_reduction=0.2,
    root_fpu_reduction=0.1,
)

NUM_GAMES = 5
print(f"Profiling: {NUM_GAMES} games x {mcts_cfg.num_simulations} sims")

profiler = cProfile.Profile()
profiler.enable()
examples, stats = generate_self_play_data(game, model, mcts_cfg, NUM_GAMES, augment=False)
profiler.disable()

print(f"\nGames: P1={stats.p1_wins} P2={stats.p2_wins} Draw={stats.draws}")
print(f"Avg game length: {stats.mean_game_length:.1f}")
print(f"\n{'='*80}")
print("TOP 40 BY CUMULATIVE TIME:")
print('='*80)
ps = pstats.Stats(profiler)
ps.sort_stats('cumulative')
ps.print_stats(40)

print(f"\n{'='*80}")
print("TOP 40 BY TOTAL TIME (self, excluding subcalls):")
print('='*80)
ps.sort_stats('tottime')
ps.print_stats(40)
