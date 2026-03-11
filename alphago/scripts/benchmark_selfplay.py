#!/usr/bin/env python3
"""Benchmark self-play speed: 10 games x 200 sims on Go 9x9.

Usage: python -u scripts/benchmark_selfplay.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from alpha_go.games.go import Go
from alpha_go.neural_net import create_model
from alpha_go.utils.config import MCTSConfig, NetworkConfig
from alpha_go.training.self_play import generate_self_play_data

game = Go(size=9)
net_cfg = NetworkConfig(network_type="cnn", num_filters=128, num_res_blocks=4)
model = create_model(game, net_cfg, lr=0.001)
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

NUM_GAMES = 10
print(f"Benchmarking: {NUM_GAMES} games x {mcts_cfg.num_simulations} sims (batch={mcts_cfg.nn_batch_size})")

t0 = time.time()
examples, stats = generate_self_play_data(game, model, mcts_cfg, NUM_GAMES, augment=False)
elapsed = time.time() - t0

print(f"\nResults:")
print(f"  Time: {elapsed:.1f}s ({elapsed/NUM_GAMES:.1f}s/game)")
print(f"  Examples: {len(examples)}")
print(f"  Outcomes: P1={stats.p1_wins} P2={stats.p2_wins} Draw={stats.draws}")
print(f"  Avg game length: {stats.mean_game_length:.1f}")
print(f"  Avg search depth: {stats.mean_search_depth:.1f}")
print(f"  Projected 100 games: {elapsed/NUM_GAMES*100:.0f}s ({elapsed/NUM_GAMES*100/60:.1f}m)")
