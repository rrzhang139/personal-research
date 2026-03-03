"""Tests for self-play, trainer, arena, and buffer strategies."""

import numpy as np

from alpha_go.games.tictactoe import TicTacToe
from alpha_go.neural_net.simple_net import SimpleNetWrapper
from alpha_go.training.arena import arena_compare, play_vs_random
from alpha_go.training.pipeline import run_pipeline
from alpha_go.training.self_play import self_play_game, generate_self_play_data
from alpha_go.training.trainer import train_on_examples
from alpha_go.utils.config import (
    AlphaZeroConfig, ArenaConfig, MCTSConfig, NetworkConfig, TrainingConfig,
)


def _make_model():
    config = NetworkConfig(hidden_size=32, num_layers=2)
    return SimpleNetWrapper(board_size=9, action_size=9, config=config, lr=0.001)


class TestSelfPlay:

    def test_single_game(self):
        game = TicTacToe()
        model = _make_model()
        config = MCTSConfig(num_simulations=5, dirichlet_epsilon=0.0)

        examples, outcome, diag = self_play_game(game, model, config)
        assert len(examples) > 0
        assert outcome in [-1, 0, 1]

        for state, pi, v in examples:
            assert state.shape == (9,)
            assert pi.shape == (9,)
            assert abs(pi.sum() - 1.0) < 1e-5
            assert v in [-1.0, 0.0, 1.0]

    def test_single_game_diagnostics(self):
        game = TicTacToe()
        model = _make_model()
        config = MCTSConfig(num_simulations=5, dirichlet_epsilon=0.0)

        examples, outcome, diag = self_play_game(game, model, config, collect_diagnostics=True)
        assert diag['game_length'] > 0
        assert 'mean_root_value' in diag
        assert 'mean_policy_entropy' in diag

    def test_generate_data(self):
        game = TicTacToe()
        model = _make_model()
        config = MCTSConfig(num_simulations=5, dirichlet_epsilon=0.0)

        examples, stats = generate_self_play_data(game, model, config, num_games=3, augment=False)
        assert len(examples) > 0
        assert stats.p1_wins + stats.p2_wins + stats.draws == 3
        assert stats.mean_game_length > 0
        assert stats.mean_policy_entropy >= 0

    def test_augmented_data(self):
        game = TicTacToe()
        model = _make_model()
        config = MCTSConfig(num_simulations=5, dirichlet_epsilon=0.0)

        examples_no_aug, _ = generate_self_play_data(game, model, config, num_games=3, augment=False)
        examples_aug, _ = generate_self_play_data(game, model, config, num_games=3, augment=True)

        # Augmented should have ~8x more examples (8 symmetries)
        assert len(examples_aug) >= len(examples_no_aug) * 4  # conservative bound


class TestTrainer:

    def test_train_on_examples(self):
        model = _make_model()
        examples = [
            (np.zeros(9, dtype=np.float32), np.ones(9, dtype=np.float32) / 9, 0.5),
            (np.ones(9, dtype=np.float32), np.ones(9, dtype=np.float32) / 9, -0.5),
        ] * 20

        losses = train_on_examples(model, examples, batch_size=8, epochs=2)
        assert losses['total_loss'] > 0


class TestArena:

    def test_vs_random(self):
        """Even an untrained model should be somewhat competitive vs random."""
        game = TicTacToe()
        model = _make_model()
        config = MCTSConfig(num_simulations=10, dirichlet_epsilon=0.0)

        win_rate = play_vs_random(game, model, config, num_games=20)
        assert 0.0 <= win_rate <= 1.0


class TestParallelSelfPlay:

    def test_parallel_self_play(self):
        """Parallel self-play should produce valid training data."""
        game = TicTacToe()
        model = _make_model()
        config = MCTSConfig(num_simulations=5, dirichlet_epsilon=0.0)

        examples, stats = generate_self_play_data(
            game, model, config, num_games=4, augment=False,
            num_workers=2, game_name='tictactoe',
        )
        assert len(examples) > 0
        assert stats.p1_wins + stats.p2_wins + stats.draws == 4

        for state, pi, v in examples:
            assert state.shape == (9,)
            assert pi.shape == (9,)
            assert abs(pi.sum() - 1.0) < 1e-5
            assert v in [-1.0, 0.0, 1.0]

    def test_parallel_arena(self):
        """Parallel arena should produce valid results."""
        game = TicTacToe()
        model1 = _make_model()
        model2 = _make_model()
        config = MCTSConfig(num_simulations=5, dirichlet_epsilon=0.0)

        win_rate, stats = arena_compare(
            game, model1, model2, config, num_games=4,
            num_workers=2, game_name='tictactoe',
        )
        assert 0.0 <= win_rate <= 1.0
        assert stats['new_wins'] + stats['old_wins'] + stats['draws'] == 4

    def test_parallel_vs_random(self):
        """Parallel vs-random should produce valid win rate."""
        game = TicTacToe()
        model = _make_model()
        config = MCTSConfig(num_simulations=5, dirichlet_epsilon=0.0)

        win_rate = play_vs_random(
            game, model, config, num_games=4,
            num_workers=2, game_name='tictactoe',
        )
        assert 0.0 <= win_rate <= 1.0


class TestWindowBuffer:

    def test_window_buffer_keeps_last_n(self):
        """Window buffer should keep only the last N iterations of data."""
        import tempfile
        import os

        game = TicTacToe()
        model = _make_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = AlphaZeroConfig(
                mcts=MCTSConfig(num_simulations=5, dirichlet_epsilon=0.0),
                network=NetworkConfig(hidden_size=32, num_layers=2),
                training=TrainingConfig(
                    num_iterations=5,
                    games_per_iteration=5,
                    epochs_per_iteration=1,
                    batch_size=8,
                    buffer_strategy="window",
                    buffer_window=3,
                    checkpoint_dir=tmpdir,
                ),
                arena=ArenaConfig(arena_games=4, update_threshold=0.55),
                game='tictactoe',
                seed=42,
            )

            history = run_pipeline(game, model, config)

            # After 5 iters with window=3, buffer should hold only ~3 iters of data
            # Each iter generates 5 games * ~9 moves * 8 augmentations ≈ 360 examples
            # 3 iters ≈ 1080, 5 iters ≈ 1800. Buffer should be < 5 iters' worth.
            final_buf = history['buffer_size'][-1]
            iter4_buf = history['buffer_size'][3]  # after 4 iters (window=3 → 3 iters)

            # After iter 4 (window full at 3), buffer should not grow further
            # (each new iter replaces the oldest)
            assert final_buf <= iter4_buf * 1.2, (
                f"Buffer should stabilize after window fills. "
                f"iter4={iter4_buf}, iter5={final_buf}"
            )
