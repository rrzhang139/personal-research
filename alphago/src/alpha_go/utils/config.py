"""All tunable parameters as typed dataclasses.

This is your control panel. Every experiment is defined by a config.
Change parameters here (or via CLI overrides) to run different experiments.
"""

from dataclasses import dataclass, field

@dataclass
class MCTSConfig:
    """Monte Carlo Tree Search parameters."""

    num_simulations: int = 25
    """Number of MCTS simulations per move. More = stronger but slower."""

    c_puct: float = 1.0
    """Exploration constant in PUCT formula. Higher = more exploration."""

    dirichlet_alpha: float = 0.3
    """Dirichlet noise parameter. Smaller = more concentrated (less noise).
    AlphaZero uses 0.3 for chess, 0.03 for Go. Scale inversely with action space."""

    dirichlet_epsilon: float = 0.25
    """Weight of Dirichlet noise vs prior. 0 = no noise, 1 = all noise.
    AlphaZero uses 0.25."""

    temperature: float = 1.0
    """Temperature for move selection from visit counts.
    1.0 = proportional to visits, ->0 = greedy (pick most visited)."""

    temp_threshold: int = 15
    """After this many moves in a game, switch to greedy (temp -> 0).
    Encourages exploration early, exploitation late."""


@dataclass
class NetworkConfig:
    """Neural network architecture parameters."""

    network_type: str = "mlp"
    """Network architecture: 'mlp' or 'cnn'."""

    hidden_size: int = 128
    """Width of hidden layers (MLP)."""

    num_layers: int = 4
    """Number of hidden layers (MLP)."""

    num_filters: int = 64
    """Number of convolutional filters (CNN)."""

    num_res_blocks: int = 4
    """Number of residual blocks (CNN)."""


@dataclass
class TrainingConfig:
    """Training loop parameters."""

    lr: float = 0.001
    """Learning rate for Adam optimizer."""

    batch_size: int = 64
    """Minibatch size for training."""

    epochs_per_iteration: int = 10
    """Training epochs per iteration (passes over replay buffer)."""

    max_buffer_size: int = 50_000
    """Maximum replay buffer size. Older games get dropped."""

    num_iterations: int = 25
    """Number of self-play → train → arena cycles."""

    games_per_iteration: int = 100
    """Self-play games generated per iteration."""

    checkpoint_dir: str = "checkpoints"
    """Where to save model checkpoints."""


@dataclass
class ArenaConfig:
    """Model evaluation (arena) parameters."""

    arena_games: int = 40
    """Number of games to play in the arena (new model vs old)."""

    update_threshold: float = 0.55
    """Win rate threshold to accept the new model.
    0.55 = new model must win >55% of arena games."""


@dataclass
class AlphaZeroConfig:
    """Top-level config composing all sub-configs."""

    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)

    game: str = "tictactoe"
    """Which game to use."""

    seed: int = 42
    """Random seed for reproducibility."""

    use_wandb: bool = False
    """Whether to log to Weights & Biases."""

    wandb_project: str = "alphazero"
    """W&B project name."""
