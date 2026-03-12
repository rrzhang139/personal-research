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
    Encourages exploration early, exploitation late.
    Ignored if temp_decay_halflife > 0 (exponential decay used instead)."""

    temp_decay_halflife: int = 0
    """Exponential temperature decay halflife (in moves). KataGo uses 19.
    0 = disabled (use temp_threshold hard cutoff instead).
    When enabled: temp(move) = temp_end + (temp_start - temp_end) * 0.5^(move/halflife)
    where temp_start = temperature, temp_end = 0.1."""

    nn_batch_size: int = 1
    """Batch size for neural network evaluations within MCTS.
    1 = sequential (original). >1 = virtual loss batching — collects
    this many leaf positions per batch, evaluates in one forward pass.
    Default 8 is a good balance for CPU; GPU benefits from larger batches."""

    playout_cap_prob: float = 1.0
    """Probability of a 'full' search per move (KataGo playout cap randomization).
    1.0 = all moves use full num_simulations (disabled, default).
    0.125 = 12.5% of moves get full search (recorded for training),
    87.5% get cheap search (not recorded). Games finish ~4-5x faster."""

    playout_cap_cheap_fraction: float = 0.25
    """Cheap search uses num_simulations * this fraction.
    E.g. with num_simulations=200 and fraction=0.15, cheap = 30 sims."""

    fpu_reduction: float = 0.0
    """First Play Urgency reduction for non-root nodes. Unvisited children get Q = parent_Q - fpu_reduction.
    0.0 = disabled (original AlphaZero). KataGo uses 0.2."""

    root_fpu_reduction: float = -1.0
    """FPU reduction at root. -1.0 = use fpu_reduction (same as non-root).
    KataGo uses 0.1 at root (lower, since Dirichlet noise already explores).
    0.0 at root means no FPU penalty = explore all moves equally."""

    c_puct_base: float = 0.0
    """Log-based c_puct scaling: effective_c = c_puct * log((parent_N + c_puct_base + 1) / c_puct_base).
    0.0 = disabled (constant c_puct). AlphaZero uses 19652. Reduces exploration as visit count grows."""

    progressive_sims: bool = False
    """Scale num_simulations linearly from min_sims to num_simulations across training.
    First iteration uses min_sims; last uses num_simulations. More games early, better search later."""

    min_sims: int = 50
    """Starting simulation count when progressive_sims=True."""


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

    dropout: float = 0.0
    """Dropout rate for CNN heads. 0.0 = no dropout, 0.3 = reference."""

    global_pool_value: bool = False
    """Add global avg+max pooling features to value head (KataGo-style).
    Improves position evaluation but incompatible with older checkpoints."""

    use_batch_norm: bool = True
    """Use batch normalization in residual blocks. KataGo found removing BN
    gives 1.6x training speedup with equal or better performance."""

    use_se: bool = False
    """Use Squeeze-and-Excitation blocks in residual tower (Leela Zero, KataGo).
    Adds channel-wise attention with ~2% more parameters. Improves feature quality."""


@dataclass
class TrainingConfig:
    """Training loop parameters."""

    lr: float = 0.001
    """Learning rate for Adam optimizer."""

    weight_decay: float = 0.0
    """Weight decay (L2 regularization) for optimizer. KataGo uses 1e-4."""

    lr_schedule: str = "constant"
    """Learning rate schedule: 'constant' or 'cosine'."""

    lr_min: float = 1e-5
    """Minimum learning rate for cosine schedule."""

    batch_size: int = 64
    """Minibatch size for training."""

    epochs_per_iteration: int = 10
    """Training epochs per iteration (passes over replay buffer)."""

    max_buffer_size: int = 50_000
    """Maximum replay buffer size. Older games get dropped (FIFO strategy)."""

    buffer_strategy: str = "fifo"
    """Buffer strategy: 'fifo' (fixed-size deque) or 'window' (last N iterations)."""

    buffer_window: int = 20
    """For 'window' strategy: keep examples from the last N iterations."""

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

    eval_games: int = 50
    """Number of games to play vs random player each iteration. 0 = skip."""


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

    num_workers: int = 1
    """Parallel workers for self-play/arena. 0=auto (cpu_count-1), 1=sequential."""

    use_cpp_mcts: bool = False
    """Use C++ MCTS engine for self-play. Requires building the C++ extension.
    Gives ~6x game logic speedup + true multi-threading (bypasses GIL)."""

    use_wandb: bool = False
    """Whether to log to Weights & Biases."""

    wandb_project: str = "alphazero"
    """W&B project name."""
