"""Neural network implementations."""

from ..utils.config import NetworkConfig
from .conv_net import ConvNetWrapper
from .simple_net import SimpleNetWrapper


def create_model(game, config: NetworkConfig, lr: float = 0.001):
    """Factory: create the right model wrapper based on config.network_type.

    Args:
        game: Game instance (must implement get_board_size, get_action_size, get_board_shape).
        config: NetworkConfig with network_type ('mlp' or 'cnn').
        lr: Learning rate for the optimizer.

    Returns:
        SimpleNetWrapper or ConvNetWrapper.
    """
    board_size = game.get_board_size()
    action_size = game.get_action_size()

    if config.network_type == "mlp":
        return SimpleNetWrapper(board_size, action_size, config, lr=lr)
    elif config.network_type == "cnn":
        board_shape = game.get_board_shape()
        return ConvNetWrapper(board_size, action_size, config, lr=lr, board_shape=board_shape)
    else:
        raise ValueError(f"Unknown network_type '{config.network_type}'. Use 'mlp' or 'cnn'.")
