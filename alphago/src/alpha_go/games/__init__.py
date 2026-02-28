"""Game implementations."""

from .connect4 import ConnectFour
from .othello import Othello
from .tictactoe import TicTacToe

GAMES = {
    'tictactoe': TicTacToe,
    'connect4': ConnectFour,
    'othello': Othello,
}


def get_game(name: str):
    """Get a game instance by name."""
    if name not in GAMES:
        raise ValueError(f"Unknown game '{name}'. Available: {list(GAMES.keys())}")
    return GAMES[name]()
