"""Game implementations."""

from .connect4 import ConnectFour
from .othello import Othello
from .tictactoe import TicTacToe

GAMES = {
    'tictactoe': TicTacToe,
    'connect4': ConnectFour,
    'othello': Othello,
    'othello6': lambda: Othello(size=6),
    'othello8': lambda: Othello(size=8),
    'othello10': lambda: Othello(size=10),
}


def get_game(name: str):
    """Get a game instance by name."""
    if name not in GAMES:
        raise ValueError(f"Unknown game '{name}'. Available: {list(GAMES.keys())}")
    return GAMES[name]()
