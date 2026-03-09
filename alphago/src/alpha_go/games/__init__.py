"""Game implementations."""

from .connect4 import ConnectFour
from .go import Go
from .othello import Othello
from .tictactoe import TicTacToe

GAMES = {
    'tictactoe': TicTacToe,
    'connect4': ConnectFour,
    'othello': Othello,
    'othello6': lambda: Othello(size=6),
    'othello8': lambda: Othello(size=8),
    'othello10': lambda: Othello(size=10),
    'go': Go,
    'go9': lambda: Go(size=9),
    'go13': lambda: Go(size=13),
    'go19': lambda: Go(size=19),
}


def get_game(name: str):
    """Get a game instance by name."""
    if name not in GAMES:
        raise ValueError(f"Unknown game '{name}'. Available: {list(GAMES.keys())}")
    return GAMES[name]()
