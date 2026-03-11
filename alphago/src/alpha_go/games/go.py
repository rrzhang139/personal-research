"""Go implementation — configurable board size (default 9x9).

Rules follow Tromp-Taylor (positional superko simplified to simple ko):
- Stones are placed on intersections. Groups with 0 liberties are captured.
- Suicide (placing a stone that leaves your own group with 0 liberties
  without capturing anything) is illegal.
- Ko: if a move captures exactly 1 stone and the capturing group has
  exactly 1 liberty (the just-captured point), that point is ko-banned
  for the next move only.
- Game ends when both players pass consecutively.
- Scoring: Tromp-Taylor area scoring (stones + surrounded empty territory)
  with 7.5 komi for White.

State representation (AlphaGo Zero style, 17 planes):
  Planes 0-7:   Player 1 (Black) stone history (0 = current, 7 = oldest)
  Planes 8-15:  Player -1 (White) stone history
  Plane 16:     Color plane (1.0 if player 1 to move, 0.0 if player -1)
  state[17*N*N]:   consecutive_pass_count (0, 1, or 2)
  state[17*N*N+1]: ko_point (-1.0 if none, else linear index 0..N*N-1)

  NN input (canonical) = first 17*N*N floats only.
  get_board_size()  = 17 * N * N
  get_board_shape() = (17, N, N)
  get_action_size() = N * N + 1  (board intersections + pass)
"""

from __future__ import annotations

import numpy as np

from .base_game import Game

# Plane indices
NUM_HISTORY = 8
P1_PLANES = slice(0, NUM_HISTORY)       # planes 0-7: player 1 history
P2_PLANES = slice(NUM_HISTORY, 2 * NUM_HISTORY)  # planes 8-15: player -1 history
COLOR_PLANE = 2 * NUM_HISTORY           # plane 16


class Go(Game):

    def __init__(self, size: int = 9):
        self.size = size
        self.n2 = size * size
        self.num_planes = 2 * NUM_HISTORY + 1  # 17
        self.nn_input_size = self.num_planes * self.n2  # 17 * N * N
        self.pass_action = self.n2  # last action index = pass
        self._neighbors = self._build_neighbors()

    def _build_neighbors(self) -> list[tuple[int, ...]]:
        """Precompute adjacency for all intersections (4-connected).

        Returns a list (indexed by position) of tuples of neighbor indices.
        Tuples are used instead of lists for faster iteration in hot loops.
        """
        n = self.size
        neighbors = []
        for idx in range(self.n2):
            r, c = divmod(idx, n)
            nbrs = []
            if r > 0:
                nbrs.append(idx - n)
            if r < n - 1:
                nbrs.append(idx + n)
            if c > 0:
                nbrs.append(idx - 1)
            if c < n - 1:
                nbrs.append(idx + 1)
            neighbors.append(tuple(nbrs))
        # Pre-allocated work arrays for flood fill (reused across calls)
        self._visited = bytearray(self.n2)
        self._stack = [0] * self.n2
        return neighbors

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def _get_planes(self, state: np.ndarray) -> np.ndarray:
        """Return (17, N*N) view of the plane data."""
        return state[:self.nn_input_size].reshape(self.num_planes, self.n2)

    def _get_current_board(self, state: np.ndarray) -> np.ndarray:
        """Reconstruct flat board (N*N,) with {0, 1, -1} from planes 0 and 8."""
        planes = self._get_planes(state)
        # Must copy: callers (_is_suicide, get_next_state) modify the board in-place
        return (planes[0] - planes[NUM_HISTORY]).copy()

    def _get_pass_count(self, state: np.ndarray) -> int:
        return int(state[self.nn_input_size])

    def _set_pass_count(self, state: np.ndarray, count: int):
        state[self.nn_input_size] = float(count)

    def _get_ko_point(self, state: np.ndarray) -> int:
        """Return ko point index, or -1 if none."""
        return int(state[self.nn_input_size + 1])

    def _set_ko_point(self, state: np.ndarray, ko: int):
        state[self.nn_input_size + 1] = float(ko)

    def _get_color_to_move(self, state: np.ndarray) -> int:
        """Return 1 if player 1 to move, -1 if player -1."""
        planes = self._get_planes(state)
        return 1 if planes[COLOR_PLANE][0] > 0.5 else -1

    # ------------------------------------------------------------------
    # Group / liberty helpers
    # ------------------------------------------------------------------

    def _find_group(self, board: np.ndarray, idx: int) -> tuple[list[int], int]:
        """Flood-fill from idx. Return (group stone list, liberty count).

        Uses pre-allocated bytearray + stack for speed (no set allocations).
        """
        color = board[idx]
        neighbors = self._neighbors
        visited = self._visited
        stack = self._stack

        group = []
        lib_count = 0
        to_clean = []

        visited[idx] = 1
        to_clean.append(idx)
        stack[0] = idx
        sp = 1

        while sp > 0:
            sp -= 1
            pos = stack[sp]
            group.append(pos)
            for nbr in neighbors[pos]:
                if visited[nbr]:
                    continue
                bval = board[nbr]
                if bval == color:
                    visited[nbr] = 1
                    to_clean.append(nbr)
                    stack[sp] = nbr
                    sp += 1
                elif bval == 0:
                    visited[nbr] = 2  # liberty marker
                    to_clean.append(nbr)
                    lib_count += 1

        for pos in to_clean:
            visited[pos] = 0
        return group, lib_count

    def _group_has_liberty(self, board: np.ndarray, idx: int) -> bool:
        """Fast check: does the group at idx have at least one liberty?

        Returns as soon as any liberty is found — much faster than _find_group
        for groups that DO have liberties (the common case in suicide checks).
        """
        color = board[idx]
        neighbors = self._neighbors
        visited = self._visited
        stack = self._stack

        visited[idx] = 1
        to_clean = [idx]
        stack[0] = idx
        sp = 1
        found = False

        while sp > 0:
            sp -= 1
            pos = stack[sp]
            for nbr in neighbors[pos]:
                if visited[nbr]:
                    continue
                bval = board[nbr]
                if bval == 0:
                    found = True
                    sp = 0  # break outer
                    break
                if bval == color:
                    visited[nbr] = 1
                    to_clean.append(nbr)
                    stack[sp] = nbr
                    sp += 1

        for pos in to_clean:
            visited[pos] = 0
        return found

    def _capture_opponent(self, board: np.ndarray, idx: int, player: int) -> list[int]:
        """After placing player's stone at idx, capture opponent groups with 0 liberties.

        Returns list of captured stone indices. Uses _visited bytearray (value 3)
        to track already-checked groups without a separate set.
        """
        opponent = -player
        captured = []
        neighbors = self._neighbors
        visited = self._visited
        all_marked = []

        for nbr in neighbors[idx]:
            if board[nbr] == opponent and not visited[nbr]:
                group, lib_count = self._find_group(board, nbr)
                # Mark group stones so we skip them if another neighbor
                # of idx belongs to this same group
                for pos in group:
                    visited[pos] = 3
                    all_marked.append(pos)
                if lib_count == 0:
                    captured.extend(group)
                    for pos in group:
                        board[pos] = 0.0

        for pos in all_marked:
            visited[pos] = 0
        return captured

    def _is_suicide(self, board: np.ndarray, idx: int, player: int) -> bool:
        """Check if placing player at idx would be suicide (illegal).

        A move is suicide if, after placing and capturing opponents,
        the player's own group has 0 liberties.

        Uses _group_has_liberty for fast early returns.
        """
        # Fast path: if any neighbor is empty, the new stone has a liberty
        for nbr in self._neighbors[idx]:
            if board[nbr] == 0:
                return False

        # Surrounded by stones — need full check
        board[idx] = player
        opponent = -player
        for nbr in self._neighbors[idx]:
            if board[nbr] == opponent:
                if not self._group_has_liberty(board, nbr):
                    board[idx] = 0.0
                    return False  # captures → not suicide

        # Check own group liberties
        has_lib = self._group_has_liberty(board, idx)
        board[idx] = 0.0
        return not has_lib

    # ------------------------------------------------------------------
    # Game interface
    # ------------------------------------------------------------------

    def get_initial_state(self) -> np.ndarray:
        """Empty board, player 1 (Black) to move, no ko, 0 passes."""
        state = np.zeros(self.nn_input_size + 2, dtype=np.float32)
        # Color plane: 1.0 = player 1 to move
        planes = self._get_planes(state)
        planes[COLOR_PLANE] = 1.0
        self._set_pass_count(state, 0)
        self._set_ko_point(state, -1)
        return state

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        new_state = state.copy()
        planes = self._get_planes(new_state)

        if action == self.pass_action:
            # Pass: increment pass count, clear ko, flip color
            self._set_pass_count(new_state, self._get_pass_count(new_state) + 1)
            self._set_ko_point(new_state, -1)
            # Flip color plane
            planes[COLOR_PLANE] = 1.0 - planes[COLOR_PLANE]
            return new_state

        # Non-pass move: reset pass count
        self._set_pass_count(new_state, 0)

        # Get current board, place stone, capture
        board = self._get_current_board(new_state)
        board[action] = player
        captured = self._capture_opponent(board, action, player)

        # Ko detection: exactly 1 captured AND own group has exactly 1 stone
        # (i.e., single stone capture that could be immediately recaptured)
        ko = -1
        if len(captured) == 1:
            own_group, own_lib_count = self._find_group(board, action)
            if len(own_group) == 1 and own_lib_count == 1:
                ko = captured[0]
        self._set_ko_point(new_state, ko)

        # Shift history planes: bulk copy (1 op instead of 7 per player)
        planes[1:NUM_HISTORY] = planes[0:NUM_HISTORY - 1]
        planes[0] = (board == 1.0).astype(np.float32)

        planes[NUM_HISTORY + 1:2 * NUM_HISTORY] = planes[NUM_HISTORY:2 * NUM_HISTORY - 1]
        planes[NUM_HISTORY] = (board == -1.0).astype(np.float32)

        # Flip color plane
        planes[COLOR_PLANE] = 1.0 - planes[COLOR_PLANE]

        return new_state

    def get_valid_moves(self, state: np.ndarray, player: int = 1) -> np.ndarray:
        valid = np.zeros(self.n2 + 1, dtype=np.float32)
        board = self._get_current_board(state)
        ko = self._get_ko_point(state)

        # Find empty intersections (candidates)
        empty = np.where(board == 0)[0]

        # Vectorized fast-path: check if any neighbor is empty
        # For each empty cell, if any neighbor is also empty -> valid (has liberty)
        for idx_val in empty:
            idx = int(idx_val)
            if idx == ko:
                continue
            # Fast path: any empty neighbor = has a liberty, not suicide
            has_liberty = False
            for nbr in self._neighbors[idx]:
                if board[nbr] == 0:
                    has_liberty = True
                    break
            if has_liberty:
                valid[idx] = 1.0
            elif not self._is_suicide(board, idx, player):
                valid[idx] = 1.0

        # Pass is always valid
        valid[self.pass_action] = 1.0
        return valid

    def check_terminal(self, state: np.ndarray, action: int, player: int = 1) -> tuple[bool, float]:
        """Game ends when consecutive_passes >= 2. Score using Tromp-Taylor."""
        if self._get_pass_count(state) < 2:
            return False, 0.0

        board = self._get_current_board(state)
        score = self._tromp_taylor_score(board)

        # score > 0 means player 1 (Black) wins, < 0 means player -1 (White) wins
        if score > 0:
            value = 1.0 if player == 1 else -1.0
        elif score < 0:
            value = 1.0 if player == -1 else -1.0
        else:
            value = 0.0

        return True, value

    def _tromp_taylor_score(self, board: np.ndarray) -> float:
        """Tromp-Taylor area scoring: stones + territory - komi.

        Territory = empty points reachable only by one color.
        Returns score from player 1's perspective (positive = Black wins).
        """
        komi = 7.5

        # Count stones
        p1_score = float(np.sum(board == 1.0))
        p2_score = float(np.sum(board == -1.0))

        # Flood-fill empty regions to determine territory
        visited = np.zeros(self.n2, dtype=bool)
        for idx in range(self.n2):
            if board[idx] != 0 or visited[idx]:
                continue
            # Flood-fill this empty region
            region = set()
            borders = set()  # colors bordering this region
            stack = [idx]
            while stack:
                pos = stack.pop()
                if visited[pos]:
                    continue
                if board[pos] != 0:
                    borders.add(int(board[pos]))
                    continue
                visited[pos] = True
                region.add(pos)
                for nbr in self._neighbors[pos]:
                    if not visited[nbr]:
                        stack.append(nbr)

            # If only one color borders this region, it's territory
            if borders == {1}:
                p1_score += len(region)
            elif borders == {-1}:
                p2_score += len(region)
            # If both colors border, or no border (shouldn't happen on non-empty board),
            # it's neutral — not counted for either player.

        return p1_score - p2_score - komi

    def get_board_size(self) -> int:
        return self.nn_input_size  # 17 * N * N

    def get_board_shape(self) -> tuple[int, int, int]:
        return (self.num_planes, self.size, self.size)

    def get_action_size(self) -> int:
        return self.n2 + 1  # board intersections + pass

    def get_canonical_state(self, state: np.ndarray, player: int) -> np.ndarray:
        """Return NN input (17*N*N floats) from given player's perspective.

        If player == 1: return planes as-is (just the NN portion).
        If player == -1: swap planes 0-7 <-> 8-15, flip color plane.
        """
        if player == 1:
            return state[:self.nn_input_size].copy()
        # player == -1: need to swap planes
        planes = state[:self.nn_input_size].reshape(self.num_planes, self.n2).copy()
        # Swap player 1 and player -1 history planes
        p1_copy = planes[P1_PLANES].copy()
        planes[P1_PLANES] = planes[P2_PLANES]
        planes[P2_PLANES] = p1_copy
        # Flip color plane
        planes[COLOR_PLANE] = 1.0 - planes[COLOR_PLANE]
        return planes.flatten()

    def get_symmetries(self, state: np.ndarray, pi: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """8-fold symmetries (4 rotations x 2 reflections) applied to all 17 planes."""
        n = self.size
        # state here is the canonical NN input (17*N*N)
        planes = state.reshape(self.num_planes, n, n)
        board_pi = pi[:self.n2].reshape(n, n)
        pass_prob = pi[self.pass_action]

        symmetries = []
        for rotation in range(4):
            rp_all = np.rot90(planes, rotation, axes=(1, 2))
            rp_pi = np.rot90(board_pi, rotation)
            sym_pi = np.append(rp_pi.flatten(), pass_prob)
            symmetries.append((rp_all.flatten().copy(), sym_pi))

            fp_all = np.flip(rp_all, axis=2)  # flip left-right
            fp_pi = np.fliplr(rp_pi)
            sym_pi_flip = np.append(fp_pi.flatten(), pass_prob)
            symmetries.append((fp_all.flatten().copy(), sym_pi_flip))

        return symmetries

    def display(self, state: np.ndarray) -> str:
        n = self.size
        board = self._get_current_board(state)
        symbols = {0: '.', 1: 'X', -1: 'O'}

        col_labels = '   ' + ' '.join(chr(ord('A') + c + (1 if c >= 8 else 0)) for c in range(n))
        rows = [col_labels]
        for r in range(n):
            row_num = n - r  # Go convention: row 1 at bottom
            row_str = f'{row_num:2d} ' + ' '.join(
                symbols[int(board[r * n + c])] for c in range(n)
            )
            rows.append(row_str)

        p1 = int(np.sum(board == 1))
        p2 = int(np.sum(board == -1))
        ko = self._get_ko_point(state)
        passes = self._get_pass_count(state)
        rows.append(f'Black(X):{p1}  White(O):{p2}  Ko:{ko}  Passes:{passes}')
        return '\n'.join(rows)
