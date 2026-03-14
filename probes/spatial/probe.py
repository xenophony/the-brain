"""
Spatial reasoning probe — Battleship next-move prediction with generated boards.

Generates random valid Battleship boards, simulates mid-game state,
presents as ASCII grid, and scores against a probability density oracle.

Output: single grid coordinate like "B4".
Scoring: cell_probability_density / max_probability_density.
Maps to: parietal lobe / visual cortex spatial processing.
"""

import re
import random
from itertools import product
from probes.registry import BaseProbe, register_probe

FLEET = [
    ("Carrier", 5),
    ("Battleship", 4),
    ("Cruiser", 3),
    ("Destroyer", 3),
    ("Submarine", 2),
]

COLS = "ABCDEFGHIJ"
ROWS = range(1, 11)
BOARD_SIZE = 10
N_BOARDS = 20
SEED = 42


def _make_empty_board():
    """Return 10x10 board initialized to None (unknown)."""
    return [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def _place_ships(rng):
    """
    Randomly place standard fleet on a 10x10 board.
    Returns a set of (row, col) tuples occupied by ships,
    and a list of ship placements [(set_of_cells, length), ...].
    """
    occupied = set()
    placements = []

    for ship_name, length in FLEET:
        placed = False
        attempts = 0
        while not placed and attempts < 1000:
            attempts += 1
            horizontal = rng.choice([True, False])
            if horizontal:
                r = rng.randint(0, 9)
                c = rng.randint(0, 10 - length)
                cells = {(r, c + k) for k in range(length)}
            else:
                r = rng.randint(0, 10 - length)
                c = rng.randint(0, 9)
                cells = {(r + k, c) for k in range(length)}

            if not cells & occupied:
                occupied |= cells
                placements.append((cells, length))
                placed = True

        if not placed:
            raise RuntimeError(f"Could not place {ship_name}")

    return occupied, placements


def _simulate_midgame(rng, ship_cells):
    """
    Simulate a mid-game board state.
    Ship cells: 30% chance HIT, rest unknown.
    Water cells: 20% chance MISS, rest unknown.

    Returns board[row][col] with values: 'H', 'M', or None (unknown).
    """
    board = _make_empty_board()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r, c) in ship_cells:
                if rng.random() < 0.30:
                    board[r][c] = 'H'
            else:
                if rng.random() < 0.20:
                    board[r][c] = 'M'
    return board


def _simulate_easy_board(rng, ship_cells):
    """
    Simulate an easy board state: single isolated hit, no adjacent misses.
    This makes the obvious next move very clear.
    """
    board = _make_empty_board()
    # Pick one random ship cell to be a hit
    ship_list = list(ship_cells)
    rng.shuffle(ship_list)
    hit_cell = ship_list[0]
    board[hit_cell[0]][hit_cell[1]] = 'H'

    # Add some misses far from the hit
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r, c) in ship_cells:
                continue
            # Skip cells adjacent to the hit
            if abs(r - hit_cell[0]) + abs(c - hit_cell[1]) <= 2:
                continue
            if rng.random() < 0.15:
                board[r][c] = 'M'
    return board


def _has_hits(board):
    """Check if board has at least one hit."""
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == 'H':
                return True
    return False


def _count_hits(board):
    """Count number of hits on the board."""
    count = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == 'H':
                count += 1
    return count


def generate_boards(seed=SEED, n=N_BOARDS):
    """
    Generate n Battleship board scenarios.
    Returns list of (board, ship_cells, placements) tuples.
    Only boards with at least one hit are included.
    """
    rng = random.Random(seed)
    boards = []
    attempts = 0
    while len(boards) < n and attempts < n * 10:
        attempts += 1
        try:
            ship_cells, placements = _place_ships(rng)
            board = _simulate_midgame(rng, ship_cells)
            if _has_hits(board):
                boards.append((board, ship_cells, placements))
        except RuntimeError:
            continue
    return boards


def generate_easy_boards(seed=SEED, n=10):
    """Generate easy boards with single isolated hits."""
    rng = random.Random(seed + 1000)  # Different seed space from standard boards
    boards = []
    attempts = 0
    while len(boards) < n and attempts < n * 10:
        attempts += 1
        try:
            ship_cells, placements = _place_ships(rng)
            board = _simulate_easy_board(rng, ship_cells)
            if _has_hits(board):
                boards.append((board, ship_cells, placements))
        except RuntimeError:
            continue
    return boards


def generate_hard_boards(seed=SEED, n=10):
    """Generate hard boards with 3+ hits and complex multi-ship states."""
    rng = random.Random(seed + 2000)  # Different seed space
    boards = []
    attempts = 0
    while len(boards) < n and attempts < n * 20:
        attempts += 1
        try:
            ship_cells, placements = _place_ships(rng)
            board = _simulate_midgame(rng, ship_cells)
            if _count_hits(board) >= 3:
                boards.append((board, ship_cells, placements))
        except RuntimeError:
            continue
    return boards


def board_to_ascii(board):
    """
    Render board as ASCII in the specified format.
    . = unknown, H = hit, M = miss
    """
    lines = [". = unknown, H = hit, M = miss", "", "+--------------------+"]
    for r in range(BOARD_SIZE):
        row_chars = []
        for c in range(BOARD_SIZE):
            cell = board[r][c]
            if cell == 'H':
                row_chars.append('H')
            elif cell == 'M':
                row_chars.append('M')
            else:
                row_chars.append('.')
        lines.append("|" + " ".join(row_chars) + "|")
    lines.append("+--------------------+")
    return "\n".join(lines)


def compute_probability_density(board):
    """
    Compute probability density for each unknown cell using ONLY visible
    board state. Does NOT use ground truth ship positions.

    For each ship size in the standard fleet, slides all valid placements
    (horizontal and vertical) across the board. A placement is valid if
    all its cells are either HIT or unknown (not MISS). Each valid
    placement increments the density of its unknown cells.

    This is the standard Battleship probability density algorithm.
    Placements that include existing HITs are weighted 2x (TARGET mode)
    vs placements over pure unknowns (HUNT mode).

    Returns dict of (row, col) -> density for unknown cells.
    """
    SHIP_LENGTHS = [5, 4, 3, 3, 2]  # standard fleet

    hits = set()
    misses = set()
    unknowns = set()

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == 'H':
                hits.add((r, c))
            elif board[r][c] == 'M':
                misses.add((r, c))
            else:
                unknowns.add((r, c))

    if not unknowns:
        return {}

    density = {(r, c): 0.0 for r, c in unknowns}

    for length in SHIP_LENGTHS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for horizontal in [True, False]:
                    if horizontal:
                        if c + length > BOARD_SIZE:
                            continue
                        cells = [(r, c + k) for k in range(length)]
                    else:
                        if r + length > BOARD_SIZE:
                            continue
                        cells = [(r + k, c) for k in range(length)]

                    # Validity: every cell must be HIT or unknown
                    valid = True
                    has_hit = False
                    unknown_in_placement = []
                    for cr, cc in cells:
                        if (cr, cc) in misses:
                            valid = False
                            break
                        if (cr, cc) in hits:
                            has_hit = True
                        elif (cr, cc) in unknowns:
                            unknown_in_placement.append((cr, cc))
                        else:
                            # Should not happen, but treat as blocking
                            valid = False
                            break

                    if valid and unknown_in_placement:
                        weight = 2.0 if has_hit else 1.0
                        for ur, uc in unknown_in_placement:
                            density[(ur, uc)] += weight

    return density


def score_response(response: str, board) -> float:
    """
    Score a coordinate response against the probability density oracle.
    Uses ONLY visible board state — no ground truth ship positions.
    Score = cell_density / max_density.
    """
    response = response.strip().upper()
    match = re.search(r'([A-J])\s*(\d{1,2})', response)
    if not match:
        return 0.0

    col_letter = match.group(1)
    row_num = int(match.group(2))

    if row_num < 1 or row_num > 10:
        return 0.0

    col = COLS.index(col_letter)
    row = row_num - 1

    # Check it's an unknown cell
    if board[row][col] is not None:
        return 0.0

    density = compute_probability_density(board)
    if not density:
        return 0.0

    max_density = max(density.values())
    if max_density == 0:
        return 0.0

    cell_density = density.get((row, col), 0.0)
    return cell_density / max_density


PROMPT_TEMPLATE = """{board_ascii}

What is the single best next shot? Answer with only a grid coordinate like "B4". No explanation."""


@register_probe
class SpatialProbe(BaseProbe):
    name = "spatial"
    description = "Battleship next-move spatial reasoning — parietal/visual circuits"

    def __init__(self):
        self._easy_boards = None
        self._hard_boards = None

    def _ensure_boards(self):
        if self._easy_boards is None:
            self._easy_boards = generate_easy_boards(seed=SEED, n=10)
        if self._hard_boards is None:
            self._hard_boards = generate_hard_boards(seed=SEED, n=10)

    def run(self, model) -> dict:
        self._ensure_boards()

        easy_scores = []
        for board, _ship_cells, _placements in self._easy_boards:
            ascii_board = board_to_ascii(board)
            prompt = PROMPT_TEMPLATE.format(board_ascii=ascii_board)
            response = model.generate_short(prompt, max_new_tokens=5, temperature=0.0)
            score = score_response(response, board)
            easy_scores.append(score)

        hard_scores = []
        for board, _ship_cells, _placements in self._hard_boards:
            ascii_board = board_to_ascii(board)
            prompt = PROMPT_TEMPLATE.format(board_ascii=ascii_board)
            response = model.generate_short(prompt, max_new_tokens=5, temperature=0.0)
            score = score_response(response, board)
            hard_scores.append(score)

        return self._make_result(easy_scores, hard_scores)
