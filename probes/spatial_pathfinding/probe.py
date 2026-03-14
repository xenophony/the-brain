"""
Spatial pathfinding probe — shortest path in ASCII grids.

Output: single integer (path length) or -1 if unsolvable.
Maps to: parietal lobe / spatial navigation circuits.
"""

from collections import deque
from probes.registry import BaseProbe, register_probe


def bfs_shortest_path(grid_lines):
    """BFS to find shortest path from S to E. Returns path length or -1."""
    rows = len(grid_lines)
    cols = len(grid_lines[0]) if rows else 0
    start = end = None
    for r in range(rows):
        for c in range(cols):
            if grid_lines[r][c] == 'S':
                start = (r, c)
            elif grid_lines[r][c] == 'E':
                end = (r, c)
    if not start or not end:
        return -1
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == end:
            return dist
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid_lines[nr][nc] != '#':
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return -1


def _parse_grid(grid_str):
    """Parse a grid string into a 2D list of characters."""
    lines = [line for line in grid_str.strip().split('\n') if line.strip()]
    return [line.split() for line in lines]


def _grid_to_str(grid_lines):
    """Convert a 2D grid back to a display string."""
    return '\n'.join(' '.join(row) for row in grid_lines)


# ------------------------------------------------------------------ #
#  EASY grids (5x5)                                                    #
# ------------------------------------------------------------------ #

_EASY_GRIDS = [
    # 1. Straight path, no walls
    """\
S . . . .
. . . . .
. . . . .
. . . . .
. . . . E""",

    # 2. One wall to go around
    """\
S . . . .
. # . . .
. . . . .
. . . . E
. . . . .""",

    # 3. L-shaped path
    """\
S # . . .
. # . . .
. . . . .
. . . . E
. . . . .""",

    # 4. Small maze
    """\
S . # . .
. . # . .
. . . . E
. . # . .
. . # . .""",

    # 5. Two walls
    """\
S . . . .
. # . . .
. . E # .
. . . . .
. . . . .""",

    # 6. Corner to corner
    """\
S . . . .
. . . . .
. . # . .
. . . . E
. . . . .""",

    # 7. UNSOLVABLE: completely walled off
    """\
S . # . .
. . # . .
# # # # #
. . . . .
. . . . E""",

    # 8. Narrow corridor
    """\
S . . . .
# # # . .
. . . . .
. . # # #
. . . . E""",
]

# ------------------------------------------------------------------ #
#  HARD grids (8x8)                                                    #
# ------------------------------------------------------------------ #

_HARD_GRIDS = [
    # 1. Complex maze
    """\
S . # . . . . .
. . # . # . # .
. . . . # . . .
# # . # # . # .
. . . . . . # .
. # # # . # . .
. . . . . # . .
. . . . . . . E""",

    # 2. Multiple possible paths
    """\
S . . . . . . .
. # . # . # . .
. # . # . # . .
. . . . . . . .
. . . . . . . .
. # . # . # . .
. # . # . # . .
. . . . . . . E""",

    # 3. Long winding path
    """\
S . . . . . . .
# # # # # # . .
. . . . . . . .
. . # # # # # #
. . . . . . . .
# # # # # # . .
. . . . . . . .
. . . . . . . E""",

    # 4. Dead ends with backtracking needed
    """\
S . . # . . . .
. # . # . # # .
. # . . . . . .
. # # # # # . .
. . . . . # . .
. # # # . # . #
. . . . . . . .
. . . . . . . E""",

    # 5. UNSOLVABLE: surrounded by walls
    """\
S . . . . . . .
. . . . . . . .
. . # # # # . .
. . # . . # . .
. . # . E # . .
. . # # # # . .
. . . . . . . .
. . . . . . . .""",

    # 6. Spiral-like maze
    """\
S . . . . . . .
# # # # # # . .
. . . . . . . .
. . # # # # # #
. . # . . . . .
. . # . # # # .
. . . . # . . .
. . . . . . . E""",

    # 7. Dense obstacles
    """\
S . # . # . . .
. . # . # . # .
. . . . . . # .
# . # # # . . .
. . # . . . # .
. . . . # . # .
. # . # . . . .
. . . . . . . E""",

    # 8. Zigzag path
    """\
S . . . # . . .
# # # . # . # .
. . . . . . # .
. # # # # . . .
. . . . # . # #
# # . # . . . .
. . . # . # # .
. . . . . . . E""",
]


def _build_items(grid_strings, difficulty):
    """Build item dicts from grid strings, computing BFS answers at load time."""
    items = []
    for grid_str in grid_strings:
        grid = _parse_grid(grid_str)
        answer = bfs_shortest_path(grid)
        display = _grid_to_str(grid)
        items.append({
            "grid": display,
            "answer": answer,
            "difficulty": difficulty,
        })
    return items


EASY_ITEMS = _build_items(_EASY_GRIDS, "easy")
HARD_ITEMS = _build_items(_HARD_GRIDS, "hard")

PROMPT_TEMPLATE = """\
{grid}

What is the length of the shortest path from S to E?
Count each step. If no path exists, answer -1.
Answer with only a number."""


def score_pathfinding(response: str, expected: int) -> float:
    """Score a pathfinding response. Exact match=1.0, off by 1=0.5, else 0.0."""
    import re
    response = response.strip()
    # Extract last integer (handles "The answer is 8" etc.)
    matches = re.findall(r'-?\d+', response)
    if not matches:
        return 0.0
    try:
        got = int(matches[-1])
    except ValueError:
        return 0.0
    if got == expected:
        return 1.0
    if abs(got - expected) == 1:
        return 0.5
    return 0.0


@register_probe
class SpatialPathfindingProbe(BaseProbe):
    name = "spatial_pathfinding"
    description = "Shortest path in ASCII grids — parietal lobe / spatial navigation circuits"

    def run(self, model) -> dict:
        easy_scores = []
        hard_scores = []
        item_results = [] if self.log_responses else None

        for item in EASY_ITEMS:
            prompt = PROMPT_TEMPLATE.format(grid=item["grid"])
            response = model.generate_short(prompt, max_new_tokens=10, temperature=0.0)
            score = score_pathfinding(response, item["answer"])
            easy_scores.append(score)
            if item_results is not None:
                item_results.append({
                    "difficulty": "easy",
                    "expected": item["answer"],
                    "response": response[:200],
                    "score": score,
                })

        for item in HARD_ITEMS:
            prompt = PROMPT_TEMPLATE.format(grid=item["grid"])
            response = model.generate_short(prompt, max_new_tokens=10, temperature=0.0)
            score = score_pathfinding(response, item["answer"])
            hard_scores.append(score)
            if item_results is not None:
                item_results.append({
                    "difficulty": "hard",
                    "expected": item["answer"],
                    "response": response[:200],
                    "score": score,
                })

        return self._make_result(easy_scores, hard_scores, item_results)
