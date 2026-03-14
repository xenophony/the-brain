"""
Spatial reasoning probe — Battleship next-move prediction.

Given a partially revealed Battleship board, predict the optimal 
next shot. Scored against a minimax/probability-density oracle.

Fast: output is a single coordinate like "B4".
Objective: oracle score measures quality of spatial inference.

Maps to: parietal lobe / visual cortex spatial processing.
"""

import re
from probes.registry import BaseProbe, register_probe


# Board scenarios: (board_state_text, oracle_best_moves, explanation)
# H=hit, M=miss, ?=unknown, ~=water confirmed
# Oracle best moves are ranked — full credit for top pick, partial for adjacent

SCENARIOS = [
    {
        "board": """
     A  B  C  D  E  F  G  H  I  J
  1  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  2  ?  ?  ?  H  ?  ?  ?  ?  ?  ?
  3  ?  ?  ?  H  ?  ?  ?  ?  ?  ?
  4  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  5  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  6  M  ?  ?  ?  ?  ?  ?  ?  ?  ?
  7  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  8  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  9  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
 10  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
""",
        "best_moves": ["D1", "D4"],   # extend the vertical hit run
        "partial_credit": ["D5", "C2", "C3", "E2", "E3"],
        "rationale": "Two vertical hits at D2,D3 suggest a vertical ship — extend up or down"
    },
    {
        "board": """
     A  B  C  D  E  F  G  H  I  J
  1  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  2  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  3  ?  H  H  H  ?  ?  ?  ?  ?  ?
  4  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  5  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  6  ?  ?  ?  ?  ?  M  ?  ?  ?  ?
  7  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  8  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  9  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
 10  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
""",
        "best_moves": ["A3", "E3"],   # extend horizontal run
        "partial_credit": ["F3"],
        "rationale": "Three horizontal hits at B3,C3,D3 — extend left to A3 or right to E3"
    },
    {
        "board": """
     A  B  C  D  E  F  G  H  I  J
  1  M  M  M  M  M  ?  ?  ?  ?  ?
  2  M  ?  ?  ?  M  ?  ?  ?  ?  ?
  3  M  ?  ?  ?  M  ?  ?  ?  ?  ?
  4  M  M  M  M  M  ?  ?  ?  ?  ?
  5  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  6  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  7  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  8  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  9  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
 10  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
""",
        "best_moves": ["B2", "C2", "D2", "B3", "C3", "D3"],  # dense unknown interior
        "partial_credit": ["F5", "G5", "H5"],
        "rationale": "Top-left zone has dense misses around an unexplored interior — high probability region"
    },
    {
        "board": """
     A  B  C  D  E  F  G  H  I  J
  1  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  2  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  3  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  4  ?  ?  ?  ?  H  ?  ?  ?  ?  ?
  5  ?  ?  ?  M  H  M  ?  ?  ?  ?
  6  ?  ?  ?  ?  H  ?  ?  ?  ?  ?
  7  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  8  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
  9  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
 10  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
""",
        "best_moves": ["E3", "E7"],  # vertical ship, flanks missed, extend further
        "partial_credit": ["E2", "E8"],
        "rationale": "Vertical hits E4,E5,E6 with horizontal misses at D5,F5 — ship is vertical, extend up/down"
    },
    {
        "board": """
     A  B  C  D  E  F  G  H  I  J
  1  M  ?  M  ?  M  ?  M  ?  M  ?
  2  ?  M  ?  M  ?  M  ?  M  ?  M
  3  M  ?  M  ?  M  ?  M  ?  M  ?
  4  ?  M  ?  M  ?  M  ?  M  ?  M
  5  M  ?  M  ?  M  ?  M  ?  M  ?
  6  ?  M  ?  M  ?  M  ?  M  ?  M
  7  M  ?  M  ?  M  ?  M  ?  M  ?
  8  ?  M  ?  M  ?  M  ?  M  ?  M
  9  M  ?  M  ?  M  ?  M  ?  M  ?
 10  ?  M  ?  M  ?  M  ?  M  ?  M
""",
        "best_moves": ["B1", "D1", "F1", "H1", "J1", "A2", "C2"],  # checkerboard — any unknown
        "partial_credit": ["B3", "D3", "J3"],
        "rationale": "Checkerboard pattern — all unknowns are equally valid, any is correct"
    },
]

PROMPT_TEMPLATE = """You are playing Battleship. The board shows:
H = hit, M = miss, ? = unknown

{board}

Based on the hits and misses, what is the single best next shot?
Output only the coordinate (e.g. "B4"). No explanation.

Best next shot:"""


def score_response(response: str, scenario: dict) -> float:
    """Score a move response against the oracle."""
    # Extract coordinate from response
    response = response.strip().upper()
    match = re.search(r'([A-J])\s*([1-9]|10)', response)
    if not match:
        return 0.0
    
    move = match.group(1) + match.group(2)
    
    if move in scenario["best_moves"]:
        return 1.0
    if move in scenario["partial_credit"]:
        return 0.5
    return 0.0


@register_probe
class SpatialProbe(BaseProbe):
    name = "spatial"
    description = "Battleship next-move spatial reasoning — parietal/visual circuits"
    
    def run(self, model) -> float:
        scores = []
        
        for scenario in SCENARIOS:
            prompt = PROMPT_TEMPLATE.format(board=scenario["board"])
            response = model.generate_short(prompt, max_new_tokens=5, temperature=0.0)
            score = score_response(response, scenario)
            scores.append(score)
            
        return sum(scores) / len(scores)
