"""
Estimation probe — Fermi estimation with log-scale partial credit.

Tests order-of-magnitude reasoning on well-known and multi-step
estimation questions.

Output: single number.
Maps to: prefrontal cortex / quantitative reasoning circuits.
"""

import re
import math
from probes.registry import BaseProbe, register_probe

EASY_ITEMS = [
    {"prompt": "How many seconds are in one day? Answer with only a number, no units or explanation.", "answer": 86400},
    {"prompt": "How many bones are in the adult human body? Answer with only a number, no units or explanation.", "answer": 206},
    {"prompt": "How many countries are in the United Nations? Answer with only a number, no units or explanation.", "answer": 193},
    {"prompt": "How many miles is the Earth from the Sun? Answer with only a number, no units or explanation.", "answer": 93000000},
    {"prompt": "How many teeth does an adult human have? Answer with only a number, no units or explanation.", "answer": 32},
    {"prompt": "How many minutes are in one week? Answer with only a number, no units or explanation.", "answer": 10080},
]

HARD_ITEMS = [
    {"prompt": "How many piano tuners are in Chicago? Answer with only a number, no units or explanation.", "answer": 225},
    {"prompt": "How many words does the average person speak per day? Answer with only a number, no units or explanation.", "answer": 16000},
    {"prompt": "How many golf balls fit in a school bus? Answer with only a number, no units or explanation.", "answer": 500000},
    {"prompt": "How many gas stations are in the United States? Answer with only a number, no units or explanation.", "answer": 150000},
    {"prompt": "How many tennis balls fit in this room (a standard 4m x 5m x 3m room)? Answer with only a number, no units or explanation.", "answer": 250000},
    {"prompt": "How many total flights take off worldwide each day? Answer with only a number, no units or explanation.", "answer": 100000},
]


def extract_number(response: str) -> float | None:
    """Extract the last number from a response, handling commas and scientific notation."""
    response = response.strip()
    # Remove commas in numbers
    response = response.replace(",", "")
    # Try scientific notation first (e.g., 9.3e7, 9.3 x 10^7)
    sci_matches = re.findall(r'(\d+\.?\d*)\s*[xX*]\s*10\^?(\d+)', response)
    if sci_matches:
        base, exp = sci_matches[-1]
        return float(base) * (10 ** int(exp))
    # Standard numbers (integer or float)
    matches = re.findall(r'-?\d+\.?\d*', response)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def score_estimation(response: str, expected: int) -> float:
    """Score a Fermi estimation with log-scale partial credit.

    Returns:
        1.0 if within 0.1 log orders of magnitude
        0.8 if within 0.3
        0.6 if within 0.5
        0.4 if within 1.0
        0.2 if within 2.0
        0.0 otherwise
    """
    got = extract_number(response)
    if got is None:
        return 0.0

    # Special case: if expected or got is 0 or negative, exact match only
    if expected <= 0 or got <= 0:
        return 1.0 if got == expected else 0.0

    try:
        log_error = abs(math.log10(got) - math.log10(expected))
    except (ValueError, ZeroDivisionError):
        return 0.0

    if log_error < 0.1:
        return 1.0
    if log_error < 0.3:
        return 0.8
    if log_error < 0.5:
        return 0.6
    if log_error < 1.0:
        return 0.4
    if log_error < 2.0:
        return 0.2
    return 0.0


@register_probe
class EstimationProbe(BaseProbe):
    name = "estimation"
    description = "Fermi estimation — quantitative reasoning circuits"

    def run(self, model) -> dict:
        easy_scores = []
        hard_scores = []
        item_results = [] if self.log_responses else None

        for difficulty, items in [("easy", self._limit(EASY_ITEMS)), ("hard", self._limit(HARD_ITEMS))]:
            scores = easy_scores if difficulty == "easy" else hard_scores
            for item in items:
                response = model.generate_short(
                    item["prompt"], max_new_tokens=15, temperature=0.0
                )
                score = score_estimation(response, item["answer"])
                scores.append(score)
                if item_results is not None:
                    item_results.append({
                        "difficulty": difficulty,
                        "expected": item["answer"],
                        "response": response[:200],
                        "extracted": extract_number(response),
                        "score": score,
                    })

        return self._make_result(easy_scores, hard_scores, item_results)
