"""
Emotional intelligence probe — EQ-Bench style intensity estimation.

Presents social scenarios and asks how intensely a person would feel
a specific emotion on a 0-9 scale. Scored with partial credit for
near-misses using expected_digit_score() from BaseProbe.

Output: single digit 0-9.
Maps to: limbic system / emotional processing circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

EASY_ITEMS = [
    {
        "prompt": (
            "You discover that a trusted partner has been lying to you for months. "
            "How intensely would you feel betrayal on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 9,
    },
    {
        "prompt": (
            "After weeks of waiting, you learn that a biopsy result is benign. "
            "How intensely would you feel relief on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 9,
    },
    {
        "prompt": (
            "Your computer crashes and you lose an entire day's unsaved work. "
            "How intensely would you feel frustration on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "You finish a marathon for the first time after months of training. "
            "How intensely would you feel pride on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "Someone cuts in front of you in a long queue and smirks. "
            "How intensely would you feel anger on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "A stranger helps you change a flat tire in the rain without asking for anything. "
            "How intensely would you feel gratitude on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "You accidentally send a private complaint about your boss to your boss. "
            "How intensely would you feel shame on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "Your child wins a school science fair you helped them prepare for. "
            "How intensely would you feel joy on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
]

HARD_ITEMS = [
    {
        "prompt": (
            "A colleague takes credit for your work in a meeting. "
            "How intensely would you feel guilt on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 2,
    },
    {
        "prompt": (
            "Your best friend gets the promotion you were both competing for. "
            "How intensely would you feel jealousy on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 5,
    },
    {
        "prompt": (
            "You spend a quiet Sunday afternoon reading in a sunlit garden. "
            "How intensely would you feel contentment on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "You receive an unexpected gift from a friend you haven't spoken to in years. "
            "How intensely would you feel surprise on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "You are waiting for important medical test results that will arrive tomorrow. "
            "How intensely would you feel anxiety on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "You win an award but know a colleague deserved it more. "
            "How intensely would you feel ambivalence on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 6,
    },
    {
        "prompt": (
            "Your ex-partner who hurt you badly asks you for help during a crisis. "
            "How intensely would you feel conflicted on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "You get a standing ovation after a presentation you thought went poorly. "
            "How intensely would you feel disbelief on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 6,
    },
]

# Legacy alias
SCENARIOS = EASY_ITEMS + HARD_ITEMS


def _extract_eq_digit(response: str) -> int | None:
    """Extract emotion intensity digit from response.

    Uses the LAST single digit to avoid catching digits from
    echoed scenario text (e.g. "I would rate this a 7").
    """
    r = response.strip()
    # Try last single digit (word-bounded) in response
    digits = re.findall(r'\b(\d)\b', r)
    if digits:
        return int(digits[-1])
    # Try any digit (last one)
    for ch in reversed(r):
        if ch.isdigit():
            return int(ch)
    return None


def _eq_digit_score(response: str, expected: int) -> float:
    """Score an EQ response with partial credit for near-misses."""
    got = _extract_eq_digit(response)
    if got is None:
        return 0.0
    if got == expected:
        return 1.0
    diff = abs(got - expected)
    if diff == 1:
        return 0.5
    if diff == 2:
        return 0.25
    return 0.0


@register_probe
class EQProbe(BaseProbe):
    name = "eq"
    description = "Emotional intensity estimation — limbic system circuits"

    def run(self, model) -> dict:
        easy_scores, easy_results = self._run_items(
            model, self._limit(EASY_ITEMS),
            prompt_fn=lambda item: item["prompt"],
            score_fn=lambda resp, item: _eq_digit_score(resp, item["expected"]),
            max_new_tokens=5, difficulty="easy")

        hard_scores, hard_results = self._run_items(
            model, self._limit(HARD_ITEMS),
            prompt_fn=lambda item: item["prompt"],
            score_fn=lambda resp, item: _eq_digit_score(resp, item["expected"]),
            max_new_tokens=5, difficulty="hard")

        item_results = (easy_results + hard_results) if self.log_responses else None
        return self._make_result(easy_scores, hard_scores, item_results)
