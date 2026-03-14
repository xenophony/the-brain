"""
Emotional intelligence probe — EQ-Bench style intensity estimation.

Presents social scenarios and asks how intensely a person would feel
a specific emotion on a 0-9 scale. Scored with partial credit for
near-misses using expected_digit_score() from BaseProbe.

Output: single digit 0-9.
Maps to: limbic system / emotional processing circuits.
"""

from probes.registry import BaseProbe, register_probe

SCENARIOS = [
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
            "Someone cuts in front of you in a long queue and smirks. "
            "How intensely would you feel anger on a scale of 0 (none) to 9 (extreme)? "
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
            "After weeks of waiting, you learn that a biopsy result is benign. "
            "How intensely would you feel relief on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 9,
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
            "You finish a marathon for the first time after months of training. "
            "How intensely would you feel pride on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
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
            "You are waiting for important medical test results that will arrive tomorrow. "
            "How intensely would you feel anxiety on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
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
            "Your computer crashes and you lose an entire day's unsaved work. "
            "How intensely would you feel frustration on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
]


@register_probe
class EQProbe(BaseProbe):
    name = "eq"
    description = "Emotional intensity estimation — limbic system circuits"

    def run(self, model) -> float:
        scores = []
        for scenario in SCENARIOS:
            response = model.generate_short(
                scenario["prompt"], max_new_tokens=5, temperature=0.0
            )
            score = self.expected_digit_score(response, scenario["expected"])
            scores.append(score)
        return sum(scores) / len(scores)
