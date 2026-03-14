"""
Math reasoning probe — hard estimation/calculation questions.

Output: single integer. Scored with partial credit for near-misses.
Maps to: prefrontal cortex / mathematical reasoning circuits.
"""

from probes.registry import BaseProbe, register_probe

EASY_ITEMS = [
    {"prompt": "What is 17 * 23? Answer with only the number.", "answer": 391},
    {"prompt": "What is 144 / 12? Answer with only the number.", "answer": 12},
    {"prompt": "What is 2^10? Answer with only the number.", "answer": 1024},
    {"prompt": "What is 999 - 573? Answer with only the number.", "answer": 426},
    {"prompt": "What is 15! / 14!? Answer with only the number.", "answer": 15},
    {"prompt": "What is 3^5? Answer with only the number.", "answer": 243},
    {"prompt": "What is the GCD of 48 and 36? Answer with only the number.", "answer": 12},
    {"prompt": "What is the 7th prime number? Answer with only the number.", "answer": 17},
]

HARD_ITEMS = [
    {"prompt": "What is the sum of integers from 1 to 20? Answer with only the number.", "answer": 210},
    {"prompt": "A triangle has sides 3, 4, 5. What is its area? Answer with only the number.", "answer": 6},
    {"prompt": "How many degrees in a regular pentagon's interior angle? Answer with only the number.", "answer": 108},
    {"prompt": "What is 256 in base 2 length (number of binary digits)? Answer with only the number.", "answer": 9},
    {"prompt": "What is the cube root of 27000? Answer with only the number.", "answer": 30},
    {"prompt": "What is 17^3? Answer with only the number.", "answer": 4913},
    {"prompt": "What is the sum of the first 15 square numbers? Answer with only the number.", "answer": 1240},
    {"prompt": "What is the LCM of 12 and 18? Answer with only the number.", "answer": 36},
]

# Legacy alias for backward compatibility
QUESTIONS = EASY_ITEMS + HARD_ITEMS


def score_math(response: str, expected: int) -> float:
    """Score a numeric response with partial credit."""
    response = response.strip()
    # Extract LAST number from response (e.g. "17 * 23 = 391" -> 391)
    import re
    matches = re.findall(r'-?\d+', response)
    if not matches:
        return 0.0

    try:
        got = int(matches[-1])
    except ValueError:
        return 0.0

    if got == expected:
        return 1.0

    # Partial credit based on relative error
    if expected != 0:
        rel_error = abs(got - expected) / abs(expected)
        if rel_error < 0.01:
            return 0.9
        if rel_error < 0.05:
            return 0.7
        if rel_error < 0.1:
            return 0.5
        if rel_error < 0.2:
            return 0.3
    return 0.0


@register_probe
class MathProbe(BaseProbe):
    name = "math"
    description = "Hard math estimation — prefrontal cortex circuits"

    def run(self, model) -> dict:
        easy_scores = []
        for q in EASY_ITEMS:
            response = model.generate_short(q["prompt"], max_new_tokens=15, temperature=0.0)
            score = score_math(response, q["answer"])
            easy_scores.append(score)

        hard_scores = []
        for q in HARD_ITEMS:
            response = model.generate_short(q["prompt"], max_new_tokens=15, temperature=0.0)
            score = score_math(response, q["answer"])
            hard_scores.append(score)

        return self._make_result(easy_scores, hard_scores)
