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
    {"prompt": "What is 45 + 78? Answer with only the number.", "answer": 123},
    {"prompt": "What is 200 - 137? Answer with only the number.", "answer": 63},
    {"prompt": "What is 25 * 4? Answer with only the number.", "answer": 100},
    {"prompt": "What is 360 / 9? Answer with only the number.", "answer": 40},
    {"prompt": "What is 50% of 84? Answer with only the number.", "answer": 42},
    {"prompt": "What is 11 squared? Answer with only the number.", "answer": 121},
    {"prompt": "What is the square root of 169? Answer with only the number.", "answer": 13},
    {"prompt": "What is 7 * 8 * 2? Answer with only the number.", "answer": 112},
    {"prompt": "What is 1000 - 625? Answer with only the number.", "answer": 375},
    {"prompt": "What is 33 + 67 + 100? Answer with only the number.", "answer": 200},
    {"prompt": "What is 15% of 200? Answer with only the number.", "answer": 30},
    {"prompt": "What is 2^8? Answer with only the number.", "answer": 256},
    {"prompt": "How many sides does a hexagon have? Answer with only the number.", "answer": 6},
    {"prompt": "What is the perimeter of a square with side 9? Answer with only the number.", "answer": 36},
    {"prompt": "What is 81 / 3? Answer with only the number.", "answer": 27},
    {"prompt": "What is 14 * 11? Answer with only the number.", "answer": 154},
    {"prompt": "What is the next number in the sequence 2, 4, 8, 16? Answer with only the number.", "answer": 32},
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
    # GSM8K-style word problems (1-2 step)
    {"prompt": "A baker made 48 cookies and packed them in boxes of 6. How many boxes did he fill? Answer with only the number.", "answer": 8},
    {"prompt": "A store had 156 apples. They sold 89 apples. How many apples are left? Answer with only the number.", "answer": 67},
    {"prompt": "If a car travels at 60 km/h for 2.5 hours, how many kilometers does it travel? Answer with only the number.", "answer": 150},
    {"prompt": "A rectangle is 12 cm long and 8 cm wide. What is its area in square cm? Answer with only the number.", "answer": 96},
    {"prompt": "If 3 friends split a $45 bill equally, how much does each person pay in dollars? Answer with only the number.", "answer": 15},
    {"prompt": "A train leaves at 9:15 AM and arrives at 11:45 AM. How many minutes is the journey? Answer with only the number.", "answer": 150},
    {"prompt": "A factory produces 350 widgets per hour. How many widgets in 4 hours? Answer with only the number.", "answer": 1400},
    {"prompt": "If a shirt costs $80 after a 20% discount, what was the original price in dollars? Answer with only the number.", "answer": 100},
    {"prompt": "How many ways can you choose 2 items from a group of 6? Answer with only the number.", "answer": 15},
    {"prompt": "A circular pizza is cut into 8 equal slices. How many degrees is each slice? Answer with only the number.", "answer": 45},
    {"prompt": "What is 13 * 17? Answer with only the number.", "answer": 221},
    {"prompt": "A pool fills at 3 gallons per minute. How many gallons after 45 minutes? Answer with only the number.", "answer": 135},
    {"prompt": "If you invest $1000 at 10% simple interest for 3 years, what is the total interest earned in dollars? Answer with only the number.", "answer": 300},
    {"prompt": "What is the sum of interior angles of a hexagon in degrees? Answer with only the number.", "answer": 720},
    {"prompt": "A bookshelf has 5 shelves with 12 books each. You remove 17 books. How many remain? Answer with only the number.", "answer": 43},
    {"prompt": "What is the 10th term of the arithmetic sequence 3, 7, 11, 15, ...? Answer with only the number.", "answer": 39},
    {"prompt": "How many distinct ways can you arrange the letters in the word CAT? Answer with only the number.", "answer": 6},
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
        easy_scores, easy_results = self._run_items(
            model, self._limit(EASY_ITEMS),
            prompt_fn=lambda item: item["prompt"],
            score_fn=lambda resp, item: score_math(resp, item["answer"]),
            max_new_tokens=15, difficulty="easy")

        hard_scores, hard_results = self._run_items(
            model, self._limit(HARD_ITEMS),
            prompt_fn=lambda item: item["prompt"],
            score_fn=lambda resp, item: score_math(resp, item["answer"]),
            max_new_tokens=15, difficulty="hard")

        item_results = (easy_results + hard_results) if self.log_responses else None
        return self._make_result(easy_scores, hard_scores, item_results)
