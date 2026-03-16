"""
Reasoning probe — multi-step problems that require chain-of-thought.

Designed to measure the "thinking circuit": problems where CoT thinking
dramatically improves accuracy vs direct answer. The gap between thinking
and no-think scores on these items identifies which layers power the
reasoning chain.

Output: single integer.
Maps to: prefrontal cortex / executive reasoning circuits.
Token budget: 500 (enough for CoT + answer).
"""

import re
from probes.registry import BaseProbe, register_probe

# Easy: 2-3 step reasoning, thinking helps but not strictly required
EASY_ITEMS = [
    {
        "prompt": "A store sells apples for $2 each and oranges for $3 each. "
                  "Maria buys 4 apples and 5 oranges. She pays with a $50 bill. "
                  "How much change does she get? Answer with only the number.",
        "answer": 27,
    },
    {
        "prompt": "A train travels at 80 km/h for 3 hours, then 60 km/h for 2 hours. "
                  "What is the total distance in km? Answer with only the number.",
        "answer": 360,
    },
    {
        "prompt": "There are 5 teams in a tournament. Each team plays every other team once. "
                  "How many games are played in total? Answer with only the number.",
        "answer": 10,
    },
]

# Hard: 3-5 step reasoning, very unlikely to solve without thinking
HARD_ITEMS = [
    {
        "prompt": "A tank fills at 3 liters per minute and drains at 1 liter per minute. "
                  "Starting empty, it runs for 20 minutes filling, then the fill stops "
                  "and only the drain runs. How many minutes after the fill stops until "
                  "the tank is empty? Answer with only the number.",
        "answer": 40,
    },
    {
        "prompt": "Alice is twice as old as Bob. In 10 years, Alice will be 1.5 times "
                  "as old as Bob. How old is Bob now? Answer with only the number.",
        "answer": 20,
    },
    {
        "prompt": "A snail climbs 3 meters up a wall each day but slides back 2 meters "
                  "each night. The wall is 10 meters tall. On which day does the snail "
                  "reach the top? Answer with only the number.",
        "answer": 8,
    },
    {
        "prompt": "You have 3 boxes. Box A has 2 red and 3 blue balls. Box B has 4 red "
                  "and 1 blue ball. Box C has 1 red and 4 blue balls. You pick a box at "
                  "random and draw a ball. It is red. What is the probability (as a "
                  "percentage, rounded to the nearest integer) that it came from Box B? "
                  "Answer with only the number.",
        "answer": 57,
    },
    {
        "prompt": "A clock's minute hand is 10 cm long. How far in cm does the tip of the "
                  "minute hand travel in 45 minutes? Round to the nearest integer. "
                  "Answer with only the number.",
        "answer": 47,
    },
    {
        "prompt": "Three workers can build a wall in 12, 15, and 20 days respectively "
                  "working alone. If all three work together, how many days will it take? "
                  "Answer with only the number.",
        "answer": 5,
    },
]


def score_reasoning(response: str, expected: int) -> float:
    """Score with partial credit for near-misses."""
    response = response.strip()
    matches = re.findall(r'-?\d+', response)
    if not matches:
        return 0.0
    try:
        got = int(matches[-1])
    except ValueError:
        return 0.0
    if got == expected:
        return 1.0
    if expected != 0:
        rel_error = abs(got - expected) / abs(expected)
        if rel_error < 0.05:
            return 0.8
        if rel_error < 0.1:
            return 0.5
        if rel_error < 0.2:
            return 0.3
    return 0.0


@register_probe
class ReasoningProbe(BaseProbe):
    name = "reasoning"
    description = "Multi-step reasoning — thinking circuit probe"

    def run(self, model) -> dict:
        easy_scores = []
        hard_scores = []
        item_results = [] if self.log_responses else None

        for difficulty, items in [("easy", self._limit(EASY_ITEMS)), ("hard", self._limit(HARD_ITEMS))]:
            scores = easy_scores if difficulty == "easy" else hard_scores
            for item in items:
                response = model.generate_short(
                    item["prompt"], max_new_tokens=500, temperature=0.0
                )
                score = score_reasoning(response, item["answer"])
                scores.append(score)
                if item_results is not None:
                    item_results.append({
                        "difficulty": difficulty,
                        "expected": item["answer"],
                        "response": response[:500],
                        "score": score,
                    })

        return self._make_result(easy_scores, hard_scores, item_results)
