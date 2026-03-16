"""
Implication probe — syllogistic reasoning under belief bias.

Tests whether the model can distinguish logically valid from invalid
arguments, especially when intuition conflicts with formal logic.

Output: single word ("valid" or "invalid").
Maps to: prefrontal cortex / deductive reasoning circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

EASY_ITEMS = [
    {
        "prompt": (
            "Premise 1: All mammals are warm-blooded. "
            "Premise 2: All dogs are mammals. "
            "Conclusion: All dogs are warm-blooded. "
            "Is this argument valid or invalid? Answer with one word."
        ),
        "answer": "valid",
    },
    {
        "prompt": (
            "Premise 1: All birds can fly. "
            "Premise 2: Penguins are birds. "
            "Conclusion: Penguins can fly. "
            "Is this argument valid or invalid? Answer with one word."
        ),
        "answer": "valid",  # logically valid given the premises, even if P1 is false
    },
    {
        "prompt": (
            "Premise 1: All cats are animals. "
            "Premise 2: Some animals are pets. "
            "Conclusion: All cats are pets. "
            "Is this argument valid or invalid? Answer with one word."
        ),
        "answer": "invalid",
    },
    {
        "prompt": (
            "Premise 1: No reptiles are mammals. "
            "Premise 2: All snakes are reptiles. "
            "Conclusion: No snakes are mammals. "
            "Is this argument valid or invalid? Answer with one word."
        ),
        "answer": "valid",
    },
    {
        "prompt": (
            "Premise 1: All roses are flowers. "
            "Premise 2: Some flowers are red. "
            "Conclusion: Some roses are red. "
            "Is this argument valid or invalid? Answer with one word."
        ),
        "answer": "invalid",
    },
    {
        "prompt": (
            "Premise 1: If it rains, the ground gets wet. "
            "Premise 2: It is raining. "
            "Conclusion: The ground is wet. "
            "Is this argument valid or invalid? Answer with one word."
        ),
        "answer": "valid",
    },
]

HARD_ITEMS = [
    {
        # Belief-bias: conclusion is intuitively true but logically invalid
        "prompt": (
            "Premise 1: All flowers need water. "
            "Premise 2: Roses need water. "
            "Conclusion: Roses are flowers. "
            "Is this argument valid or invalid? Answer with one word."
        ),
        "answer": "invalid",  # affirming the consequent
    },
    {
        # Belief-bias: conclusion is intuitively false but logically valid
        "prompt": (
            "Premise 1: All things made of cheese are edible. "
            "Premise 2: The moon is made of cheese. "
            "Conclusion: The moon is edible. "
            "Is this argument valid or invalid? Answer with one word."
        ),
        "answer": "valid",  # valid given premises, though P2 is false
    },
    {
        # Belief-bias: intuitively true conclusion, invalid logic
        "prompt": (
            "Premise 1: All fish live in water. "
            "Premise 2: Whales live in water. "
            "Conclusion: Whales are fish. "
            "Is this argument valid or invalid? Answer with one word."
        ),
        "answer": "invalid",  # affirming the consequent
    },
    {
        # Denying the antecedent
        "prompt": (
            "Premise 1: If a person is a doctor, they went to medical school. "
            "Premise 2: John did not go to medical school. "
            "Conclusion: John is not a doctor. "
            "Is this argument valid or invalid? Answer with one word."
        ),
        "answer": "valid",  # modus tollens — this IS valid
    },
    {
        # Belief-bias: sounds right, logically wrong
        "prompt": (
            "Premise 1: All criminals break the law. "
            "Premise 2: John broke the law. "
            "Conclusion: John is a criminal. "
            "Is this argument valid or invalid? Answer with one word."
        ),
        "answer": "invalid",  # affirming the consequent
    },
    {
        # Denying the antecedent (invalid)
        "prompt": (
            "Premise 1: If it snows, school is cancelled. "
            "Premise 2: It did not snow. "
            "Conclusion: School is not cancelled. "
            "Is this argument valid or invalid? Answer with one word."
        ),
        "answer": "invalid",  # denying the antecedent
    },
]


def score_implication(response: str, expected: str) -> float:
    """Score an implication response. Exact match only."""
    r = response.strip().lower()
    # Check "invalid" before "valid" since "invalid" contains "valid"
    if "invalid" in r:
        return 1.0 if expected == "invalid" else 0.0
    if "valid" in r:
        return 1.0 if expected == "valid" else 0.0
    return 0.0


@register_probe
class ImplicationProbe(BaseProbe):
    name = "implication"
    description = "Syllogistic reasoning under belief bias — deductive reasoning circuits"

    def run(self, model) -> dict:
        easy_scores = []
        hard_scores = []
        item_results = [] if self.log_responses else None

        for difficulty, items in [("easy", self._limit(EASY_ITEMS)), ("hard", self._limit(HARD_ITEMS))]:
            scores = easy_scores if difficulty == "easy" else hard_scores
            for item in items:
                response = model.generate_short(
                    item["prompt"], max_new_tokens=5, temperature=0.0
                )
                score = score_implication(response, item["answer"])
                scores.append(score)
                if item_results is not None:
                    item_results.append({
                        "difficulty": difficulty,
                        "expected": item["answer"],
                        "response": response[:200],
                        "score": score,
                    })

        return self._make_result(easy_scores, hard_scores, item_results)
