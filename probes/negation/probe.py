"""
Negation probe — understanding of negation, double negation, and scope.

Tests whether the model correctly handles negated statements by presenting
paired positive/negated versions of the same fact. Models that default to
"true" for everything score 0.5 (chance level per pair).

Output: single word ("true" or "false").
Maps to: Broca/Wernicke language processing circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

EASY_PAIRS = [
    {
        "positive": {"prompt": "The Earth orbits the Sun. True or false? Answer with one word.", "answer": "true"},
        "negated": {"prompt": "The Earth does not orbit the Sun. True or false? Answer with one word.", "answer": "false"},
    },
    {
        "positive": {"prompt": "Water boils at 100 degrees Celsius at sea level. True or false? Answer with one word.", "answer": "true"},
        "negated": {"prompt": "Water does not boil at 100 degrees Celsius at sea level. True or false? Answer with one word.", "answer": "false"},
    },
    {
        "positive": {"prompt": "Humans need oxygen to survive. True or false? Answer with one word.", "answer": "true"},
        "negated": {"prompt": "Humans do not need oxygen to survive. True or false? Answer with one word.", "answer": "false"},
    },
]

HARD_PAIRS = [
    {
        # Double negation
        "positive": {"prompt": "It is true that the moon orbits the Earth. True or false? Answer with one word.", "answer": "true"},
        "negated": {"prompt": "It is not untrue that the moon orbits the Earth. True or false? Answer with one word.", "answer": "true"},
    },
    {
        # Counterintuitive negation (negating a common misconception)
        "positive": {"prompt": "Lightning never strikes the same place twice. True or false? Answer with one word.", "answer": "false"},
        "negated": {"prompt": "It is false that lightning never strikes the same place twice. True or false? Answer with one word.", "answer": "true"},
    },
    {
        # Scope negation (not all vs all not)
        "positive": {"prompt": "All swans are white. True or false? Answer with one word.", "answer": "false"},
        "negated": {"prompt": "Not all swans are white. True or false? Answer with one word.", "answer": "true"},
    },
    # --- Added pairs for convergence reliability ---
    {
        # Double negation with false base
        "positive": {"prompt": "The Sun revolves around the Earth. True or false? Answer with one word.", "answer": "false"},
        "negated": {"prompt": "It is not the case that the Sun revolves around the Earth. True or false? Answer with one word.", "answer": "true"},
    },
    {
        # Negated universal
        "positive": {"prompt": "All metals are solid at room temperature. True or false? Answer with one word.", "answer": "false"},
        "negated": {"prompt": "Not all metals are solid at room temperature. True or false? Answer with one word.", "answer": "true"},
    },
    {
        # Double negation preserving truth
        "positive": {"prompt": "Gravity pulls objects toward the Earth. True or false? Answer with one word.", "answer": "true"},
        "negated": {"prompt": "It is not false that gravity pulls objects toward the Earth. True or false? Answer with one word.", "answer": "true"},
    },
    {
        # Simple negation of true fact
        "positive": {"prompt": "The Amazon rainforest produces oxygen. True or false? Answer with one word.", "answer": "true"},
        "negated": {"prompt": "The Amazon rainforest does not produce oxygen. True or false? Answer with one word.", "answer": "false"},
    },
    {
        # Negation of false claim
        "positive": {"prompt": "Whales are fish. True or false? Answer with one word.", "answer": "false"},
        "negated": {"prompt": "Whales are not fish. True or false? Answer with one word.", "answer": "true"},
    },
    {
        # Triple negation
        "positive": {"prompt": "Diamonds are the hardest natural material. True or false? Answer with one word.", "answer": "true"},
        "negated": {"prompt": "It is not true that diamonds are not the hardest natural material. True or false? Answer with one word.", "answer": "true"},
    },
    {
        # Negated existential
        "positive": {"prompt": "Some mammals can fly. True or false? Answer with one word.", "answer": "true"},
        "negated": {"prompt": "No mammals can fly. True or false? Answer with one word.", "answer": "false"},
    },
    {
        # Simple negation of true fact
        "positive": {"prompt": "Sound travels faster in water than in air. True or false? Answer with one word.", "answer": "true"},
        "negated": {"prompt": "Sound does not travel faster in water than in air. True or false? Answer with one word.", "answer": "false"},
    },
]


def score_negation_item(response: str, expected: str) -> float:
    """Score a single true/false response. Exact match only."""
    r = response.strip().lower()
    if "false" in r:
        return 1.0 if expected == "false" else 0.0
    if "true" in r:
        return 1.0 if expected == "true" else 0.0
    return 0.0


def score_negation_pair(pos_response: str, neg_response: str,
                        pos_expected: str, neg_expected: str) -> float:
    """Score a positive/negated pair. Average of both scores."""
    pos_score = score_negation_item(pos_response, pos_expected)
    neg_score = score_negation_item(neg_response, neg_expected)
    return (pos_score + neg_score) / 2.0


@register_probe
class NegationProbe(BaseProbe):
    name = "negation"
    description = "Negation understanding — language processing circuits"

    def run(self, model) -> dict:
        easy_scores = []
        hard_scores = []
        item_results = [] if self.log_responses else None

        for difficulty, pairs in [("easy", self._limit(EASY_PAIRS)), ("hard", self._limit(HARD_PAIRS))]:
            scores = easy_scores if difficulty == "easy" else hard_scores
            for pair in pairs:
                pos_item = pair["positive"]
                neg_item = pair["negated"]

                pos_response = model.generate_short(
                    pos_item["prompt"], max_new_tokens=5, temperature=0.0
                )
                neg_response = model.generate_short(
                    neg_item["prompt"], max_new_tokens=5, temperature=0.0
                )

                pair_score = score_negation_pair(
                    pos_response, neg_response,
                    pos_item["answer"], neg_item["answer"],
                )
                scores.append(pair_score)

                if item_results is not None:
                    item_results.append({
                        "difficulty": difficulty,
                        "positive_expected": pos_item["answer"],
                        "positive_response": pos_response[:200],
                        "positive_score": score_negation_item(pos_response, pos_item["answer"]),
                        "negated_expected": neg_item["answer"],
                        "negated_response": neg_response[:200],
                        "negated_score": score_negation_item(neg_response, neg_item["answer"]),
                        "pair_score": pair_score,
                    })

        return self._make_result(easy_scores, hard_scores, item_results)
