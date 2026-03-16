"""
Factual recall probe — genuinely obscure verifiable facts.

Questions are calibrated to challenge 30B models: well-known facts
produce ceiling effects. These require precise recall of specific
numeric values and uncommon proper nouns.

Output: number or single word.
Scoring: exact match = 1.0, off-by-one (numbers) = 0.5, else 0.0.
Maps to: hippocampus / factual memory circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

EASY_ITEMS = [
    {
        "prompt": "What element has atomic number 79? Answer with only the element name.",
        "answer": "gold",
        "type": "word",
        "alternates": [],
    },
    {
        "prompt": "In what year did the Berlin Wall fall? Answer with only the number.",
        "answer": "1989",
        "type": "number",
    },
    {
        "prompt": "What is the chemical formula for water? Answer with only the formula.",
        "answer": "h2o",
        "type": "word",
        "alternates": [],
    },
    {
        "prompt": "How many chromosomes do humans have? Answer with only the number.",
        "answer": "46",
        "type": "number",
    },
    {
        "prompt": "What planet is closest to the Sun? Answer with only the planet name.",
        "answer": "mercury",
        "type": "word",
        "alternates": [],
    },
    {
        "prompt": "What is the speed of sound in m/s at sea level (approximate)? Answer with only the number.",
        "answer": "343",
        "type": "number",
    },
    {
        "prompt": "What is the largest organ in the human body? Answer with only the organ name.",
        "answer": "skin",
        "type": "word",
        "alternates": [],
    },
    {
        "prompt": "In what year was the first iPhone released? Answer with only the number.",
        "answer": "2007",
        "type": "number",
    },
    {
        "prompt": "What is the most abundant gas in Earth's atmosphere? Answer with only the gas name.",
        "answer": "nitrogen",
        "type": "word",
        "alternates": [],
    },
    {
        "prompt": "How many bones are in the adult human body? Answer with only the number.",
        "answer": "206",
        "type": "number",
    },
]

HARD_ITEMS = [
    {
        "prompt": "What is the atomic number of molybdenum? Answer with only the number.",
        "answer": "42",
        "type": "number",
    },
    {
        "prompt": "In what year was the Treaty of Westphalia signed? Answer with only the number.",
        "answer": "1648",
        "type": "number",
    },
    {
        "prompt": "What is the capital of Bhutan? Answer with only the city name.",
        "answer": "thimphu",
        "type": "word",
        "alternates": [],
    },
    {
        "prompt": "How many teeth does an adult human normally have? Answer with only the number.",
        "answer": "32",
        "type": "number",
    },
    {
        "prompt": "What element has the chemical symbol 'W'? Answer with only the element name.",
        "answer": "tungsten",
        "type": "word",
        "alternates": ["wolfram"],
    },
    {
        "prompt": "In what year did the Chernobyl disaster occur? Answer with only the number.",
        "answer": "1986",
        "type": "number",
    },
    {
        "prompt": "What is the deepest lake in the world? Answer with only the lake name.",
        "answer": "baikal",
        "type": "word",
        "alternates": ["lake baikal"],
    },
    {
        "prompt": "What is the diameter of Earth in kilometers (approximate)? Answer with only the number.",
        "answer": "12742",
        "type": "number",
    },
    {
        "prompt": "What country has the most time zones? Answer with only the country name.",
        "answer": "france",
        "type": "word",
        "alternates": [],
    },
    {
        "prompt": "What is the boiling point of ethanol in Celsius? Answer with only the number.",
        "answer": "78",
        "type": "number",
    },
]

# Legacy alias
QUESTIONS = EASY_ITEMS + HARD_ITEMS


def score_factual(response: str, question: dict) -> float:
    """Score a factual response. Exact match = 1.0, near-miss = 0.5, else 0.0."""
    response = response.strip().lower()

    if question["type"] == "word":
        expected = question["answer"].lower()
        alternates = [a.lower() for a in question.get("alternates", [])]
        if expected in response:
            return 1.0
        for alt in alternates:
            if alt in response:
                return 1.0
        return 0.0

    # Number type — extract LAST number from response
    matches = re.findall(r'-?[\d]+\.?[\d]*', response)
    if not matches:
        return 0.0

    try:
        got = float(matches[-1])
        expected = float(question["answer"])
    except ValueError:
        return 0.0

    if got == expected:
        return 1.0

    # Off-by-one for integers
    if expected == int(expected) and got == int(got):
        if abs(got - expected) == 1:
            return 0.5

    # Relative tolerance for larger numbers (within 1%)
    if expected != 0:
        rel_error = abs(got - expected) / abs(expected)
        if rel_error < 0.01:
            return 0.5

    return 0.0


@register_probe
class FactualProbe(BaseProbe):
    name = "factual"
    description = "Obscure fact recall — hippocampus circuits"

    def run(self, model) -> dict:
        easy_scores = []
        for q in self._limit(EASY_ITEMS):
            response = model.generate_short(q["prompt"], max_new_tokens=15, temperature=0.0)
            score = score_factual(response, q)
            easy_scores.append(score)

        hard_scores = []
        for q in self._limit(HARD_ITEMS):
            response = model.generate_short(q["prompt"], max_new_tokens=15, temperature=0.0)
            score = score_factual(response, q)
            hard_scores.append(score)

        return self._make_result(easy_scores, hard_scores)
