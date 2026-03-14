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
        "prompt": "What is the atomic number of gold? Answer with only the number.",
        "answer": "79",
        "type": "number",
    },
    {
        "prompt": "In what year did World War 2 end? Answer with only the number.",
        "answer": "1945",
        "type": "number",
    },
    {
        "prompt": "How many bones are in the adult human body? Answer with only the number.",
        "answer": "206",
        "type": "number",
    },
    {
        "prompt": "How many continents are there on Earth? Answer with only the number.",
        "answer": "7",
        "type": "number",
    },
    {
        "prompt": "What is the speed of light in km/s (approximate)? Answer with only the number.",
        "answer": "299792",
        "type": "number",
    },
    {
        "prompt": "How many chromosomes do humans have? Answer with only the number.",
        "answer": "46",
        "type": "number",
    },
    {
        "prompt": "What is the boiling point of water in degrees Celsius? Answer with only the number.",
        "answer": "100",
        "type": "number",
    },
    {
        "prompt": "What is the chemical symbol for iron? Answer with only the symbol.",
        "answer": "fe",
        "type": "word",
        "alternates": [],
    },
]

HARD_ITEMS = [
    {
        "prompt": "What is the melting point of tungsten in degrees Celsius? Answer with only the number.",
        "answer": "3422",
        "type": "number",
    },
    {
        "prompt": "What is the rest mass energy of an electron in keV? Answer with only the number.",
        "answer": "511",
        "type": "number",
    },
    {
        "prompt": "What is the half-life of carbon-14 in years? Answer with only the number.",
        "answer": "5730",
        "type": "number",
    },
    {
        "prompt": "What is the boiling point of nitrogen in degrees Celsius? Answer with only the number.",
        "answer": "-196",
        "type": "number",
    },
    {
        "prompt": "How many known moons does Uranus have as of 2024? Answer with only the number.",
        "answer": "28",
        "type": "number",
    },
    {
        "prompt": "What is the ionization energy of hydrogen in electron volts? Answer with only the number (2 decimal places).",
        "answer": "13.6",
        "type": "number",
    },
    {
        "prompt": "What is the Mohs hardness of topaz? Answer with only the number.",
        "answer": "8",
        "type": "number",
    },
    {
        "prompt": "In what year was the element gallium discovered? Answer with only the number.",
        "answer": "1875",
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
        for q in EASY_ITEMS:
            response = model.generate_short(q["prompt"], max_new_tokens=15, temperature=0.0)
            score = score_factual(response, q)
            easy_scores.append(score)

        hard_scores = []
        for q in HARD_ITEMS:
            response = model.generate_short(q["prompt"], max_new_tokens=15, temperature=0.0)
            score = score_factual(response, q)
            hard_scores.append(score)

        return self._make_result(easy_scores, hard_scores)
