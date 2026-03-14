"""
Factual recall probe — obscure but verifiable facts.

Tests the model's ability to recall precise, non-trivial factual knowledge.
Answers are numbers or single unambiguous words.

Output: number or single word.
Scoring: exact match = 1.0, off-by-one (numbers) = 0.5, else 0.0.
Maps to: hippocampus / factual memory circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

QUESTIONS = [
    {
        "prompt": "How many bones are in an adult human body? Answer with only the number.",
        "answer": "206",
        "type": "number",
    },
    {
        "prompt": "What is the melting point of tungsten in degrees Celsius? Answer with only the number.",
        "answer": "3422",
        "type": "number",
    },
    {
        "prompt": "What is the speed of light in km/s (rounded to nearest integer)? Answer with only the number.",
        "answer": "299792",
        "type": "number",
    },
    {
        "prompt": "What is the atomic number of gold? Answer with only the number.",
        "answer": "79",
        "type": "number",
    },
    {
        "prompt": "What is the tallest mountain in the solar system? Answer with only the name (one word).",
        "answer": "olympus",
        "type": "word",
        "alternates": ["olympus mons"],
    },
    {
        "prompt": "What is the deepest ocean trench on Earth? Answer with only the name (one word).",
        "answer": "mariana",
        "type": "word",
        "alternates": ["mariana trench", "marianas"],
    },
    {
        "prompt": "What is the half-life of carbon-14 in years? Answer with only the number.",
        "answer": "5730",
        "type": "number",
    },
    {
        "prompt": "How many chromosomes do humans have (total, not pairs)? Answer with only the number.",
        "answer": "46",
        "type": "number",
    },
    {
        "prompt": "What is the boiling point of nitrogen in degrees Celsius? Answer with only the number.",
        "answer": "-196",
        "type": "number",
    },
    {
        "prompt": "What is the rest mass energy of an electron in keV? Answer with only the number.",
        "answer": "511",
        "type": "number",
    },
    {
        "prompt": "What is the average distance from the Earth to the Moon in kilometers? Answer with only the number.",
        "answer": "384400",
        "type": "number",
    },
    {
        "prompt": "What is Planck's constant in units of 10^-34 J*s? Answer with only the number (e.g. 6.626).",
        "answer": "6.626",
        "type": "number",
    },
]


def score_factual(response: str, question: dict) -> float:
    """Score a factual response. Exact match = 1.0, near-miss = 0.5, else 0.0."""
    response = response.strip().lower()

    if question["type"] == "word":
        expected = question["answer"].lower()
        alternates = [a.lower() for a in question.get("alternates", [])]
        # Check response contains the expected word
        if expected in response:
            return 1.0
        for alt in alternates:
            if alt in response:
                return 1.0
        return 0.0

    # Number type
    expected_str = question["answer"]
    # Extract number from response
    match = re.search(r'-?[\d]+\.?[\d]*', response)
    if not match:
        return 0.0

    try:
        got = float(match.group())
        expected = float(expected_str)
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

    def run(self, model) -> float:
        scores = []
        for q in QUESTIONS:
            response = model.generate_short(q["prompt"], max_new_tokens=15, temperature=0.0)
            score = score_factual(response, q)
            scores.append(score)
        return sum(scores) / len(scores)
