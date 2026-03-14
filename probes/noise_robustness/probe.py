"""
Noise robustness probe -- processing stability under input variation.

10 base questions, each asked 4 ways:
  Version A: clean canonical phrasing
  Version B: different wording, same meaning
  Version C: added irrelevant context
  Version D: conversational/casual framing

Scoring per base question: all 4 correct=1.0, 3/4=0.75, 2/4=0.5, 1/4=0.25, 0=0.0.
Mean across 10 base questions.

Output: short answer per version.
Maps to: general processing stability / noise filtering circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

BASE_QUESTIONS = [
    {
        "id": 0,
        "versions": {
            "A": "What is the capital of France? Answer with one word.",
            "B": "Name the capital city of France. Answer with one word.",
            "C": "I was eating pizza yesterday and wondering, what is the capital of France? Answer with one word.",
            "D": "hey so like whats the capital of france? one word pls",
        },
        "accept": ["paris"],
    },
    {
        "id": 1,
        "versions": {
            "A": "What is 7 * 8? Answer with only the number.",
            "B": "Calculate the product of seven and eight. Answer with only the number.",
            "C": "My friend asked me a math question while we were hiking: what is 7 * 8? Answer with only the number.",
            "D": "yo whats 7 times 8? just the number",
        },
        "accept": ["56"],
    },
    {
        "id": 2,
        "versions": {
            "A": "What element has the chemical symbol 'O'? Answer with one word.",
            "B": "Which chemical element is represented by the symbol O? Answer with one word.",
            "C": "While reading a chemistry textbook about molecular bonds, I noticed the symbol O. What element is it? Answer with one word.",
            "D": "whats the element for symbol O? one word",
        },
        "accept": ["oxygen"],
    },
    {
        "id": 3,
        "versions": {
            "A": "How many sides does a hexagon have? Answer with only the number.",
            "B": "State the number of sides in a hexagon. Answer with only the number.",
            "C": "I was looking at a stop sign, which is an octagon, but that made me wonder: how many sides does a hexagon have? Answer with only the number.",
            "D": "hexagon - how many sides? number only",
        },
        "accept": ["6", "six"],
    },
    {
        "id": 4,
        "versions": {
            "A": "What is the largest planet in our solar system? Answer with one word.",
            "B": "Name the biggest planet orbiting our Sun. Answer with one word.",
            "C": "After watching a documentary about Mars rovers and asteroid mining, I wondered: what is the largest planet in our solar system? Answer with one word.",
            "D": "biggest planet in the solar system? one word",
        },
        "accept": ["jupiter"],
    },
    {
        "id": 5,
        "versions": {
            "A": "What is the freezing point of water in Celsius? Answer with only the number.",
            "B": "At what temperature in Celsius does water freeze? Answer with only the number.",
            "C": "My car's thermometer showed 5 degrees today, and I thought about ice formation. What is the freezing point of water in Celsius? Answer with only the number.",
            "D": "freezing point of water celsius? just the number",
        },
        "accept": ["0", "zero"],
    },
    {
        "id": 6,
        "versions": {
            "A": "What color do you get by mixing red and blue? Answer with one word.",
            "B": "Name the color produced by combining red and blue. Answer with one word.",
            "C": "While painting a landscape with yellow and green trees, I ran out of purple. What color do you get by mixing red and blue? Answer with one word.",
            "D": "red + blue = what color? one word",
        },
        "accept": ["purple", "violet"],
    },
    {
        "id": 7,
        "versions": {
            "A": "How many continents are there on Earth? Answer with only the number.",
            "B": "State the total number of continents on our planet. Answer with only the number.",
            "C": "I just returned from a trip to Asia, my third continent visited. How many continents are there on Earth in total? Answer with only the number.",
            "D": "how many continents on earth? number only",
        },
        "accept": ["7", "seven"],
    },
    {
        "id": 8,
        "versions": {
            "A": "What is the chemical formula for table salt? Answer in chemical notation.",
            "B": "Write the chemical formula of sodium chloride. Answer in chemical notation.",
            "C": "While seasoning my steak with various spices and herbs, I grabbed the salt shaker. What is the chemical formula for table salt? Answer in chemical notation.",
            "D": "chemical formula for salt? just the formula",
        },
        "accept": ["nacl"],
    },
    {
        "id": 9,
        "versions": {
            "A": "What is the square root of 144? Answer with only the number.",
            "B": "Calculate the square root of 144. Answer with only the number.",
            "C": "I was helping my nephew with his algebra homework about exponents. What is the square root of 144? Answer with only the number.",
            "D": "sqrt of 144? just the number",
        },
        "accept": ["12", "twelve"],
    },
]


def score_single_version(response: str, accept: list[str]) -> bool:
    """Check if a response is correct for a given question."""
    r = response.strip().lower()
    for acc in accept:
        if acc.lower() in r:
            return True
    return False


def score_noise_robustness_base(results: list[bool]) -> float:
    """Score a base question given correctness of its 4 versions."""
    correct_count = sum(1 for r in results if r)
    return correct_count / len(results)


@register_probe
class NoiseRobustnessProbe(BaseProbe):
    name = "noise_robustness"
    description = "Noise robustness -- general processing stability circuits"

    def run(self, model) -> float:
        scores = []
        for bq in BASE_QUESTIONS:
            results = []
            for version_key in ["A", "B", "C", "D"]:
                prompt = bq["versions"][version_key]
                response = model.generate_short(
                    prompt, max_new_tokens=15, temperature=0.0
                )
                correct = score_single_version(response, bq["accept"])
                results.append(correct)
            scores.append(score_noise_robustness_base(results))
        return sum(scores) / len(scores)
