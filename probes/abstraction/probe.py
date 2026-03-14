"""
Abstraction probe -- representational flexibility and level shifting.

15 items across 3 types:
  Type A: Concrete to abstract (5) -- name the abstract category
  Type B: Abstract to concrete (5) -- name a specific example
  Type C: Level identification (5) -- which is more abstract: X or Y?

Scoring:
  Type A: exact or acceptable match
  Type B: any accepted concrete example
  Type C: exact match (x or y)

Output: single word or short phrase.
Maps to: association cortex / representational flexibility circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

QUESTIONS = [
    # --- Type A: Concrete to abstract (5) ---
    {
        "type": "A",
        "prompt": (
            "What general category do dog, cat, and hamster belong to? "
            "Answer with one word."
        ),
        "accept": ["pets", "animals", "mammals"],
    },
    {
        "type": "A",
        "prompt": (
            "What general category do red, blue, and green belong to? "
            "Answer with one word."
        ),
        "accept": ["colors", "colours"],
    },
    {
        "type": "A",
        "prompt": (
            "What general category do addition, subtraction, and multiplication belong to? "
            "Answer with one or two words."
        ),
        "accept": ["operations", "arithmetic", "math operations", "mathematical operations"],
    },
    {
        "type": "A",
        "prompt": (
            "What general category do happiness, sadness, and anger belong to? "
            "Answer with one word."
        ),
        "accept": ["emotions", "feelings"],
    },
    {
        "type": "A",
        "prompt": (
            "What general category do oak, maple, and pine belong to? "
            "Answer with one word."
        ),
        "accept": ["trees", "plants"],
    },
    # --- Type B: Abstract to concrete (5) ---
    {
        "type": "B",
        "prompt": (
            "Name a specific example of a mammal. Answer with one word."
        ),
        "accept": [
            "dog", "cat", "horse", "cow", "elephant", "whale", "dolphin",
            "mouse", "rat", "bear", "lion", "tiger", "wolf", "deer", "rabbit",
            "monkey", "ape", "gorilla", "bat", "fox", "pig", "sheep", "goat",
            "human", "hamster",
        ],
    },
    {
        "type": "B",
        "prompt": (
            "Name a specific example of a geometric shape. Answer with one word."
        ),
        "accept": [
            "circle", "square", "triangle", "rectangle", "pentagon", "hexagon",
            "oval", "cube", "sphere", "cylinder", "diamond", "rhombus",
            "trapezoid", "parallelogram", "octagon",
        ],
    },
    {
        "type": "B",
        "prompt": (
            "Name a specific example of a musical instrument. Answer with one word."
        ),
        "accept": [
            "piano", "guitar", "violin", "drum", "flute", "trumpet", "cello",
            "harp", "saxophone", "clarinet", "trombone", "bass", "oboe",
            "harmonica", "ukulele", "banjo", "tuba", "organ",
        ],
    },
    {
        "type": "B",
        "prompt": (
            "Name a specific example of a programming language. Answer with one word."
        ),
        "accept": [
            "python", "java", "javascript", "c", "c++", "rust", "go",
            "ruby", "swift", "kotlin", "typescript", "php", "perl",
            "haskell", "scala", "r", "matlab", "lua", "julia",
        ],
    },
    {
        "type": "B",
        "prompt": (
            "Name a specific example of a chemical element. Answer with one word."
        ),
        "accept": [
            "hydrogen", "helium", "oxygen", "carbon", "nitrogen", "iron",
            "gold", "silver", "copper", "zinc", "sodium", "potassium",
            "calcium", "lead", "mercury", "uranium", "platinum", "neon",
            "argon", "chlorine", "sulfur", "phosphorus", "lithium",
        ],
    },
    # --- Type C: Level identification (5) ---
    {
        "type": "C",
        "prompt": (
            "Which is more abstract: 'vehicle' or 'red Toyota Camry'? "
            "Answer only vehicle or red Toyota Camry."
        ),
        "answer": "vehicle",
    },
    {
        "type": "C",
        "prompt": (
            "Which is more abstract: 'justice' or 'a court ruling'? "
            "Answer only justice or a court ruling."
        ),
        "answer": "justice",
    },
    {
        "type": "C",
        "prompt": (
            "Which is more abstract: 'my pet dog Rex' or 'animal'? "
            "Answer only my pet dog Rex or animal."
        ),
        "answer": "animal",
    },
    {
        "type": "C",
        "prompt": (
            "Which is more abstract: 'number' or '42'? "
            "Answer only number or 42."
        ),
        "answer": "number",
    },
    {
        "type": "C",
        "prompt": (
            "Which is more abstract: 'communication' or 'a phone call'? "
            "Answer only communication or a phone call."
        ),
        "answer": "communication",
    },
]


def score_abstraction(response: str, question: dict) -> float:
    """Score an abstraction response."""
    r = response.strip().lower()

    if question["type"] in ("A", "B"):
        # Check if any accepted answer appears in the response
        for acc in question["accept"]:
            if acc.lower() in r:
                return 1.0
        return 0.0

    if question["type"] == "C":
        expected = question["answer"].lower()
        if expected in r:
            return 1.0
        return 0.0

    return 0.0


@register_probe
class AbstractionProbe(BaseProbe):
    name = "abstraction"
    description = "Abstraction levels -- association cortex / representational flexibility"

    def run(self, model) -> float:
        scores = []
        for q in QUESTIONS:
            response = model.generate_short(q["prompt"], max_new_tokens=15, temperature=0.0)
            score = score_abstraction(response, q)
            scores.append(score)
        return sum(scores) / len(scores)
