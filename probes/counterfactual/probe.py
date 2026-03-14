"""
Counterfactual reasoning probe -- hypothetical scenario modeling.

15 scenarios across 3 types:
  Type A: Physical counterfactuals (faster/slower/same)
  Type B: Social counterfactuals (multiple choice A/B/C)
  Type C: Logical counterfactuals (multiple choice A/B/C/D)

Scoring: exact match per question, mean across 15.

Output: short answer (faster/slower/same or A/B/C/D).
Maps to: prefrontal cortex / hypothetical modeling circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

SCENARIOS = [
    # --- Type A: Physical counterfactuals (5) ---
    {
        "type": "A",
        "prompt": (
            "If gravity were twice as strong, would a dropped ball hit the ground "
            "faster, slower, or the same? Answer only faster, slower, or same."
        ),
        "answer": "faster",
    },
    {
        "type": "A",
        "prompt": (
            "If the Moon were twice as far from Earth, would ocean tides be "
            "stronger, weaker, or the same? Answer only stronger, weaker, or same."
        ),
        "answer": "weaker",
    },
    {
        "type": "A",
        "prompt": (
            "If air had zero viscosity, would a parachute fall "
            "faster, slower, or the same as without a parachute? "
            "Answer only faster, slower, or same."
        ),
        "answer": "same",
    },
    {
        "type": "A",
        "prompt": (
            "If Earth had no atmosphere, would surface temperature variation "
            "between day and night be larger, smaller, or the same? "
            "Answer only larger, smaller, or same."
        ),
        "answer": "larger",
    },
    {
        "type": "A",
        "prompt": (
            "If ice were denser than liquid water, would ice sink or float in water? "
            "Answer only sink or float."
        ),
        "answer": "sink",
    },
    # --- Type B: Social counterfactuals (5) ---
    {
        "type": "B",
        "prompt": (
            "If the printing press had never been invented, which would most likely be true?\n"
            "A) Literacy rates would be similar to today\n"
            "B) Literacy rates would be much lower\n"
            "C) Literacy rates would be higher\n"
            "Answer only A, B, or C."
        ),
        "answer": "B",
    },
    {
        "type": "B",
        "prompt": (
            "If humans had never developed agriculture, which would most likely be true?\n"
            "A) Cities would still have formed at the same rate\n"
            "B) Population density would be much lower\n"
            "C) Technology would have advanced faster\n"
            "Answer only A, B, or C."
        ),
        "answer": "B",
    },
    {
        "type": "B",
        "prompt": (
            "If electricity had never been harnessed, which would most likely be true?\n"
            "A) The industrial revolution would have been unaffected\n"
            "B) Communication would still rely on physical mail and signals\n"
            "C) Computers would have been invented anyway using other means\n"
            "Answer only A, B, or C."
        ),
        "answer": "B",
    },
    {
        "type": "B",
        "prompt": (
            "If antibiotics had never been discovered, which would most likely be true?\n"
            "A) Average human lifespan would be shorter\n"
            "B) Humans would have evolved natural immunity to all bacteria\n"
            "C) Surgery would remain just as safe\n"
            "Answer only A, B, or C."
        ),
        "answer": "A",
    },
    {
        "type": "B",
        "prompt": (
            "If the internet had never been created, which would most likely be true?\n"
            "A) Global trade would be at the same level\n"
            "B) Information would spread more slowly\n"
            "C) Social media would exist in the same form via other technology\n"
            "Answer only A, B, or C."
        ),
        "answer": "B",
    },
    # --- Type C: Logical counterfactuals (5) ---
    {
        "type": "C",
        "prompt": (
            "In a world where all cats can fly, and Whiskers is a cat, "
            "which must be true?\n"
            "A) Whiskers cannot fly\n"
            "B) Whiskers can fly\n"
            "C) Only some cats can fly\n"
            "D) Whiskers is not a cat\n"
            "Answer only A, B, C, or D."
        ),
        "answer": "B",
    },
    {
        "type": "C",
        "prompt": (
            "In a world where water flows uphill, which would be true?\n"
            "A) Rivers would flow from sea to mountains\n"
            "B) Rivers would flow from mountains to sea as normal\n"
            "C) Rivers would not exist\n"
            "D) Gravity would be unchanged\n"
            "Answer only A, B, C, or D."
        ),
        "answer": "A",
    },
    {
        "type": "C",
        "prompt": (
            "In a world where the number 3 does not exist, what comes after 2?\n"
            "A) 3\n"
            "B) 4\n"
            "C) Nothing\n"
            "D) 2\n"
            "Answer only A, B, C, or D."
        ),
        "answer": "B",
    },
    {
        "type": "C",
        "prompt": (
            "In a world where all metals are liquid at room temperature, "
            "which would be true?\n"
            "A) Buildings could still use steel beams\n"
            "B) Coins would not hold their shape\n"
            "C) Metal tools would work the same way\n"
            "D) Cars would be unchanged\n"
            "Answer only A, B, C, or D."
        ),
        "answer": "B",
    },
    {
        "type": "C",
        "prompt": (
            "In a world where humans do not need sleep, which would most likely be true?\n"
            "A) The concept of 'night' would not exist\n"
            "B) Productivity per person could increase\n"
            "C) Beds would never have been invented\n"
            "D) Humans would still sleep out of habit\n"
            "Answer only A, B, C, or D."
        ),
        "answer": "B",
    },
]


def score_counterfactual(response: str, scenario: dict) -> float:
    """Score a counterfactual response by exact match."""
    r = response.strip().lower()
    expected = scenario["answer"].lower()

    # For Type A: look for the keyword
    if scenario["type"] == "A":
        if expected in r:
            return 1.0
        return 0.0

    # For Types B and C: look for the letter
    # Extract letter answers from response
    match = re.search(r'\b([abcd])\b', r)
    if match and match.group(1).upper() == expected.upper():
        return 1.0

    # Also accept if the response starts with the letter
    if r.startswith(expected):
        return 1.0

    return 0.0


@register_probe
class CounterfactualProbe(BaseProbe):
    name = "counterfactual"
    description = "Counterfactual reasoning -- prefrontal cortex / hypothetical modeling"

    def run(self, model) -> float:
        scores = []
        for scenario in SCENARIOS:
            response = model.generate_short(
                scenario["prompt"], max_new_tokens=15, temperature=0.0
            )
            score = score_counterfactual(response, scenario)
            scores.append(score)
        return sum(scores) / len(scores)
