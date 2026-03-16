"""
Negation logprob probe — negation understanding via logprobs.

Logprob variant of the negation probe. Zero decode steps.
Each pair (positive + negated) becomes two separate items.
"""

from probes.registry import BaseLogprobProbe, register_probe
from probes.negation.probe import EASY_PAIRS, HARD_PAIRS

CHOICES = ["true", "false"]

ITEMS = []
for pair in EASY_PAIRS:
    ITEMS.append({
        "prompt": pair["positive"]["prompt"],
        "answer": pair["positive"]["answer"].lower(),
        "difficulty": "easy",
    })
    ITEMS.append({
        "prompt": pair["negated"]["prompt"],
        "answer": pair["negated"]["answer"].lower(),
        "difficulty": "easy",
    })
for pair in HARD_PAIRS:
    ITEMS.append({
        "prompt": pair["positive"]["prompt"],
        "answer": pair["positive"]["answer"].lower(),
        "difficulty": "hard",
    })
    ITEMS.append({
        "prompt": pair["negated"]["prompt"],
        "answer": pair["negated"]["answer"].lower(),
        "difficulty": "hard",
    })


@register_probe
class NegationLogprobProbe(BaseLogprobProbe):
    name = "negation_logprob"
    description = "Negation understanding via logprobs — language processing circuits"
    ITEMS = ITEMS
    CHOICES = CHOICES
