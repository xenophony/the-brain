"""
Implication logprob probe — syllogistic reasoning via logprobs.

Logprob variant of the implication probe. Zero decode steps.
"""

from probes.registry import BaseLogprobProbe, register_probe
from probes.implication.probe import EASY_ITEMS, HARD_ITEMS

CHOICES = ["valid", "invalid"]

ITEMS = [
    {
        "prompt": item["prompt"],
        "answer": item["answer"].lower(),
        "difficulty": "easy",
    }
    for item in EASY_ITEMS
] + [
    {
        "prompt": item["prompt"],
        "answer": item["answer"].lower(),
        "difficulty": "hard",
    }
    for item in HARD_ITEMS
]


@register_probe
class ImplicationLogprobProbe(BaseLogprobProbe):
    name = "implication_logprob"
    description = "Syllogistic reasoning via logprobs — deductive reasoning circuits"
    ITEMS = ITEMS
    CHOICES = CHOICES
