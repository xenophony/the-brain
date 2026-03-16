"""
Routing logprob probe — task domain classification via logprobs.

Logprob variant of the routing probe. Zero decode steps.
For items with multiple accepted answers, uses the first accepted domain.
"""

from probes.registry import BaseLogprobProbe, register_probe
from probes.routing.probe import EASY_ITEMS, HARD_ITEMS, _INSTRUCTION

CHOICES = ["math", "language", "spatial", "factual", "emotional"]

ITEMS = [
    {
        "prompt": f"Task: {item['task']}\n{_INSTRUCTION}",
        "answer": item["accept"][0].lower(),
        "difficulty": "easy",
    }
    for item in EASY_ITEMS
] + [
    {
        "prompt": f"Task: {item['task']}\n{_INSTRUCTION}",
        "answer": item["accept"][0].lower(),
        "difficulty": "hard",
    }
    for item in HARD_ITEMS
]


@register_probe
class RoutingLogprobProbe(BaseLogprobProbe):
    name = "routing_logprob"
    description = "Task domain classification via logprobs — executive control circuits"
    ITEMS = ITEMS
    CHOICES = CHOICES
