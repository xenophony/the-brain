"""
Judgement logprob probe — answer correctness evaluation via logprobs.

Logprob variant of the judgement probe. Zero decode steps.
"""

from probes.registry import BaseLogprobProbe, register_probe
from probes.judgement.probe import EASY_ITEMS, HARD_ITEMS, _INSTRUCTION

CHOICES = ["correct", "incorrect"]

ITEMS = [
    {
        "prompt": (
            f"Question: {item['question']}\n"
            f"Answer given: {item['answer']}\n"
            f"{_INSTRUCTION}"
        ),
        "answer": "correct" if item["is_correct"] else "incorrect",
        "difficulty": "easy",
    }
    for item in EASY_ITEMS
] + [
    {
        "prompt": (
            f"Question: {item['question']}\n"
            f"Answer given: {item['answer']}\n"
            f"{_INSTRUCTION}"
        ),
        "answer": "correct" if item["is_correct"] else "incorrect",
        "difficulty": "hard",
    }
    for item in HARD_ITEMS
]


@register_probe
class JudgementLogprobProbe(BaseLogprobProbe):
    name = "judgement_logprob"
    description = "Answer correctness evaluation via logprobs — self-monitoring circuits"
    ITEMS = ITEMS
    CHOICES = CHOICES
