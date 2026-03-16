"""
Language logprob probe — grammaticality judgment via logprobs.

Logprob variant of the language probe. Zero decode steps.
"""

from probes.registry import BaseLogprobProbe, register_probe
from probes.language.probe import EASY_ITEMS, HARD_ITEMS, PROMPT_TEMPLATE

CHOICES = ["grammatical", "ungrammatical"]

ITEMS = [
    {
        "prompt": PROMPT_TEMPLATE.format(sentence=item["sentence"]),
        "answer": item["label"].lower(),
        "difficulty": "easy",
    }
    for item in EASY_ITEMS
] + [
    {
        "prompt": PROMPT_TEMPLATE.format(sentence=item["sentence"]),
        "answer": item["label"].lower(),
        "difficulty": "hard",
    }
    for item in HARD_ITEMS
]


@register_probe
class LanguageLogprobProbe(BaseLogprobProbe):
    name = "language_logprob"
    description = "Grammaticality judgment via logprobs — Broca/Wernicke circuits"
    ITEMS = ITEMS
    CHOICES = CHOICES
