"""
Causal reasoning logprob probe — world model / physics circuits.

Measures P(correct) for cause-effect questions via logprobs.
Zero decode steps — 1 forward pass per question.
Calibrated at 50% accuracy on Qwen3-32B (no_think).

Maps to: world model / causal reasoning circuits.
"""

from probes.registry import BaseLogprobProbe, register_probe

ITEMS = [
    # Easy (obvious cause-effect)
    {"prompt": "You drop a glass on concrete. Will it likely break?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "You leave ice cream in the sun. Will it melt?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "You water a plant regularly. Will it likely grow?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "You unplug a lamp. Will it stay on?", "answer": "no", "difficulty": "easy"},
    {"prompt": "A ball is thrown upward. Will it eventually come back down?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "You put a metal spoon in a microwave. Is this safe?", "answer": "no", "difficulty": "easy"},
    # Hard (counterintuitive, common misconceptions)
    {"prompt": "A heavier object falls faster than a lighter one in a vacuum. True or false?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Adding salt to water raises its boiling point. True or false?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "A person standing still in a moving train is moving relative to the ground. True or false?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "Hot water freezes faster than cold water under certain conditions. True or false?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "If you spin a coin that lands heads 10 times in a row, is the next flip more likely heads?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Lightning never strikes the same place twice. True or false?", "answer": "no", "difficulty": "hard"},
]


@register_probe
class CausalLogprobProbe(BaseLogprobProbe):
    name = "causal_logprob"
    description = "Causal/physical reasoning via logprobs — world model circuits"
    ITEMS = ITEMS
    CHOICES = ["yes", "no"]
