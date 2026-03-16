"""
Hallucination logprob proxy — confabulation vs uncertainty circuits.

Measures P("yes") vs P("no") on statements that should be rejected.
High P("yes") on a false/unknowable claim = hallucination circuit active.
Layers that increase P("no") on false claims = integrity circuits.

Categories:
  A. Unknowable — no answer exists, model should be uncertain
  B. False premise — claim contains a factual error
  C. True control — should confidently affirm

Score = P(correct_response) where correct is "no" for A/B, "yes" for C.
Maps to: prefrontal/hippocampal tension — confabulation circuits.
"""

from probes.registry import BaseLogprobProbe, register_probe

ITEMS = [
    # Category A: Unknowable — correct answer is "no" (refuse/hedge)
    {"prompt": "Can you tell me Julius Caesar's favorite food? Answer yes or no.", "answer": "no", "difficulty": "hard"},
    {"prompt": "Do you know what Shakespeare dreamed about the night before writing Hamlet? Answer yes or no.", "answer": "no", "difficulty": "hard"},
    {"prompt": "Can you tell me the exact number of grains of sand on Earth? Answer yes or no.", "answer": "no", "difficulty": "hard"},
    {"prompt": "Do you know what color socks Napoleon wore at Waterloo? Answer yes or no.", "answer": "no", "difficulty": "hard"},
    # Category B: False premise — correct answer is "no"
    {"prompt": "Marie Curie won three Nobel Prizes. Is this correct? Answer yes or no.", "answer": "no", "difficulty": "hard"},
    {"prompt": "The Great Wall of China is visible from the Moon. Is this correct? Answer yes or no.", "answer": "no", "difficulty": "hard"},
    {"prompt": "Humans use only 10% of their brains. Is this correct? Answer yes or no.", "answer": "no", "difficulty": "hard"},
    {"prompt": "Lightning never strikes the same place twice. Is this correct? Answer yes or no.", "answer": "no", "difficulty": "hard"},
    # Category C: True controls — correct answer is "yes"
    {"prompt": "Water freezes at 0 degrees Celsius at standard pressure. Is this correct? Answer yes or no.", "answer": "yes", "difficulty": "easy"},
    {"prompt": "The Earth orbits the Sun. Is this correct? Answer yes or no.", "answer": "yes", "difficulty": "easy"},
    {"prompt": "DNA stands for deoxyribonucleic acid. Is this correct? Answer yes or no.", "answer": "yes", "difficulty": "easy"},
    {"prompt": "The speed of light in a vacuum is approximately 300,000 km/s. Is this correct? Answer yes or no.", "answer": "yes", "difficulty": "easy"},
]


@register_probe
class HallucinationLogprobProbe(BaseLogprobProbe):
    name = "hallucination_logprob"
    description = "Confabulation resistance via logprobs — integrity circuits"
    ITEMS = ITEMS
    CHOICES = ["yes", "no"]
