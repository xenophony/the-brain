"""
Temporal ordering logprob probe — sequence/historical knowledge circuits.

Measures P(correct) for "which came first" questions via logprobs.
Zero decode steps — 1 forward pass per question.
Calibrated at 66.7% accuracy on Qwen3-32B (no_think).

Maps to: temporal/sequential reasoning circuits.
"""

from probes.registry import BaseLogprobProbe, register_probe

ITEMS = [
    # Easy (well-known historical order)
    {"prompt": "Which came first: (A) World War I or (B) World War II?", "answer": "a", "difficulty": "easy"},
    {"prompt": "Which came first: (A) invention of the telephone or (B) invention of the internet?", "answer": "a", "difficulty": "easy"},
    {"prompt": "Which came first: (A) Ancient Egypt or (B) the Roman Empire?", "answer": "a", "difficulty": "easy"},
    {"prompt": "Which came first: (A) the Renaissance or (B) the Industrial Revolution?", "answer": "a", "difficulty": "easy"},
    {"prompt": "Which came first: (A) the Moon landing or (B) the fall of the Berlin Wall?", "answer": "a", "difficulty": "easy"},
    {"prompt": "Which came first: (A) the printing press or (B) the steam engine?", "answer": "a", "difficulty": "easy"},
    # Hard (counterintuitive, surprising temporal facts)
    {"prompt": "Which came first: (A) Oxford University founded or (B) the Aztec Empire founded?", "answer": "a", "difficulty": "hard"},
    {"prompt": "Which came first: (A) the fax machine patent or (B) the telephone patent?", "answer": "a", "difficulty": "hard"},
    {"prompt": "Which came first: (A) Harvard University or (B) calculus invented?", "answer": "a", "difficulty": "hard"},
    {"prompt": "Which came first: (A) Nintendo founded or (B) the Ottoman Empire fell?", "answer": "a", "difficulty": "hard"},
    {"prompt": "Which came first: (A) the samurai abolished in Japan or (B) the London Underground opened?", "answer": "b", "difficulty": "hard"},
    {"prompt": "Which came first: (A) Cleopatra or (B) the construction of the Great Pyramid?", "answer": "b", "difficulty": "hard"},
]


@register_probe
class TemporalLogprobProbe(BaseLogprobProbe):
    name = "temporal_logprob"
    description = "Temporal ordering via logprobs — sequence/planning circuits"
    ITEMS = ITEMS
    CHOICES = ["a", "b"]
