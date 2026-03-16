"""
Error detection logprob probe — self-monitoring / metacognition circuits.

Measures P("correct") vs P("incorrect") via logprobs.
Generation-based scoring failed (model returns "True"/"False", not labels).
Logprob scoring bypasses this.

Maps to: metacognition / self-monitoring circuits.
"""

from probes.registry import BaseLogprobProbe, register_probe

ITEMS = [
    # Easy (obvious errors)
    {"prompt": "Is this correct or incorrect: '2 + 2 = 5'", "answer": "incorrect", "difficulty": "easy"},
    {"prompt": "Is this correct or incorrect: 'The Earth orbits the Sun.'", "answer": "correct", "difficulty": "easy"},
    {"prompt": "Is this correct or incorrect: 'There are 24 hours in a day.'", "answer": "correct", "difficulty": "easy"},
    {"prompt": "Is this correct or incorrect: 'The capital of Australia is Sydney.'", "answer": "incorrect", "difficulty": "easy"},
    {"prompt": "Is this correct or incorrect: 'Humans have 206 bones.'", "answer": "correct", "difficulty": "easy"},
    {"prompt": "Is this correct or incorrect: 'The Sun revolves around the Earth.'", "answer": "incorrect", "difficulty": "easy"},
    # Hard (subtle errors, near-misses, commonly confused)
    {"prompt": "Is this correct or incorrect: 'The speed of light is approximately 3 x 10^9 m/s.'", "answer": "incorrect", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'There are 52 weeks in a year.'", "answer": "correct", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'Abraham Lincoln was the 15th president.'", "answer": "incorrect", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'Pi rounded to two decimal places is 3.15.'", "answer": "incorrect", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'The Great Wall of China is visible from space.'", "answer": "incorrect", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'Sound travels faster than light.'", "answer": "incorrect", "difficulty": "hard"},
]


@register_probe
class ErrorLogprobProbe(BaseLogprobProbe):
    name = "error_logprob"
    description = "Error/fact verification via logprobs — metacognition circuits"
    ITEMS = ITEMS
    CHOICES = ["correct", "incorrect"]
