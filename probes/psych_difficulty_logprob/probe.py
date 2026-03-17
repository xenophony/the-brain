"""
Psycholinguistic difficulty gradient probe.

Same yes/no format at three difficulty levels. If psych_confidence
decreases and psych_hedging increases with difficulty, the signal
is real. If the gradient doesn't appear, psych capture is noise.

All items are "Is X correct?" with known answers.

Maps to: calibration / metacognition circuits.
"""

from probes.registry import BaseLogprobProbe, register_probe

ITEMS = [
    # Easy — model should be very confident
    {"prompt": "Is 2 + 2 equal to 4?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "Is the sky blue on a clear day?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "Is 10 greater than 3?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "Is ice frozen water?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "Is 100 divided by 10 equal to 5?", "answer": "no", "difficulty": "easy"},
    {"prompt": "Is the Sun a planet?", "answer": "no", "difficulty": "easy"},

    # Medium — requires some knowledge, model should be less certain
    {"prompt": "Is 17 a prime number?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "Is the atomic number of carbon 6?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "Is the square root of 169 equal to 13?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "Is the boiling point of ethanol 78 degrees Celsius?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "Is the atomic number of nitrogen 8?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Is the square root of 224 equal to 15?", "answer": "no", "difficulty": "hard"},
]


@register_probe
class PsychDifficultyLogprobProbe(BaseLogprobProbe):
    name = "psych_difficulty_logprob"
    description = "Difficulty gradient for psych signal validation"
    ITEMS = ITEMS
    CHOICES = ["yes", "no"]
