"""
Psycholinguistic unknowable probe — genuine model uncertainty.

Questions the model genuinely cannot answer. Forces a yes/no
commitment on unknowable facts. The psych profile should show
high hedging/epistemic_uncertain regardless of the answer chosen.

Control items are easy knowable facts for comparison.
The delta between unknowable and knowable psych profiles
validates whether the signal captures genuine uncertainty.

Maps to: uncertainty calibration / epistemic circuits.
"""

from probes.registry import BaseLogprobProbe, register_probe

ITEMS = [
    # Unknowable — model must guess, should show uncertainty
    {"prompt": "Is the 47th digit of pi a 5?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Did it rain in central Tokyo on March 3rd, 1847?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Was Einstein's childhood bicycle red?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Is the 10000th prime number odd?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "Did Shakespeare eat fish on his 30th birthday?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Was the temperature in Rome exactly 22 degrees on June 15th, year 1200?", "answer": "no", "difficulty": "hard"},

    # Knowable controls — model should show confidence
    {"prompt": "Is water composed of hydrogen and oxygen?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "Is the Earth the third planet from the Sun?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "Is Mount Everest the tallest mountain above sea level?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "Is the chemical symbol for sodium Na?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "Is the Great Wall of China located in Japan?", "answer": "no", "difficulty": "easy"},
    {"prompt": "Is the Amazon River in Europe?", "answer": "no", "difficulty": "easy"},
]


@register_probe
class PsychUnknowableLogprobProbe(BaseLogprobProbe):
    name = "psych_unknowable_logprob"
    description = "Genuine uncertainty signal — unknowable vs knowable"
    ITEMS = ITEMS
    CHOICES = ["yes", "no"]
