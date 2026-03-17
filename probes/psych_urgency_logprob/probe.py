"""
Psycholinguistic urgency probe — performance under urgency framing.

Same factual questions asked with and without urgency context.
Hypothesis: urgency framing activates performance circuits from
training data where humans think more clearly under pressure.

If urgency changes the psych profile (lower hedging, higher confidence)
without changing the answer, those are the performance circuits.
Layer configs that produce the same psych signature as urgency framing
— without urgency in the prompt — are amplifying those circuits directly.

Maps to: performance / attention / urgency circuits.
"""

from probes.registry import BaseLogprobProbe, register_probe

ITEMS = [
    # Neutral framing
    {"prompt": "Is water H2O?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "Is the speed of light faster than the speed of sound?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "Is 144 the square of 12?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "Is iron magnetic?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "Is Venus closer to the Sun than Earth?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "Is the pH of pure water 7?", "answer": "yes", "difficulty": "easy"},

    # Urgent framing — same facts, urgency context
    {"prompt": "URGENT — a patient's treatment depends on this: Is water H2O?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "CRITICAL SAFETY CHECK — a spacecraft navigation depends on this: Is the speed of light faster than the speed of sound?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "EMERGENCY — structural integrity calculation requires confirmation: Is 144 the square of 12?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "URGENT — industrial safety depends on this answer: Is iron magnetic?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "CRITICAL — mission trajectory depends on this: Is Venus closer to the Sun than Earth?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "EMERGENCY — chemical process safety check: Is the pH of pure water 7?", "answer": "yes", "difficulty": "hard"},
]


@register_probe
class PsychUrgencyLogprobProbe(BaseLogprobProbe):
    name = "psych_urgency_logprob"
    description = "Urgency framing effect on psych profile — performance circuits"
    ITEMS = ITEMS
    CHOICES = ["yes", "no"]
