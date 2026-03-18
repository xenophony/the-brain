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
    # --- Added items for convergence reliability ---
    # Easy
    {"prompt": "Is this correct or incorrect: 'The chemical formula for water is H2O.'", "answer": "correct", "difficulty": "easy"},
    {"prompt": "Is this correct or incorrect: 'There are 7 continents on Earth.'", "answer": "correct", "difficulty": "easy"},
    {"prompt": "Is this correct or incorrect: 'Mars is the closest planet to the Sun.'", "answer": "incorrect", "difficulty": "easy"},
    {"prompt": "Is this correct or incorrect: 'A triangle has four sides.'", "answer": "incorrect", "difficulty": "easy"},
    {"prompt": "Is this correct or incorrect: 'The boiling point of water is 100 degrees Celsius at sea level.'", "answer": "correct", "difficulty": "easy"},
    {"prompt": "Is this correct or incorrect: 'Tokyo is the capital of China.'", "answer": "incorrect", "difficulty": "easy"},
    {"prompt": "Is this correct or incorrect: 'There are 60 seconds in a minute.'", "answer": "correct", "difficulty": "easy"},
    {"prompt": "Is this correct or incorrect: 'Spiders have six legs.'", "answer": "incorrect", "difficulty": "easy"},
    {"prompt": "Is this correct or incorrect: 'Oxygen is necessary for combustion.'", "answer": "correct", "difficulty": "easy"},
    {"prompt": "Is this correct or incorrect: 'The Moon is larger than the Earth.'", "answer": "incorrect", "difficulty": "easy"},
    # Hard
    {"prompt": "Is this correct or incorrect: 'The human body has 204 bones.'", "answer": "incorrect", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'Mount Everest is in Nepal.'", "answer": "correct", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'There are 118 elements in the periodic table.'", "answer": "correct", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'The Amazon is the longest river in the world.'", "answer": "incorrect", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'Einstein published the theory of general relativity in 1905.'", "answer": "incorrect", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'Electrons are smaller than protons.'", "answer": "correct", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'Venus has a thicker atmosphere than Earth.'", "answer": "correct", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'The Pythagorean theorem works for all triangles.'", "answer": "incorrect", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'Light bends when it passes through a prism.'", "answer": "correct", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'Pluto was reclassified as a dwarf planet in 2003.'", "answer": "incorrect", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'DNA has a double helix structure.'", "answer": "correct", "difficulty": "hard"},
    {"prompt": "Is this correct or incorrect: 'The French Revolution began in 1789.'", "answer": "correct", "difficulty": "hard"},
]


@register_probe
class ErrorLogprobProbe(BaseLogprobProbe):
    name = "error_logprob"
    description = "Error/fact verification via logprobs — metacognition circuits"
    ITEMS = ITEMS
    CHOICES = ["correct", "incorrect"]
