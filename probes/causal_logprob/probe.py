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
    # --- Added items for convergence reliability ---
    # Easy
    {"prompt": "You put a wet towel in the freezer. Will it freeze?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "You leave a bicycle in the rain for months. Will it rust?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "You hold a magnet near iron filings. Will they be attracted?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "A candle is lit in a sealed jar. Will it eventually go out?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "You mix baking soda and vinegar. Will there be a chemical reaction?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "A car with no fuel can still drive normally. True or false?", "answer": "no", "difficulty": "easy"},
    {"prompt": "You touch a hot stove. Will it burn your hand?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "An egg dropped from a table onto a tile floor will likely survive intact. True or false?", "answer": "no", "difficulty": "easy"},
    {"prompt": "Pouring water on a grease fire will safely extinguish it. True or false?", "answer": "no", "difficulty": "easy"},
    {"prompt": "Leaving bread out for a week will cause it to grow mold. True or false?", "answer": "yes", "difficulty": "easy"},
    # Hard
    {"prompt": "Glass is a liquid that flows very slowly over centuries. True or false?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Cutting an onion underwater prevents you from crying. True or false?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "A compass needle points to geographic north. True or false?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Diamonds can be burned if heated enough in oxygen. True or false?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "A penny dropped from the Empire State Building can kill a person. True or false?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Antibiotics are effective against viral infections. True or false?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Bananas are naturally radioactive. True or false?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "The vacuum of space is completely empty with nothing in it. True or false?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Cooking food always destroys all its nutritional value. True or false?", "answer": "no", "difficulty": "hard"},
    {"prompt": "If you mix all colors of light together, you get white light. True or false?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "A helium balloon released outdoors will rise forever. True or false?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Fresh water and salt water freeze at the same temperature. True or false?", "answer": "no", "difficulty": "hard"},
]


@register_probe
class CausalLogprobProbe(BaseLogprobProbe):
    name = "causal_logprob"
    description = "Causal/physical reasoning via logprobs — world model circuits"
    ITEMS = ITEMS
    CHOICES = ["yes", "no"]
