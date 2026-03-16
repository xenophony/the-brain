"""
Logical deduction logprob probe — syllogistic reasoning circuits.

Measures P(correct) for syllogisms and logical inference via logprobs.
Zero decode steps — 1 forward pass per question.
Calibrated at 75% accuracy on Qwen3-32B (no_think).
p_correct tracking may reveal sub-threshold effects even at high argmax accuracy.

Maps to: prefrontal reasoning circuits.
"""

from probes.registry import BaseLogprobProbe, register_probe

ITEMS = [
    # Easy (straightforward valid syllogisms)
    {"prompt": "All dogs are animals. Rex is a dog. Is Rex an animal?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "All cats have tails. Whiskers is a cat. Does Whiskers have a tail?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "No fish can fly. A salmon is a fish. Can a salmon fly?", "answer": "no", "difficulty": "easy"},
    {"prompt": "All squares are rectangles. Is every square a rectangle?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "If it is raining, the ground is wet. It is raining. Is the ground wet?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "No mammals lay eggs. A cow is a mammal. Does a cow lay eggs?", "answer": "no", "difficulty": "easy"},
    # Hard (invalid syllogisms, affirming the consequent, undistributed middle)
    {"prompt": "All roses are flowers. Some flowers fade quickly. Do all roses fade quickly?", "answer": "no", "difficulty": "hard"},
    {"prompt": "All birds have feathers. A penguin has feathers. Is a penguin definitely a bird based only on this?", "answer": "no", "difficulty": "hard"},
    {"prompt": "No reptiles are mammals. Some pets are mammals. Are no pets reptiles?", "answer": "no", "difficulty": "hard"},
    {"prompt": "If it snows, schools close. Schools are closed. Did it necessarily snow?", "answer": "no", "difficulty": "hard"},
    {"prompt": "All A are B. All B are C. No C are D. Can any A be D?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Some dogs are large. Some large things are dangerous. Are some dogs necessarily dangerous?", "answer": "no", "difficulty": "hard"},
]


@register_probe
class LogicLogprobProbe(BaseLogprobProbe):
    name = "logic_logprob"
    description = "Syllogistic reasoning via logprobs — prefrontal circuits"
    ITEMS = ITEMS
    CHOICES = ["yes", "no"]
