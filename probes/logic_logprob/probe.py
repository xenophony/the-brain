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
    # --- Added items for convergence reliability ---
    # Easy
    {"prompt": "All humans are mortal. Socrates is a human. Is Socrates mortal?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "No vegetables are animals. A carrot is a vegetable. Is a carrot an animal?", "answer": "no", "difficulty": "easy"},
    {"prompt": "All even numbers are divisible by 2. Is 8 divisible by 2?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "If you study, you will pass. You studied. Will you pass?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "No children are allowed in the bar. Sam is a child. Is Sam allowed in the bar?", "answer": "no", "difficulty": "easy"},
    {"prompt": "All planets orbit a star. Earth is a planet. Does Earth orbit a star?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "No insects have exactly two legs. An ant is an insect. Does an ant have exactly two legs?", "answer": "no", "difficulty": "easy"},
    {"prompt": "All triangles have three sides. This shape has three sides. Could it be a triangle?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "If it is Tuesday, the shop is open. It is Tuesday. Is the shop open?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "All cars need fuel. This vehicle is a car. Does it need fuel?", "answer": "yes", "difficulty": "easy"},
    # Hard
    {"prompt": "Some A are B. Some B are C. Are some A necessarily C?", "answer": "no", "difficulty": "hard"},
    {"prompt": "If it rains, the ground is wet. The ground is wet. Did it necessarily rain?", "answer": "no", "difficulty": "hard"},
    {"prompt": "All doctors are educated. All lawyers are educated. Are all doctors lawyers?", "answer": "no", "difficulty": "hard"},
    {"prompt": "If P then Q. If Q then R. If P, does R follow?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "No A are B. Some C are B. Can any C be A?", "answer": "no", "difficulty": "hard"},
    {"prompt": "All X are Y. No Y are Z. If something is X, can it be Z?", "answer": "no", "difficulty": "hard"},
    {"prompt": "Some athletes are tall. All basketball players are athletes. Are all basketball players tall?", "answer": "no", "difficulty": "hard"},
    {"prompt": "If you speed, you get a ticket. You did not speed. Can you still get a ticket?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "All metals conduct electricity. Copper conducts electricity. Is copper necessarily a metal based only on this?", "answer": "no", "difficulty": "hard"},
    {"prompt": "If not P, then Q. P is true. Must Q be false?", "answer": "no", "difficulty": "hard"},
    {"prompt": "All squares have four sides. All rectangles have four sides. Are all squares rectangles?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "Some birds can swim. Some birds can fly. Can some birds both swim and fly?", "answer": "yes", "difficulty": "hard"},
]


@register_probe
class LogicLogprobProbe(BaseLogprobProbe):
    name = "logic_logprob"
    description = "Syllogistic reasoning via logprobs — prefrontal circuits"
    ITEMS = ITEMS
    CHOICES = ["yes", "no"]
