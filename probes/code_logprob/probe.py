"""
Code completion logprob probe — programming circuit mapping.

Presents incomplete code and measures P(correct_completion_token).
Tests different coding skills: recursion, loops, data structures,
string ops, logic, math operations.

Hypothesis: coding is compound — different skills use different
circuits. Recursion may use reasoning (layers 2-3), string ops
may use translation (layers 29-34), logic may use mid-layers.

Maps to: code generation circuits (unknown — this probe discovers them).
"""

from probes.registry import BaseLogprobProbe, register_probe

ITEMS = [
    # Recursion / base cases
    {"prompt": "Complete the code: def fibonacci(n):\n    if n <= 1:\n        return\nThe next token is:", "answer": "n", "difficulty": "easy"},
    {"prompt": "Complete the code: def factorial(n):\n    if n == 0:\n        return\nThe next token is:", "answer": "1", "difficulty": "easy"},

    # Loop patterns
    {"prompt": "Complete the code: for i in range(10):\n    print\nThe next token is:", "answer": "(", "difficulty": "easy"},
    {"prompt": "Complete the code: while True:\n    data = input()\n    if data == 'quit':\n        \nThe next token is:", "answer": "break", "difficulty": "easy"},

    # Data structures
    {"prompt": "Complete the code: my_dict = {}\nmy_dict['key'] =\nThe next token is:", "answer": "'", "difficulty": "hard"},
    {"prompt": "Complete the code: result = [x for x in range(10) if x %\nThe next token is:", "answer": "2", "difficulty": "hard"},

    # String operations
    {"prompt": "Complete the code: name = 'hello world'\nupper_name = name.\nThe next token is:", "answer": "upper", "difficulty": "easy"},
    {"prompt": "Complete the code: words = sentence.\nThe next token after 'sentence.' for splitting is:", "answer": "split", "difficulty": "easy"},

    # Math operations
    {"prompt": "Complete the code: import math\nresult = math.\nThe next token for square root is:", "answer": "sqrt", "difficulty": "hard"},
    {"prompt": "Complete the code: total = sum(numbers) / \nThe next token for average is:", "answer": "len", "difficulty": "hard"},

    # Logic / conditionals
    {"prompt": "Complete the code: if x > 0 and y > 0:\n    return 'both positive'\nelif x > 0:\n    return 'only x'\n\nThe next keyword is:", "answer": "else", "difficulty": "easy"},
    {"prompt": "Complete the code: try:\n    result = dangerous_function()\nexcept\nThe next token is:", "answer": ":", "difficulty": "hard"},
]


@register_probe
class CodeLogprobProbe(BaseLogprobProbe):
    name = "code_logprob"
    description = "Code completion via logprobs — programming circuit mapping"
    ITEMS = ITEMS
    CHOICES = ["n", "1", "(", "break", "'", "2", "upper", "split", "sqrt", "len", "else", ":"]
