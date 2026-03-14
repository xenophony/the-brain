"""
Code generation probe — simple function completion scored by unit tests.

Output: short code snippet. Scored by executing test cases.
Maps to: cerebellum / motor (sequential procedure) circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

CHALLENGES = [
    {
        "prompt": "Write a Python function `double(x)` that returns x*2. Output only the function, no explanation.\n\ndef double(x):",
        "test_cases": [
            ("double(3)", 6),
            ("double(0)", 0),
            ("double(-5)", -10),
        ],
        "extract_after": "def double(x):",
    },
    {
        "prompt": "Write a Python function `is_even(n)` that returns True if n is even, False otherwise. Output only the function.\n\ndef is_even(n):",
        "test_cases": [
            ("is_even(4)", True),
            ("is_even(7)", False),
            ("is_even(0)", True),
        ],
        "extract_after": "def is_even(n):",
    },
    {
        "prompt": "Write a Python function `max_of_three(a,b,c)` that returns the largest of three numbers. Output only the function.\n\ndef max_of_three(a,b,c):",
        "test_cases": [
            ("max_of_three(1,2,3)", 3),
            ("max_of_three(5,5,5)", 5),
            ("max_of_three(9,3,6)", 9),
        ],
        "extract_after": "def max_of_three(a,b,c):",
    },
    {
        "prompt": "Write a Python function `factorial(n)` that returns n! for non-negative integers. Output only the function.\n\ndef factorial(n):",
        "test_cases": [
            ("factorial(0)", 1),
            ("factorial(5)", 120),
            ("factorial(1)", 1),
        ],
        "extract_after": "def factorial(n):",
    },
    {
        "prompt": "Write a Python function `reverse_string(s)` that returns the reversed string. Output only the function.\n\ndef reverse_string(s):",
        "test_cases": [
            ("reverse_string('hello')", "olleh"),
            ("reverse_string('')", ""),
            ("reverse_string('a')", "a"),
        ],
        "extract_after": "def reverse_string(s):",
    },
    {
        "prompt": "Write a Python function `count_vowels(s)` that counts vowels (aeiou, case insensitive) in a string. Output only the function.\n\ndef count_vowels(s):",
        "test_cases": [
            ("count_vowels('hello')", 2),
            ("count_vowels('AEIOU')", 5),
            ("count_vowels('xyz')", 0),
        ],
        "extract_after": "def count_vowels(s):",
    },
]


def score_code(response: str, challenge: dict) -> float:
    """Execute generated code against test cases, return fraction passing."""
    # Reconstruct full function
    func_header = challenge["extract_after"]
    body = response.strip()
    # If model repeated the header, strip it
    if body.startswith("def "):
        code = body
    else:
        code = func_header + "\n" + body

    # Clean up — take only until the next def or class or empty lines
    lines = code.split("\n")
    clean_lines = [lines[0]]
    for line in lines[1:]:
        if line.strip() and not line[0].isspace() and not line.strip().startswith("#"):
            break
        clean_lines.append(line)
    code = "\n".join(clean_lines)

    passed = 0
    total = len(challenge["test_cases"])

    for call, expected in challenge["test_cases"]:
        try:
            namespace = {}
            exec(code, namespace)
            result = eval(call, namespace)
            if result == expected:
                passed += 1
        except Exception:
            pass

    return passed / total if total > 0 else 0.0


@register_probe
class CodeProbe(BaseProbe):
    name = "code"
    description = "Code generation with unit test scoring — cerebellum/motor circuits"

    def run(self, model) -> float:
        scores = []
        for challenge in CHALLENGES:
            response = model.generate_short(
                challenge["prompt"], max_new_tokens=80, temperature=0.0
            )
            score = score_code(response, challenge)
            scores.append(score)
        return sum(scores) / len(scores)
