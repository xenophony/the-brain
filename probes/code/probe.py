"""
Code generation probe — function completion scored by unit tests.

Challenges are calibrated for 30B models: easy problems produce ceiling
effects and flat heatmaps. These require algorithmic insight, recursion,
and multi-step reasoning.

Output: short code snippet. Scored by executing test cases.
Maps to: cerebellum / motor (sequential procedure) circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

CHALLENGES = [
    {
        "prompt": (
            "Write a Python function `flatten(lst)` that deeply flattens a nested list. "
            "E.g. flatten([1,[2,[3,4],5],6]) returns [1,2,3,4,5,6]. "
            "Output only the function, no explanation.\n\n"
            "def flatten(lst):"
        ),
        "test_cases": [
            ("flatten([1,[2,[3,4],5],6])", [1, 2, 3, 4, 5, 6]),
            ("flatten([[[[1]]]])", [1]),
            ("flatten([])", []),
            ("flatten([1,2,3])", [1, 2, 3]),
            ("flatten([[1,2],[3,[4,[5]]]])", [1, 2, 3, 4, 5]),
        ],
        "extract_after": "def flatten(lst):",
    },
    {
        "prompt": (
            "Write a Python function `lcs(a, b)` that returns the length of the "
            "longest common subsequence of two strings. Use dynamic programming. "
            "Output only the function.\n\n"
            "def lcs(a, b):"
        ),
        "test_cases": [
            ("lcs('abcde', 'ace')", 3),
            ("lcs('abc', 'abc')", 3),
            ("lcs('abc', 'def')", 0),
            ("lcs('', 'abc')", 0),
            ("lcs('abcbdab', 'bdcab')", 4),
        ],
        "extract_after": "def lcs(a, b):",
    },
    {
        "prompt": (
            "Write a Python function `balanced(s)` that returns True if the string s "
            "has balanced parentheses, brackets, and braces. Supports ()[]{}. "
            "Output only the function.\n\n"
            "def balanced(s):"
        ),
        "test_cases": [
            ("balanced('([]){}')", True),
            ("balanced('([)]')", False),
            ("balanced('')", True),
            ("balanced('((()))')", True),
            ("balanced('{[}]')", False),
            ("balanced('({[]})')", True),
        ],
        "extract_after": "def balanced(s):",
    },
    {
        "prompt": (
            "Write a Python function `merge_intervals(intervals)` that merges "
            "overlapping intervals. Input: list of [start, end] pairs. "
            "Return sorted merged intervals. "
            "Output only the function.\n\n"
            "def merge_intervals(intervals):"
        ),
        "test_cases": [
            ("merge_intervals([[1,3],[2,6],[8,10],[15,18]])", [[1, 6], [8, 10], [15, 18]]),
            ("merge_intervals([[1,4],[4,5]])", [[1, 5]]),
            ("merge_intervals([])", []),
            ("merge_intervals([[1,2]])", [[1, 2]]),
            ("merge_intervals([[1,10],[2,3],[4,5]])", [[1, 10]]),
        ],
        "extract_after": "def merge_intervals(intervals):",
    },
    {
        "prompt": (
            "Write a Python function `spiral_order(matrix)` that returns elements "
            "of a 2D matrix in spiral order (clockwise from top-left). "
            "Output only the function.\n\n"
            "def spiral_order(matrix):"
        ),
        "test_cases": [
            ("spiral_order([[1,2,3],[4,5,6],[7,8,9]])", [1, 2, 3, 6, 9, 8, 7, 4, 5]),
            ("spiral_order([[1,2],[3,4]])", [1, 2, 4, 3]),
            ("spiral_order([[1]])", [1]),
            ("spiral_order([[1,2,3,4]])", [1, 2, 3, 4]),
            ("spiral_order([[1],[2],[3]])", [1, 2, 3]),
        ],
        "extract_after": "def spiral_order(matrix):",
    },
    {
        "prompt": (
            "Write a Python function `eval_rpn(tokens)` that evaluates a list of "
            "tokens in Reverse Polish Notation. Tokens are strings: numbers or "
            "operators (+, -, *, /). Division truncates toward zero. "
            "Output only the function.\n\n"
            "def eval_rpn(tokens):"
        ),
        "test_cases": [
            ("eval_rpn(['2','1','+','3','*'])", 9),
            ("eval_rpn(['4','13','5','/','+'])", 6),
            ("eval_rpn(['10','6','9','3','+','-11','*','/','*','17','+','5','+'])", 22),
            ("eval_rpn(['3','4','-'])", -1),
        ],
        "extract_after": "def eval_rpn(tokens):",
    },
    {
        "prompt": (
            "Write a Python function `permutations(nums)` that returns all "
            "permutations of a list of distinct integers. Return a list of lists. "
            "Output only the function.\n\n"
            "def permutations(nums):"
        ),
        "test_cases": [
            ("sorted(permutations([1,2,3]))", sorted([[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]])),
            ("permutations([1])", [[1]]),
            ("len(permutations([1,2,3,4]))", 24),
        ],
        "extract_after": "def permutations(nums):",
    },
]


def score_code(response: str, challenge: dict) -> float:
    """Execute generated code against test cases, return fraction passing."""
    func_header = challenge["extract_after"]
    body = response.strip()
    if body.startswith("def "):
        code = body
    else:
        code = func_header + "\n" + body

    # Take only the function body
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
                challenge["prompt"], max_new_tokens=150, temperature=0.0
            )
            score = score_code(response, challenge)
            scores.append(score)
        return sum(scores) / len(scores)
