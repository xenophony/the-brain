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

EASY_ITEMS = [
    {
        "prompt": (
            "Write a Python function `is_even(n)` that returns True if n is even, False otherwise. "
            "Output only the function, no explanation.\n\n"
            "def is_even(n):"
        ),
        "test_cases": [
            ("is_even(4)", True),
            ("is_even(3)", False),
            ("is_even(0)", True),
            ("is_even(-2)", True),
        ],
        "extract_after": "def is_even(n):",
    },
    {
        "prompt": (
            "Write a Python function `sum_list(lst)` that returns the sum of all numbers in a list. "
            "Output only the function, no explanation.\n\n"
            "def sum_list(lst):"
        ),
        "test_cases": [
            ("sum_list([1,2,3])", 6),
            ("sum_list([])", 0),
            ("sum_list([10])", 10),
            ("sum_list([-1,1])", 0),
        ],
        "extract_after": "def sum_list(lst):",
    },
    {
        "prompt": (
            "Write a Python function `first_element(lst)` that returns the first element of a list, "
            "or None if the list is empty. Output only the function, no explanation.\n\n"
            "def first_element(lst):"
        ),
        "test_cases": [
            ("first_element([1,2,3])", 1),
            ("first_element([])", None),
            ("first_element(['a'])", 'a'),
        ],
        "extract_after": "def first_element(lst):",
    },
    {
        "prompt": (
            "Write a Python function `reverse_string(s)` that returns the reversed string. "
            "Output only the function, no explanation.\n\n"
            "def reverse_string(s):"
        ),
        "test_cases": [
            ("reverse_string('hello')", 'olleh'),
            ("reverse_string('')", ''),
            ("reverse_string('a')", 'a'),
            ("reverse_string('ab')", 'ba'),
        ],
        "extract_after": "def reverse_string(s):",
    },
    {
        "prompt": (
            "Write a Python function `count_vowels(s)` that returns the number of vowels (a,e,i,o,u) "
            "in the string s (case-insensitive). Output only the function, no explanation.\n\n"
            "def count_vowels(s):"
        ),
        "test_cases": [
            ("count_vowels('hello')", 2),
            ("count_vowels('AEIOU')", 5),
            ("count_vowels('xyz')", 0),
            ("count_vowels('')", 0),
        ],
        "extract_after": "def count_vowels(s):",
    },
    {
        "prompt": (
            "Write a Python function `fizzbuzz_single(n)` that returns 'FizzBuzz' if n is divisible "
            "by both 3 and 5, 'Fizz' if divisible by 3, 'Buzz' if divisible by 5, "
            "or str(n) otherwise. Output only the function, no explanation.\n\n"
            "def fizzbuzz_single(n):"
        ),
        "test_cases": [
            ("fizzbuzz_single(15)", 'FizzBuzz'),
            ("fizzbuzz_single(3)", 'Fizz'),
            ("fizzbuzz_single(5)", 'Buzz'),
            ("fizzbuzz_single(7)", '7'),
        ],
        "extract_after": "def fizzbuzz_single(n):",
    },
    {
        "prompt": (
            "Write a Python function `abs_value(n)` that returns the absolute value of n "
            "without using the built-in abs(). Output only the function, no explanation.\n\n"
            "def abs_value(n):"
        ),
        "test_cases": [
            ("abs_value(5)", 5),
            ("abs_value(-5)", 5),
            ("abs_value(0)", 0),
        ],
        "extract_after": "def abs_value(n):",
    },
    {
        "prompt": (
            "Write a Python function `is_palindrome(s)` that returns True if the string s "
            "is a palindrome (case-insensitive, ignoring spaces). Output only the function.\n\n"
            "def is_palindrome(s):"
        ),
        "test_cases": [
            ("is_palindrome('racecar')", True),
            ("is_palindrome('hello')", False),
            ("is_palindrome('A man a plan a canal Panama'.replace(' ','').lower())", True),
            ("is_palindrome('')", True),
        ],
        "extract_after": "def is_palindrome(s):",
    },
]

HARD_ITEMS = [
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
    {
        "prompt": (
            "Write a Python function `binary_search(arr, target)` that returns the "
            "index of target in a sorted array arr, or -1 if not found. "
            "Output only the function.\n\n"
            "def binary_search(arr, target):"
        ),
        "test_cases": [
            ("binary_search([1,2,3,4,5], 3)", 2),
            ("binary_search([1,2,3,4,5], 6)", -1),
            ("binary_search([], 1)", -1),
            ("binary_search([1], 1)", 0),
            ("binary_search([1,3,5,7,9], 7)", 3),
        ],
        "extract_after": "def binary_search(arr, target):",
    },
]

# Legacy alias for backward compatibility
CHALLENGES = EASY_ITEMS + HARD_ITEMS


def score_code(response: str, challenge: dict) -> float:
    """Execute generated code against test cases, return fraction passing."""
    func_header = challenge["extract_after"]
    func_name = func_header.split("(")[0].replace("def ", "").strip()

    body = response.strip()

    # 1. Try to extract from ```python ... ``` code fence
    fence_match = re.search(r'```(?:python)?\s*\n(.*?)```', body, re.DOTALL)
    if fence_match:
        body = fence_match.group(1).strip()

    # 2. Find the function definition line
    lines = body.split("\n")
    func_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(f"def {func_name}("):
            func_start = i
            break

    if func_start == -1:
        # No function found — try prepending the header
        if not body.startswith("def "):
            body = func_header + "\n" + body
        code = body
    else:
        # Extract from function start to end of function body
        func_lines = [lines[func_start]]
        for line in lines[func_start + 1:]:
            if line.strip() and not line[0].isspace() and not line.strip().startswith("#"):
                break
            func_lines.append(line)
        code = "\n".join(func_lines)

    # Execute against test cases
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

    def run(self, model) -> dict:
        easy_scores = []
        hard_scores = []
        item_results = [] if self.log_responses else None

        for difficulty, items in [("easy", EASY_ITEMS), ("hard", HARD_ITEMS)]:
            scores = easy_scores if difficulty == "easy" else hard_scores
            for challenge in items:
                response = model.generate_short(
                    challenge["prompt"], max_new_tokens=400, temperature=0.0
                )
                score = score_code(response, challenge)
                scores.append(score)
                if item_results is not None:
                    item_results.append({
                        "difficulty": difficulty,
                        "func": challenge["extract_after"],
                        "response": response[:500],
                        "score": score,
                    })

        return self._make_result(easy_scores, hard_scores, item_results)
