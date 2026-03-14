"""
Instruction following probe — comply with multiple explicit constraints.

Each scenario gives 3-4 constraints. The model must produce a short response
that satisfies all of them. Each constraint has a programmatic checker.

Output: short text (< 20 tokens).
Scoring: fraction of constraints satisfied.
Maps to: PFC / working memory circuits.
"""

import re
from probes.registry import BaseProbe, register_probe


def _has_uppercase(response: str) -> bool:
    """Check that response is entirely uppercase letters (ignoring non-alpha)."""
    alpha = [ch for ch in response if ch.isalpha()]
    return len(alpha) > 0 and all(ch.isupper() for ch in alpha)


def _has_number(response: str) -> bool:
    """Check that response contains at least one digit."""
    return any(ch.isdigit() for ch in response)


def _ends_with_exclamation(response: str) -> bool:
    return response.rstrip().endswith("!")


def _ends_with_period(response: str) -> bool:
    return response.rstrip().endswith(".")


def _word_count(response: str, n: int) -> bool:
    words = response.strip().split()
    return len(words) == n


def _starts_with_letter(response: str, letter: str) -> bool:
    r = response.strip()
    return len(r) > 0 and r[0].upper() == letter.upper()


def _no_vowels(response: str) -> bool:
    alpha = [ch for ch in response.lower() if ch.isalpha()]
    return len(alpha) > 0 and not any(ch in "aeiou" for ch in alpha)


def _contains_color(response: str) -> bool:
    colors = {"red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink", "brown"}
    words = set(response.lower().split())
    return bool(words & colors)


def _contains_animal(response: str) -> bool:
    animals = {
        "cat", "dog", "fox", "bear", "wolf", "deer", "bird", "fish", "lion",
        "tiger", "hawk", "owl", "cow", "pig", "rat", "bat", "ant", "bee",
    }
    words = set(response.lower().split())
    return bool(words & animals)


def _is_palindrome_word(response: str) -> bool:
    """Check if response contains at least one palindrome word (3+ letters)."""
    for word in response.strip().split():
        w = re.sub(r'[^a-zA-Z]', '', word).lower()
        if len(w) >= 3 and w == w[::-1]:
            return True
    return False


def _all_words_same_first_letter(response: str) -> bool:
    """Check alliteration: all alpha words start with the same letter."""
    words = [w for w in response.strip().split() if w[0].isalpha()] if response.strip() else []
    if len(words) < 2:
        return False
    first_letters = {w[0].lower() for w in words}
    return len(first_letters) == 1


def _all_lowercase(response: str) -> bool:
    """Check that response is entirely lowercase letters (ignoring non-alpha)."""
    alpha = [ch for ch in response if ch.isalpha()]
    return len(alpha) > 0 and all(ch.islower() for ch in alpha)


def _no_spaces(response: str) -> bool:
    """Check no spaces in the response."""
    return " " not in response.strip()


def _contains_question_mark(response: str) -> bool:
    return "?" in response


SCENARIOS = [
    {
        "prompt": (
            "Write a short response (under 10 words) that satisfies ALL these constraints:\n"
            "1. All letters must be uppercase\n"
            "2. Must contain at least one number\n"
            "3. Must end with an exclamation mark\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("all_uppercase", _has_uppercase),
            ("has_number", _has_number),
            ("ends_exclamation", _ends_with_exclamation),
        ],
    },
    {
        "prompt": (
            "Write a response of EXACTLY 3 words that satisfies ALL these constraints:\n"
            "1. Exactly 3 words\n"
            "2. All letters must be uppercase\n"
            "3. Must contain at least one number\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("exactly_3_words", lambda r: _word_count(r, 3)),
            ("all_uppercase", _has_uppercase),
            ("has_number", _has_number),
        ],
    },
    {
        "prompt": (
            "Write a short response (under 10 words) that satisfies ALL these constraints:\n"
            "1. Starts with the letter Z\n"
            "2. Contains at least one number\n"
            "3. All letters must be uppercase\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("starts_with_z", lambda r: _starts_with_letter(r, "Z")),
            ("has_number", _has_number),
            ("all_uppercase", _has_uppercase),
        ],
    },
    {
        "prompt": (
            "Write a short response (under 10 words) that satisfies ALL these constraints:\n"
            "1. Contains no vowels (a, e, i, o, u) in any word\n"
            "2. Contains at least one number\n"
            "3. Ends with an exclamation mark\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("no_vowels", _no_vowels),
            ("has_number", _has_number),
            ("ends_exclamation", _ends_with_exclamation),
        ],
    },
    {
        "prompt": (
            "Write a short response (under 10 words) that satisfies ALL these constraints:\n"
            "1. Contains a palindrome word (at least 3 letters)\n"
            "2. Contains at least one number\n"
            "3. Ends with an exclamation mark\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("palindrome", _is_palindrome_word),
            ("has_number", _has_number),
            ("ends_exclamation", _ends_with_exclamation),
        ],
    },
    {
        "prompt": (
            "Write a short response (under 10 words) that satisfies ALL these constraints:\n"
            "1. Contains the name of a color\n"
            "2. Contains the name of an animal\n"
            "3. Contains at least one number\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("has_color", _contains_color),
            ("has_animal", _contains_animal),
            ("has_number", _has_number),
        ],
    },
    {
        "prompt": (
            "Write a short response (under 10 words) that satisfies ALL these constraints:\n"
            "1. Uses alliteration (all words start with the same letter)\n"
            "2. At least 3 words\n"
            "3. All letters must be uppercase\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("alliteration", _all_words_same_first_letter),
            ("at_least_3_words", lambda r: len(r.strip().split()) >= 3),
            ("all_uppercase", _has_uppercase),
        ],
    },
    {
        "prompt": (
            "Write a short response (under 10 words) that satisfies ALL these constraints:\n"
            "1. Ends with a period\n"
            "2. Contains at least one number\n"
            "3. Exactly 4 words\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("ends_period", _ends_with_period),
            ("has_number", _has_number),
            ("exactly_4_words", lambda r: _word_count(r, 4)),
        ],
    },
    {
        "prompt": (
            "Write a short response (under 10 words) that satisfies ALL these constraints:\n"
            "1. Starts with the letter A\n"
            "2. Ends with an exclamation mark\n"
            "3. All letters must be uppercase\n"
            "4. Contains at least one number\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("starts_with_a", lambda r: _starts_with_letter(r, "A")),
            ("ends_exclamation", _ends_with_exclamation),
            ("all_uppercase", _has_uppercase),
            ("has_number", _has_number),
        ],
    },
    {
        "prompt": (
            "Write a short response (under 10 words) that satisfies ALL these constraints:\n"
            "1. Contains exactly 2 words\n"
            "2. Both words must start with the same letter\n"
            "3. Contains at least one number\n"
            "4. Ends with a period\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("exactly_2_words", lambda r: _word_count(r, 2)),
            ("same_first_letter", lambda r: _all_words_same_first_letter(r) if len(r.strip().split()) >= 2 else False),
            ("has_number", _has_number),
            ("ends_period", _ends_with_period),
        ],
    },
    # --- Conflicting-constraint scenarios (fight natural language defaults) ---
    {
        "prompt": (
            "Write a short response that satisfies ALL these constraints:\n"
            "1. All letters must be lowercase (no capitals at all)\n"
            "2. Must end with a question mark\n"
            "3. Exactly 3 words\n"
            "Follow these rules exactly. Do NOT capitalize the first letter."
        ),
        "checkers": [
            ("all_lowercase", _all_lowercase),
            ("ends_question_mark", _contains_question_mark),
            ("exactly_3_words", lambda r: _word_count(r, 3)),
        ],
    },
    {
        "prompt": (
            "Write a short response that satisfies ALL these constraints:\n"
            "1. No spaces anywhere in the response\n"
            "2. Contains at least one number\n"
            "3. Contains at least one letter\n"
            "4. Ends with a period\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("no_spaces", _no_spaces),
            ("has_number", _has_number),
            ("has_letter", lambda r: any(ch.isalpha() for ch in r)),
            ("ends_period", _ends_with_period),
        ],
    },
    {
        "prompt": (
            "Write a short response that satisfies ALL these constraints:\n"
            "1. All letters lowercase (no capitals)\n"
            "2. Exactly 1 word\n"
            "3. Must contain at least one number embedded in the word\n"
            "4. Must end with an exclamation mark\n"
            "Follow these rules exactly. Do NOT capitalize anything."
        ),
        "checkers": [
            ("all_lowercase", _all_lowercase),
            ("exactly_1_word", lambda r: _word_count(r, 1)),
            ("has_number", _has_number),
            ("ends_exclamation", _ends_with_exclamation),
        ],
    },
]


@register_probe
class InstructionProbe(BaseProbe):
    name = "instruction"
    description = "Multi-constraint instruction following — working memory circuits"

    def run(self, model) -> float:
        scores = []
        for scenario in SCENARIOS:
            response = model.generate_short(
                scenario["prompt"], max_new_tokens=20, temperature=0.0
            )
            # Score = fraction of constraints satisfied
            n_constraints = len(scenario["checkers"])
            satisfied = 0
            for _name, checker in scenario["checkers"]:
                try:
                    if checker(response):
                        satisfied += 1
                except Exception:
                    pass
            scores.append(satisfied / n_constraints)
        return sum(scores) / len(scores)
