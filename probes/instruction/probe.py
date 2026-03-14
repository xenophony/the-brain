"""
Instruction following probe — comply with multiple explicit constraints.

Each scenario gives 1-4 constraints. The model must produce a short response
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


def _no_contractions(response: str) -> bool:
    """Check response contains no contractions."""
    contractions = ["don't", "can't", "won't", "isn't", "aren't", "doesn't",
                    "didn't", "hasn't", "haven't", "couldn't", "wouldn't",
                    "shouldn't", "it's", "i'm", "we're", "they're", "you're"]
    r = response.lower()
    return not any(c in r for c in contractions)


def _no_word_the(response: str) -> bool:
    """Check response doesn't contain the word 'the'."""
    words = response.lower().split()
    return "the" not in words


def _is_bullet_points(response: str) -> bool:
    """Check response is formatted as bullet points."""
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    return all(l.startswith(("-", "*", "\u2022")) for l in lines)


def _max_words(response: str, n: int) -> bool:
    """Check response has at most n words."""
    return len(response.strip().split()) <= n


EASY_ITEMS = [
    # Simple 3-constraint scenarios
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
    # Type B: single-check instruction persistence
    {
        "prompt": (
            "IMPORTANT RULE: You must respond ONLY with bullet points. "
            "No prose, no sentences, only bullet points starting with '-'. "
            "List 3 colors.\n"
            "Follow this rule exactly."
        ),
        "checkers": [
            ("is_bullet_points", _is_bullet_points),
        ],
    },
    {
        "prompt": (
            "CRITICAL INSTRUCTION: Your entire response must be in ALL CAPS. "
            "No lowercase letters at all. Tell me a fact about the moon.\n"
            "Follow this rule exactly."
        ),
        "checkers": [
            ("all_uppercase", _has_uppercase),
        ],
    },
    {
        "prompt": (
            "STRICT RULE: Respond with exactly 2 words. Nothing more, nothing less. "
            "What do you think about rain?\n"
            "Follow this rule exactly."
        ),
        "checkers": [
            ("exactly_2_words", lambda r: _word_count(r, 2)),
        ],
    },
    {
        "prompt": (
            "MANDATORY: End every sentence with an exclamation mark. No periods. "
            "Write two sentences about the sun.\n"
            "Follow this rule exactly."
        ),
        "checkers": [
            ("ends_exclamation", _ends_with_exclamation),
            ("no_periods", lambda r: "." not in r.strip()),
        ],
    },
    {
        "prompt": (
            "Respond in ALL CAPS. Your answer should be one sentence.\n"
            "What color is the sky?"
        ),
        "checkers": [
            ("all_uppercase", _has_uppercase),
        ],
    },
]

HARD_ITEMS = [
    # Conflicting-constraint scenarios
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
    # Type A: instruction vs model preference conflict
    {
        "prompt": (
            "Respond only in formal English. Do not use any contractions. "
            "Do not use the word 'the'. Write one sentence about weather.\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("no_contractions", _no_contractions),
            ("no_word_the", _no_word_the),
            ("has_letter", lambda r: any(ch.isalpha() for ch in r)),
        ],
    },
    {
        "prompt": (
            "Write exactly 5 words. Every word must start with a vowel (a, e, i, o, u). "
            "All letters must be lowercase.\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("exactly_5_words", lambda r: _word_count(r, 5)),
            ("all_start_vowel", lambda r: all(
                w[0].lower() in "aeiou"
                for w in r.strip().split()
                if w and w[0].isalpha()
            ) if r.strip() else False),
            ("all_lowercase", _all_lowercase),
        ],
    },
    {
        "prompt": (
            "Write a sentence where every word has exactly 4 letters. "
            "Use at least 3 words. No punctuation except a period at the end.\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("all_4_letter", lambda r: all(
                len(re.sub(r'[^a-zA-Z]', '', w)) == 4
                for w in r.rstrip(".").strip().split()
                if w
            ) if r.strip() else False),
            ("at_least_3_words", lambda r: len(r.strip().split()) >= 3),
            ("ends_period", _ends_with_period),
        ],
    },
    # Type C: nested instruction conflict
    {
        "prompt": (
            "Be extremely concise (max 5 words). Also be very specific and detailed. "
            "What is a computer?\n"
            "Balance both constraints."
        ),
        "checkers": [
            ("max_5_words", lambda r: _max_words(r, 5)),
            ("has_content", lambda r: len(r.strip()) > 5),
        ],
    },
    {
        "prompt": (
            "Write exactly 3 words. Each word must be longer than 6 letters. "
            "All letters lowercase.\n"
            "Follow these rules exactly."
        ),
        "checkers": [
            ("exactly_3_words", lambda r: _word_count(r, 3)),
            ("long_words", lambda r: all(
                len(re.sub(r'[^a-zA-Z]', '', w)) > 6
                for w in r.strip().split() if w
            ) if r.strip() else False),
            ("all_lowercase", _all_lowercase),
        ],
    },
]

# Legacy alias
SCENARIOS = EASY_ITEMS + HARD_ITEMS


@register_probe
class InstructionProbe(BaseProbe):
    name = "instruction"
    description = "Multi-constraint instruction following — working memory circuits"

    def run(self, model) -> dict:
        easy_scores = []
        for scenario in EASY_ITEMS:
            response = model.generate_short(
                scenario["prompt"], max_new_tokens=20, temperature=0.0
            )
            n_constraints = len(scenario["checkers"])
            satisfied = 0
            for _name, checker in scenario["checkers"]:
                try:
                    if checker(response):
                        satisfied += 1
                except Exception:
                    pass
            easy_scores.append(satisfied / n_constraints)

        hard_scores = []
        for scenario in HARD_ITEMS:
            response = model.generate_short(
                scenario["prompt"], max_new_tokens=20, temperature=0.0
            )
            n_constraints = len(scenario["checkers"])
            satisfied = 0
            for _name, checker in scenario["checkers"]:
                try:
                    if checker(response):
                        satisfied += 1
                except Exception:
                    pass
            hard_scores.append(satisfied / n_constraints)

        return self._make_result(easy_scores, hard_scores)
