"""
Emotional intelligence probe — EQ-Bench style intensity estimation.

Presents social scenarios and asks how intensely a person would feel
a specific emotion on a 0-9 scale. Scored with partial credit for
near-misses using expected_digit_score() from BaseProbe.

Output: single digit 0-9.
Maps to: limbic system / emotional processing circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

EASY_ITEMS = [
    {
        "prompt": (
            "You discover that a trusted partner has been lying to you for months. "
            "How intensely would you feel betrayal on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 9,
    },
    {
        "prompt": (
            "After weeks of waiting, you learn that a biopsy result is benign. "
            "How intensely would you feel relief on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 9,
    },
    {
        "prompt": (
            "Your computer crashes and you lose an entire day's unsaved work. "
            "How intensely would you feel frustration on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "You finish a marathon for the first time after months of training. "
            "How intensely would you feel pride on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "Someone cuts in front of you in a long queue and smirks. "
            "How intensely would you feel anger on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "A stranger helps you change a flat tire in the rain without asking for anything. "
            "How intensely would you feel gratitude on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "You accidentally send a private complaint about your boss to your boss. "
            "How intensely would you feel shame on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "Your child wins a school science fair you helped them prepare for. "
            "How intensely would you feel joy on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "You receive news that a close family member has been in a serious car accident. "
            "How intensely would you feel fear on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 9,
    },
    {
        "prompt": (
            "You open your mailbox and find a handwritten thank-you letter from a student you mentored years ago. "
            "How intensely would you feel warmth on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "Your dog greets you at the door after a long day at work, tail wagging. "
            "How intensely would you feel happiness on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "You are fired from your job without warning on a Monday morning. "
            "How intensely would you feel shock on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 9,
    },
    {
        "prompt": (
            "Your best friend throws you a surprise birthday party with all your loved ones. "
            "How intensely would you feel delight on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "A coworker publicly mocks your accent during a team meeting. "
            "How intensely would you feel humiliation on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "You watch a documentary about children suffering from famine. "
            "How intensely would you feel sadness on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "You finally pay off your student loans after ten years of payments. "
            "How intensely would you feel relief on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 9,
    },
    {
        "prompt": (
            "Your neighbor's loud music wakes you up at 3 AM for the third night in a row. "
            "How intensely would you feel irritation on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "You find out your sibling has been secretly helping your elderly parents with bills. "
            "How intensely would you feel admiration on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "You witness a stranger yelling at a small child in a parking lot. "
            "How intensely would you feel outrage on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "You get accepted into your dream graduate program after two previous rejections. "
            "How intensely would you feel elation on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 9,
    },
    {
        "prompt": (
            "A friend cancels plans with you last minute for the fifth time. "
            "How intensely would you feel disappointment on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "You accidentally break your grandmother's antique vase that she treasured. "
            "How intensely would you feel guilt on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
    {
        "prompt": (
            "You hold your newborn baby for the first time. "
            "How intensely would you feel love on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 9,
    },
    {
        "prompt": (
            "A spider crawls across your pillow as you are about to sleep. "
            "How intensely would you feel disgust on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "Your flight is delayed by 6 hours and you miss an important event. "
            "How intensely would you feel frustration on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 8,
    },
]

HARD_ITEMS = [
    {
        "prompt": (
            "A colleague takes credit for your work in a meeting. "
            "How intensely would you feel guilt on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 2,
    },
    {
        "prompt": (
            "Your best friend gets the promotion you were both competing for. "
            "How intensely would you feel jealousy on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 5,
    },
    {
        "prompt": (
            "You spend a quiet Sunday afternoon reading in a sunlit garden. "
            "How intensely would you feel contentment on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "You receive an unexpected gift from a friend you haven't spoken to in years. "
            "How intensely would you feel surprise on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "You are waiting for important medical test results that will arrive tomorrow. "
            "How intensely would you feel anxiety on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "You win an award but know a colleague deserved it more. "
            "How intensely would you feel ambivalence on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 6,
    },
    {
        "prompt": (
            "Your ex-partner who hurt you badly asks you for help during a crisis. "
            "How intensely would you feel conflicted on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "You get a standing ovation after a presentation you thought went poorly. "
            "How intensely would you feel disbelief on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 6,
    },
    {
        "prompt": (
            "You see an old couple holding hands on a park bench. "
            "How intensely would you feel nostalgia on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 4,
    },
    {
        "prompt": (
            "A friend achieves something you have been failing at for years. "
            "How intensely would you feel envy on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 5,
    },
    {
        "prompt": (
            "You are praised publicly for a project, but you know a teammate did most of the work. "
            "How intensely would you feel discomfort on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 6,
    },
    {
        "prompt": (
            "You complete a routine errand like buying groceries on a normal day. "
            "How intensely would you feel excitement on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 1,
    },
    {
        "prompt": (
            "Your manager gives you neutral feedback that is neither positive nor negative. "
            "How intensely would you feel anxiety on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 3,
    },
    {
        "prompt": (
            "You learn that someone you dislike has suffered a serious misfortune. "
            "How intensely would you feel satisfaction on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 3,
    },
    {
        "prompt": (
            "You are about to give a speech to a large audience but feel well-prepared. "
            "How intensely would you feel nervousness on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 4,
    },
    {
        "prompt": (
            "A distant relative you barely know passes away at an old age. "
            "How intensely would you feel grief on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 3,
    },
    {
        "prompt": (
            "You discover that a charity you donated to generously has been misusing funds. "
            "How intensely would you feel betrayal on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 6,
    },
    {
        "prompt": (
            "You overhear coworkers laughing but you are unsure if they are laughing about you. "
            "How intensely would you feel insecurity on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 4,
    },
    {
        "prompt": (
            "You receive a promotion but it means relocating away from your family. "
            "How intensely would you feel conflicted on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 7,
    },
    {
        "prompt": (
            "A former bully from school sincerely apologizes to you twenty years later. "
            "How intensely would you feel moved on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 5,
    },
    {
        "prompt": (
            "You watch a sunset alone after completing a difficult personal goal. "
            "How intensely would you feel serenity on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 6,
    },
    {
        "prompt": (
            "Your teenager says they hate you during an argument about curfew. "
            "How intensely would you feel hurt on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 5,
    },
    {
        "prompt": (
            "You realize you forgot to call your mother on her birthday. "
            "How intensely would you feel regret on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 6,
    },
    {
        "prompt": (
            "A stranger smiles at you warmly on an otherwise gloomy day. "
            "How intensely would you feel uplift on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 4,
    },
    {
        "prompt": (
            "You find out your partner planned a vacation without telling you, and you dislike the destination. "
            "How intensely would you feel irritation on a scale of 0 (none) to 9 (extreme)? "
            "Answer with only a digit."
        ),
        "expected": 5,
    },
]

# Legacy alias
SCENARIOS = EASY_ITEMS + HARD_ITEMS


def _extract_eq_digit(response: str) -> int | None:
    """Extract emotion intensity digit from response.

    Uses the LAST single digit to avoid catching digits from
    echoed scenario text (e.g. "I would rate this a 7").
    """
    r = response.strip()
    # Try last single digit (word-bounded) in response
    digits = re.findall(r'\b(\d)\b', r)
    if digits:
        return int(digits[-1])
    # Try any ASCII digit (last one) — skip unicode superscripts etc.
    for ch in reversed(r):
        if ch in "0123456789":
            return int(ch)
    return None


def _eq_digit_score(response: str, expected: int) -> float:
    """Score an EQ response with partial credit for near-misses."""
    got = _extract_eq_digit(response)
    if got is None:
        return 0.0
    if got == expected:
        return 1.0
    diff = abs(got - expected)
    if diff == 1:
        return 0.5
    if diff == 2:
        return 0.25
    return 0.0


@register_probe
class EQProbe(BaseProbe):
    name = "eq"
    description = "Emotional intensity estimation — limbic system circuits"

    def run(self, model) -> dict:
        easy_scores, easy_results = self._run_items(
            model, self._limit(EASY_ITEMS),
            prompt_fn=lambda item: item["prompt"],
            score_fn=lambda resp, item: _eq_digit_score(resp, item["expected"]),
            max_new_tokens=5, difficulty="easy")

        hard_scores, hard_results = self._run_items(
            model, self._limit(HARD_ITEMS),
            prompt_fn=lambda item: item["prompt"],
            score_fn=lambda resp, item: _eq_digit_score(resp, item["expected"]),
            max_new_tokens=5, difficulty="hard")

        item_results = (easy_results + hard_results) if self.log_responses else None
        return self._make_result(easy_scores, hard_scores, item_results)
