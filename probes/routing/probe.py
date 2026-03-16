"""
Routing probe — task domain classification.

Tests the "metacognitive router" circuit: given a problem, classify
which cognitive domain it belongs to. Multi-label where legitimate
(e.g. a word problem is both MATH and REASONING).

Output: single word (domain label).
Maps to: prefrontal cortex / executive control / routing circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

DOMAINS = ["MATH", "LANGUAGE", "SPATIAL", "FACTUAL", "EMOTIONAL"]

_INSTRUCTION = (
    "What cognitive ability does this task primarily require? "
    "Answer with exactly one word from: MATH, LANGUAGE, SPATIAL, FACTUAL, EMOTIONAL."
)

# Easy: unambiguous single domain
EASY_ITEMS = [
    {"task": "What is 17 * 23?", "accept": ["MATH"]},
    {"task": "Is this sentence grammatically correct: 'Him went store.'", "accept": ["LANGUAGE"]},
    {"task": "What is the capital of Mongolia?", "accept": ["FACTUAL"]},
    {"task": "How would someone feel after being publicly humiliated?", "accept": ["EMOTIONAL"]},
    {"task": "Describe the path from the kitchen to the bedroom in a house with this floor plan.", "accept": ["SPATIAL"]},
    {"task": "What is the opposite of 'benevolent'?", "accept": ["LANGUAGE"]},
]

# Hard: ambiguous — multiple domains apply, or the obvious answer is wrong
HARD_ITEMS = [
    {"task": "You're facing north. You turn right, walk 3 blocks, turn left. What direction are you facing?",
     "accept": ["SPATIAL"]},
    {"task": "Is the speaker being sarcastic: 'Oh great, another meeting about meetings.'",
     "accept": ["EMOTIONAL", "LANGUAGE"]},
    {"task": "In the sentence 'The bank was steep', does 'bank' mean a financial institution or a riverbank?",
     "accept": ["LANGUAGE"]},
    {"task": "How many bones are in the adult human hand?", "accept": ["FACTUAL"]},
    {"task": "A child is crying alone at a playground. What should you do first?",
     "accept": ["EMOTIONAL"]},
    {"task": "Looking at a city map, which route from A to B avoids the one-way streets?",
     "accept": ["SPATIAL"]},
]


def score_routing(response: str, accept: list[str]) -> float:
    """Score routing classification. Any accepted category = 1.0."""
    r = response.strip().upper()
    ALIASES = {
        "MATH": ["MATH", "MATHEMATICAL", "ARITHMETIC"],
        "LANGUAGE": ["LANGUAGE", "LINGUISTIC", "GRAMMAR"],
        "SPATIAL": ["SPATIAL", "SPAT", "VISUAL"],
        "FACTUAL": ["FACTUAL", "FACT", "KNOWLEDGE", "RECALL"],
        "EMOTIONAL": ["EMOTIONAL", "EMOTION", "EQ", "EMPATHY"],
    }
    detected = None
    for cat, aliases in ALIASES.items():
        for alias in aliases:
            if alias in r:
                detected = cat
                break
        if detected:
            break
    if detected is None:
        return 0.0
    return 1.0 if detected in accept else 0.0


@register_probe
class RoutingProbe(BaseProbe):
    name = "routing"
    description = "Task domain classification — executive control circuits"

    def run(self, model) -> dict:
        easy_scores = []
        hard_scores = []
        item_results = [] if self.log_responses else None

        for difficulty, items in [("easy", self._limit(EASY_ITEMS)), ("hard", self._limit(HARD_ITEMS))]:
            scores = easy_scores if difficulty == "easy" else hard_scores
            for item in items:
                prompt = f"Task: {item['task']}\n{_INSTRUCTION}"
                response = model.generate_short(
                    prompt, max_new_tokens=5, temperature=0.0
                )
                score = score_routing(response, item["accept"])
                scores.append(score)
                if item_results is not None:
                    item_results.append({
                        "difficulty": difficulty,
                        "task": item["task"][:100],
                        "accepted": item["accept"],
                        "response": response[:100],
                        "score": score,
                    })

        return self._make_result(easy_scores, hard_scores, item_results)
