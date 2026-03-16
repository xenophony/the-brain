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

# Hard: the obvious domain is WRONG — tests if the model can see past surface cues
HARD_ITEMS = [
    # Has numbers but the challenge is reading comprehension, not calculation
    {"task": "Tom has 3 sisters. Each sister has 1 brother. How many brothers does Tom have?",
     "accept": ["LANGUAGE"]},  # Trick question — answer is 0 (Tom IS the brother). It's a language/logic trap, not math.
    # Looks emotional but requires factual knowledge to judge
    {"task": "A doctor tells a patient they have 6 months to live. The patient smiles. Why might they smile?",
     "accept": ["EMOTIONAL"]},
    # Looks like factual recall but requires spatial visualization
    {"task": "If you fold a standard letter in thirds, which part is visible on top?",
     "accept": ["SPATIAL"]},
    # Looks like math but is really about language precision
    {"task": "A farmer has 17 sheep. All but 9 die. How many are left?",
     "accept": ["LANGUAGE"]},  # "All but 9" = 9 remain. Language comprehension, not subtraction.
    # Looks like language but requires emotional intelligence
    {"task": "Your friend says 'I'm fine' while avoiding eye contact. What do they mean?",
     "accept": ["EMOTIONAL"]},
    # Looks spatial but is really factual
    {"task": "Which is further north: New York City or Rome?",
     "accept": ["FACTUAL", "SPATIAL"]},
    # Sounds like math but the answer requires no calculation
    {"task": "If there are 3 apples and you take away 2, how many apples do YOU have?",
     "accept": ["LANGUAGE"]},  # You have 2 — you took them. Language trick.
    # Looks emotional but requires factual cultural knowledge
    {"task": "In Japan, is it rude to tip at a restaurant?",
     "accept": ["FACTUAL"]},
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
