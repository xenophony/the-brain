"""
Holistic/analogy probe — complete non-obvious conceptual analogies.

Tests associative and analogical reasoning by presenting "A is to B as C is to ?"
patterns that require conceptual bridging rather than surface similarity.

Output: single word.
Scoring: exact match OR semantically equivalent (from pre-defined accept list).
Maps to: default mode network / associative thinking circuits.
"""

from probes.registry import BaseProbe, register_probe

EASY_ITEMS = [
    {
        "prompt": "Hand is to glove as foot is to ___. Answer with only one word.",
        "accept": ["sock", "shoe", "boot"],
    },
    {
        "prompt": "Book is to library as painting is to ___. Answer with only one word.",
        "accept": ["museum", "gallery"],
    },
    {
        "prompt": "Lock is to key as account is to ___. Answer with only one word.",
        "accept": ["password", "credential"],
    },
    {
        "prompt": "Summer is to winter as day is to ___. Answer with only one word.",
        "accept": ["night"],
    },
    {
        "prompt": "Stars is to telescope as cells is to ___. Answer with only one word.",
        "accept": ["microscope"],
    },
    {
        "prompt": "Desert is to oasis as ocean is to ___. Answer with only one word.",
        "accept": ["island"],
    },
    {
        "prompt": "Teacher is to student as doctor is to ___. Answer with only one word.",
        "accept": ["patient"],
    },
    {
        "prompt": "Bark is to dog as meow is to ___. Answer with only one word.",
        "accept": ["cat"],
    },
]

HARD_ITEMS = [
    {
        "prompt": "Canvas is to painter as score is to ___. Answer with only one word.",
        "accept": ["composer", "musician"],
    },
    {
        "prompt": "Gill is to fish as lung is to ___. Answer with only one word.",
        "accept": ["bird", "mammal", "human"],
    },
    {
        "prompt": "Pupil is to eye as eardrum is to ___. Answer with only one word.",
        "accept": ["ear"],
    },
    {
        "prompt": "Ship is to captain as orchestra is to ___. Answer with only one word.",
        "accept": ["conductor", "maestro"],
    },
    {
        "prompt": "Weapon is to sword as writing is to ___. Answer with only one word.",
        "accept": ["pen", "quill"],
    },
    {
        "prompt": "Building is to architect as play is to ___. Answer with only one word.",
        "accept": ["playwright", "dramatist", "writer"],
    },
    {
        "prompt": "Map is to territory as menu is to ___. Answer with only one word.",
        "accept": ["meal", "food", "restaurant", "cuisine"],
    },
    {
        "prompt": "Cocoon is to butterfly as seed is to ___. Answer with only one word.",
        "accept": ["tree", "plant", "flower"],
    },
]

# Legacy alias
ANALOGIES = EASY_ITEMS + HARD_ITEMS


def score_analogy(response: str, accepted: list[str]) -> float:
    """Score analogy completion. 1.0 if response matches any accepted answer."""
    response = response.strip().lower().rstrip(".")
    # Take the last word if model outputs a phrase
    words = response.split()
    if not words:
        return 0.0
    # Check each word against accepted answers
    for word in words:
        word_clean = word.strip(".,;:!?\"'()").lower()
        if word_clean in [a.lower() for a in accepted]:
            return 1.0
    return 0.0


@register_probe
class HolisticProbe(BaseProbe):
    name = "holistic"
    description = "Analogy completion — default mode network circuits"

    def run(self, model) -> dict:
        easy_scores = []
        for item in self._limit(EASY_ITEMS):
            response = model.generate_short(item["prompt"], max_new_tokens=10, temperature=0.0)
            score = score_analogy(response, item["accept"])
            easy_scores.append(score)

        hard_scores = []
        for item in self._limit(HARD_ITEMS):
            response = model.generate_short(item["prompt"], max_new_tokens=10, temperature=0.0)
            score = score_analogy(response, item["accept"])
            hard_scores.append(score)

        return self._make_result(easy_scores, hard_scores)
