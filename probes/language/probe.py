"""
Language/syntax probe — grammaticality judgment.

Presents sentences and asks whether each is grammatical or ungrammatical.
Uses subtle violations: agreement errors, garden paths, center embedding,
scope ambiguity.

Output: single word "grammatical" or "ungrammatical".
Scoring: exact match = 1.0.
Maps to: Broca/Wernicke language circuits.
"""

from probes.registry import BaseProbe, register_probe

EASY_ITEMS = [
    # --- Clear ungrammatical ---
    {"sentence": "The children plays in the garden every afternoon.", "label": "ungrammatical"},
    {"sentence": "Him went to the store to buy some groceries.", "label": "ungrammatical"},
    {"sentence": "She don't like the way he talks to her.", "label": "ungrammatical"},
    {"sentence": "They was planning to leave early this morning.", "label": "ungrammatical"},
    {"sentence": "A books are sitting on the table over there.", "label": "ungrammatical"},
    {"sentence": "He goed to the market yesterday afternoon.", "label": "ungrammatical"},
    {"sentence": "She is more taller than her older sister.", "label": "ungrammatical"},
    # --- Clear grammatical ---
    {"sentence": "The committee has decided to postpone the meeting.", "label": "grammatical"},
]

HARD_ITEMS = [
    # --- Subtle grammatical sentences ---
    {"sentence": "Neither the students nor the teacher was aware of the change.", "label": "grammatical"},
    {"sentence": "Each of the participants has submitted a report.", "label": "grammatical"},
    {"sentence": "The manager, along with her assistants, is attending the conference.", "label": "grammatical"},
    {"sentence": "Had I known, I would have acted differently.", "label": "grammatical"},
    {"sentence": "That that is is that that is not is not.", "label": "grammatical"},
    {"sentence": "The books that I bought yesterday are on the shelf.", "label": "grammatical"},
    # --- Subtle ungrammatical ---
    {"sentence": "The team are going to their separate homes tonight.", "label": "grammatical"},
    {"sentence": "Between you and I, this project is failing.", "label": "ungrammatical"},
]

# Legacy alias
SENTENCES = EASY_ITEMS + HARD_ITEMS

PROMPT_TEMPLATE = (
    'Is the following sentence grammatical or ungrammatical?\n\n'
    '"{sentence}"\n\n'
    'Answer with only one word: "grammatical" or "ungrammatical".'
)


def score_language(response: str, expected_label: str) -> float:
    """Score grammaticality judgment. Exact match = 1.0."""
    response = response.strip().lower()
    # Extract the key word
    if "ungrammatical" in response:
        got = "ungrammatical"
    elif "grammatical" in response:
        got = "grammatical"
    else:
        return 0.0
    return 1.0 if got == expected_label else 0.0


@register_probe
class LanguageProbe(BaseProbe):
    name = "language"
    description = "Syntactic anomaly detection — Broca/Wernicke circuits"

    def run(self, model) -> dict:
        easy_scores = []
        for item in self._limit(EASY_ITEMS):
            prompt = PROMPT_TEMPLATE.format(sentence=item["sentence"])
            response = model.generate_short(prompt, max_new_tokens=5, temperature=0.0)
            score = score_language(response, item["label"])
            easy_scores.append(score)

        hard_scores = []
        for item in self._limit(HARD_ITEMS):
            prompt = PROMPT_TEMPLATE.format(sentence=item["sentence"])
            response = model.generate_short(prompt, max_new_tokens=5, temperature=0.0)
            score = score_language(response, item["label"])
            hard_scores.append(score)

        return self._make_result(easy_scores, hard_scores)
