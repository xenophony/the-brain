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

SENTENCES = [
    # --- Grammatical sentences ---
    {
        "sentence": "The committee has decided to postpone the meeting.",
        "label": "grammatical",
    },
    {
        "sentence": "Neither the students nor the teacher was aware of the change.",
        "label": "grammatical",
    },
    {
        "sentence": "The books that I bought yesterday are on the shelf.",
        "label": "grammatical",
    },
    {
        "sentence": "Each of the participants has submitted a report.",
        "label": "grammatical",
    },
    {
        "sentence": "The horse raced past the barn fell.",
        "label": "grammatical",  # Garden path — reduced relative clause, actually grammatical
    },
    {
        "sentence": "That that is is that that is not is not.",
        "label": "grammatical",  # Confusing but grammatical
    },
    {
        "sentence": "The manager, along with her assistants, is attending the conference.",
        "label": "grammatical",
    },
    {
        "sentence": "Had I known, I would have acted differently.",
        "label": "grammatical",
    },
    # --- Ungrammatical sentences ---
    {
        "sentence": "The children plays in the garden every afternoon.",
        "label": "ungrammatical",  # Subject-verb agreement
    },
    {
        "sentence": "Him went to the store to buy some groceries.",
        "label": "ungrammatical",  # Case error
    },
    {
        "sentence": "She don't like the way he talks to her.",
        "label": "ungrammatical",  # Auxiliary agreement
    },
    {
        "sentence": "They was planning to leave early this morning.",
        "label": "ungrammatical",  # Subject-verb agreement
    },
    {
        "sentence": "A books are sitting on the table over there.",
        "label": "ungrammatical",  # Determiner-noun number agreement
    },
    {
        "sentence": "He goed to the market yesterday afternoon.",
        "label": "ungrammatical",  # Irregular past tense
    },
    {
        "sentence": "She is more taller than her older sister.",
        "label": "ungrammatical",  # Double comparative
    },
    {
        "sentence": "The dog that the cat that the rat bit chased died.",
        "label": "ungrammatical",  # Center embedding overload — effectively unprocessable
    },
]

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

    def run(self, model) -> float:
        scores = []
        for item in SENTENCES:
            prompt = PROMPT_TEMPLATE.format(sentence=item["sentence"])
            response = model.generate_short(prompt, max_new_tokens=5, temperature=0.0)
            score = score_language(response, item["label"])
            scores.append(score)
        return sum(scores) / len(scores)
