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
    {"sentence": "Me and him went to the park after school.", "label": "ungrammatical"},
    {"sentence": "The dog chased it own tail around the yard.", "label": "ungrammatical"},
    {"sentence": "She have been waiting for over an hour now.", "label": "ungrammatical"},
    {"sentence": "There is many reasons to be optimistic about the future.", "label": "ungrammatical"},
    {"sentence": "He runned all the way to the finish line.", "label": "ungrammatical"},
    {"sentence": "The cats meowed loudly because it were hungry.", "label": "ungrammatical"},
    {"sentence": "Her and me decided to split the dessert.", "label": "ungrammatical"},
    {"sentence": "Each student must bring their own pencil to the exam.", "label": "grammatical"},
    {"sentence": "The professor explained the theory clearly and concisely.", "label": "grammatical"},
    {"sentence": "We have been living in this city for five years.", "label": "grammatical"},
    {"sentence": "The birds flew south before the first frost arrived.", "label": "grammatical"},
    {"sentence": "Running through the forest, she felt completely free.", "label": "grammatical"},
    {"sentence": "Neither option appeals to me at this point.", "label": "grammatical"},
    {"sentence": "The children were excited about the upcoming field trip.", "label": "grammatical"},
    {"sentence": "She suggested that we leave before the storm hits.", "label": "grammatical"},
    {"sentence": "He don't know nothing about what happened last night.", "label": "ungrammatical"},
    {"sentence": "The furnitures in the office need to be replaced.", "label": "ungrammatical"},
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
    {"sentence": "The horse raced past the barn fell.", "label": "grammatical"},
    {"sentence": "The man who the woman who the child saw greeted left.", "label": "grammatical"},
    {"sentence": "If I were you, I would reconsider that decision.", "label": "grammatical"},
    {"sentence": "It is essential that he be present at the hearing.", "label": "grammatical"},
    {"sentence": "The data suggests that our hypothesis were correct.", "label": "ungrammatical"},
    {"sentence": "Whom did the committee decide should lead the project?", "label": "ungrammatical"},
    {"sentence": "The number of students who has passed the exam is rising.", "label": "ungrammatical"},
    {"sentence": "She insisted that the report is submitted by Friday.", "label": "ungrammatical"},
    {"sentence": "The fact that that explanation was accepted surprised everyone.", "label": "grammatical"},
    {"sentence": "More people than ever before are working from home.", "label": "grammatical"},
    {"sentence": "What he said and what he did was two different things.", "label": "ungrammatical"},
    {"sentence": "The student who the teacher that the principal hired taught passed.", "label": "grammatical"},
    {"sentence": "Time flies like an arrow; fruit flies like a banana.", "label": "grammatical"},
    {"sentence": "The old man the boats when the young sailors are away.", "label": "grammatical"},
    {"sentence": "Not only did she win the race, but she also set a record.", "label": "grammatical"},
    {"sentence": "The reason is because he forgot to set his alarm.", "label": "ungrammatical"},
    {"sentence": "Neither the teacher nor the students was able to solve it.", "label": "ungrammatical"},
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
        easy_scores, easy_results = self._run_items(
            model, self._limit(EASY_ITEMS),
            prompt_fn=lambda item: PROMPT_TEMPLATE.format(sentence=item["sentence"]),
            score_fn=lambda resp, item: score_language(resp, item["label"]),
            max_new_tokens=5, difficulty="easy")

        hard_scores, hard_results = self._run_items(
            model, self._limit(HARD_ITEMS),
            prompt_fn=lambda item: PROMPT_TEMPLATE.format(sentence=item["sentence"]),
            score_fn=lambda resp, item: score_language(resp, item["label"]),
            max_new_tokens=5, difficulty="hard")

        item_results = (easy_results + hard_results) if self.log_responses else None
        return self._make_result(easy_scores, hard_scores, item_results)
