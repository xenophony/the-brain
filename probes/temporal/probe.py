"""
Temporal reasoning probe -- understanding time relationships.

16 questions across 4 types:
  Type A: Causal chain sequencing (yes/no)
  Type B: Relative time inference (integer days, partial credit +/-1)
  Type C: Temporal contradiction detection (consistent/inconsistent)
  Type D: Counterfactual temporal (short deterministic answer)

Distinct from planning (ordering actions) -- this measures understanding
of temporal relationships, cause-effect chains, and time logic.

Output: short answer (yes/no, integer, consistent/inconsistent, or short phrase).
Maps to: hippocampus / temporal lobe sequencing circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

QUESTIONS = [
    # --- Type A: Causal chain sequencing (4) ---
    {
        "type": "A",
        "prompt": (
            "Event chain: A fire starts. Then the alarm sounds. Then the fire truck arrives. "
            "Then water is sprayed. Could the fire truck have arrived before the alarm sounded? "
            "Answer only yes or no."
        ),
        "answer": "no",
    },
    {
        "type": "A",
        "prompt": (
            "Event chain: The sun rises. Then birds start singing. Then people wake up. "
            "Could the birds have started singing after the sun rose? Answer only yes or no."
        ),
        "answer": "yes",
    },
    {
        "type": "A",
        "prompt": (
            "Event chain: A seed is planted. Then it rains. Then the plant grows. "
            "Could the plant have grown after it rained? Answer only yes or no."
        ),
        "answer": "yes",
    },
    {
        "type": "A",
        "prompt": (
            "Event chain: A cake is mixed. It is baked. It cools. It is frosted. "
            "Could the cake be frosted before it is baked? Answer only yes or no."
        ),
        "answer": "no",
    },
    # --- Type B: Relative time inference (4) ---
    {
        "type": "B",
        "prompt": (
            "Alice arrived on Monday. Bob arrived 3 days after Alice. "
            "Carol arrived 2 days before Bob. How many days after Alice did Carol arrive? "
            "Answer with only a number."
        ),
        "answer": 1,
    },
    {
        "type": "B",
        "prompt": (
            "A package was shipped on March 5. It took 7 days to arrive. "
            "The recipient opened it 2 days later. How many days after shipping was it opened? "
            "Answer with only a number."
        ),
        "answer": 9,
    },
    {
        "type": "B",
        "prompt": (
            "Event X happened on January 10. Event Y happened 15 days later. "
            "Event Z happened 5 days before Y. How many days after X did Z happen? "
            "Answer with only a number."
        ),
        "answer": 10,
    },
    {
        "type": "B",
        "prompt": (
            "Tom started a project on day 1. He finished phase 1 on day 8. "
            "He finished phase 2 twelve days after phase 1. "
            "How many days from the start did phase 2 finish? Answer with only a number."
        ),
        "answer": 20,
    },
    # --- Type C: Temporal contradiction detection (4) ---
    {
        "type": "C",
        "prompt": (
            "Read this passage: 'John graduated from college in 2015. "
            "He started his PhD in 2014. He completed his PhD in 2019.' "
            "Is the timeline consistent or inconsistent? Answer only consistent or inconsistent."
        ),
        "answer": "inconsistent",
    },
    {
        "type": "C",
        "prompt": (
            "Read this passage: 'The store opened at 9 AM. The first customer arrived at 9:15 AM. "
            "The store was empty until 10 AM.' "
            "Is the timeline consistent or inconsistent? Answer only consistent or inconsistent."
        ),
        "answer": "inconsistent",
    },
    {
        "type": "C",
        "prompt": (
            "Read this passage: 'Maria was born in 1990. She started school at age 6. "
            "She graduated high school in 2008.' "
            "Is the timeline consistent or inconsistent? Answer only consistent or inconsistent."
        ),
        "answer": "consistent",
    },
    {
        "type": "C",
        "prompt": (
            "Read this passage: 'The building was demolished in March. "
            "Renovations were completed in April. New tenants moved in May.' "
            "Is the timeline consistent or inconsistent? Answer only consistent or inconsistent."
        ),
        "answer": "inconsistent",
    },
    # --- Type D: Counterfactual temporal (4) ---
    {
        "type": "D",
        "prompt": (
            "Normally: egg is boiled for 10 minutes, then peeled, then eaten. "
            "If the egg were peeled before boiling, would the result be the same? "
            "Answer only yes or no."
        ),
        "answer": "no",
    },
    {
        "type": "D",
        "prompt": (
            "Normally: clothes are washed, then dried, then folded. "
            "If clothes were folded before drying, would they stay neatly folded? "
            "Answer only yes or no."
        ),
        "answer": "no",
    },
    {
        "type": "D",
        "prompt": (
            "Normally: a photo is taken, then developed, then framed. "
            "If the photo is framed before being taken, is there anything to frame? "
            "Answer only yes or no."
        ),
        "answer": "no",
    },
    {
        "type": "D",
        "prompt": (
            "Normally: a letter is written, then sealed in an envelope, then mailed. "
            "If the letter is sealed before writing, can the recipient read it? "
            "Answer only yes or no."
        ),
        "answer": "no",
    },
]


def score_temporal(response: str, question: dict) -> float:
    """Score a temporal reasoning response."""
    r = response.strip().lower()
    qtype = question["type"]

    if qtype == "B":
        # Integer answer with partial credit for +/-1
        expected = question["answer"]
        matches = re.findall(r'-?\d+', r)
        if not matches:
            return 0.0
        try:
            got = int(matches[-1])
        except ValueError:
            return 0.0
        if got == expected:
            return 1.0
        if abs(got - expected) == 1:
            return 0.5
        return 0.0

    # Types A, C, D: exact match
    expected = question["answer"].lower()
    if expected in r:
        return 1.0
    return 0.0


@register_probe
class TemporalProbe(BaseProbe):
    name = "temporal"
    description = "Temporal reasoning -- hippocampus / temporal lobe sequencing"

    def run(self, model) -> float:
        scores = []
        for q in QUESTIONS:
            response = model.generate_short(q["prompt"], max_new_tokens=15, temperature=0.0)
            score = score_temporal(response, q)
            scores.append(score)
        return sum(scores) / len(scores)
