"""
Routing probe — can the model correctly classify what capability a task needs?

Tests the "metacognitive router" circuit: given a problem, the model must
identify which cognitive domain is required (MATH, REASONING, LANGUAGE,
SPATIAL, FACTUAL, EMOTIONAL). This maps the decision-making circuit that
would control dynamic self-routing in an adaptive LLM.

Output: single word (domain label).
Maps to: prefrontal cortex / executive control / routing circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

# Categories the model must choose from
CATEGORIES = ["MATH", "REASONING", "LANGUAGE", "SPATIAL", "FACTUAL", "EMOTIONAL"]

_INSTRUCTION = (
    "What cognitive ability does this task primarily require? "
    "Answer with exactly one word from: MATH, REASONING, LANGUAGE, SPATIAL, FACTUAL, EMOTIONAL."
)

# Easy: clear-cut domain, no ambiguity
EASY_ITEMS = [
    {"task": "What is 17 * 23?", "answer": "MATH"},
    {"task": "Is this sentence grammatically correct: 'Him went store.'", "answer": "LANGUAGE"},
    {"task": "What is the capital of Mongolia?", "answer": "FACTUAL"},
    {"task": "How would someone feel after being publicly humiliated?", "answer": "EMOTIONAL"},
    {"task": "If a room is 4m x 5m, how many 1m x 1m tiles fit?", "answer": "SPATIAL"},
    {"task": "A train leaves at 9am going 60mph. Another at 10am going 80mph. When do they meet?",
     "answer": "REASONING"},
]

# Hard: ambiguous problems where multiple domains apply but one is PRIMARY
HARD_ITEMS = [
    # Looks like math but primarily needs logical reasoning (setup/constraint analysis)
    {"task": "Three friends split a bill. Alice pays twice what Bob pays. Carol pays $10 more than Bob. Total is $70. How much does Bob pay?",
     "answer": "REASONING"},
    # Looks like factual but requires spatial reasoning about geography
    {"task": "You're facing north in Chicago. You turn right, walk 3 blocks, turn left. What direction are you facing?",
     "answer": "SPATIAL"},
    # Looks like language but requires emotional understanding of tone
    {"task": "Is the speaker being sarcastic: 'Oh great, another meeting about meetings.'",
     "answer": "EMOTIONAL"},
    # Looks like reasoning but it's pure arithmetic
    {"task": "What is 15% of 240 minus 8% of 150?",
     "answer": "MATH"},
    # Looks like factual recall but requires language analysis
    {"task": "In the sentence 'The bank was steep', does 'bank' refer to a financial institution or a riverbank?",
     "answer": "LANGUAGE"},
    # Looks like math but is actually a factual question dressed up
    {"task": "How many bones are in the adult human hand?",
     "answer": "FACTUAL"},
]


def score_routing(response: str, expected: str) -> float:
    """Score routing classification. Exact category match = 1.0."""
    r = response.strip().upper()
    # Handle truncated labels (e.g. "EMOTION" for "EMOTIONAL")
    ALIASES = {
        "MATH": ["MATH", "MATHEMATICAL", "ARITHMETIC"],
        "REASONING": ["REASONING", "REASON", "LOGIC", "LOGICAL"],
        "LANGUAGE": ["LANGUAGE", "LINGUISTIC", "GRAMMAR"],
        "SPATIAL": ["SPATIAL", "VISUAL"],
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
    return 1.0 if detected == expected else 0.0


@register_probe
class RoutingProbe(BaseProbe):
    name = "routing"
    description = "Task routing classification — executive control circuits"

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
                score = score_routing(response, item["answer"])
                scores.append(score)
                if item_results is not None:
                    item_results.append({
                        "difficulty": difficulty,
                        "task": item["task"][:100],
                        "expected": item["answer"],
                        "response": response[:100],
                        "score": score,
                    })

        return self._make_result(easy_scores, hard_scores, item_results)
