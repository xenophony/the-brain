"""
Judgement probe — can the model evaluate if a response correctly answers a question?

Maps the self-evaluation circuit: the ability to judge answer correctness.
This is the circuit needed for retry-with-feedback, self-correction, and
quality gating in agentic workflows.

Each item presents a question + candidate answer. The model must judge
"correct" or "incorrect". Mix of actually correct answers, wrong answers,
and subtly wrong answers (right format, wrong value).

Output: single word ("correct" or "incorrect").
Maps to: prefrontal cortex / self-monitoring / metacognitive evaluation.
"""

import re
from probes.registry import BaseProbe, register_probe

_INSTRUCTION = (
    "Was the given answer correct? "
    "Answer with exactly one word: correct or incorrect."
)

# Easy: obviously right or obviously wrong
EASY_ITEMS = [
    {
        "question": "What is 2 + 2?",
        "answer": "4",
        "is_correct": True,
    },
    {
        "question": "What is the capital of France?",
        "answer": "London",
        "is_correct": False,
    },
    {
        "question": "How many days are in a week?",
        "answer": "7",
        "is_correct": True,
    },
    {
        "question": "What color is the sky on a clear day?",
        "answer": "green",
        "is_correct": False,
    },
    {
        "question": "Is water wet?",
        "answer": "Yes",
        "is_correct": True,
    },
    {
        "question": "What is 10 * 5?",
        "answer": "500",
        "is_correct": False,
    },
]

# Hard: subtly wrong, plausible but incorrect, or correct but unexpected format
HARD_ITEMS = [
    {
        # Off by one — close but wrong, requires mental math to catch
        "question": "What is 17 * 23?",
        "answer": "392",
        "is_correct": False,  # 391
    },
    {
        # Common misconception — model might agree with wrong answer
        "question": "What is the largest desert in the world?",
        "answer": "The Sahara",
        "is_correct": False,  # Antarctica is larger
    },
    {
        # Correct but counterintuitive — model might say incorrect
        "question": "Which is heavier: a pound of feathers or a pound of steel?",
        "answer": "They weigh the same",
        "is_correct": True,
    },
    {
        # Subtly wrong — right method, wrong result
        "question": "If a shirt costs $80 after a 20% discount, what was the original price?",
        "answer": "$96",
        "is_correct": False,  # Should be $100
    },
    {
        # Wrong but confident-sounding answer
        "question": "What is the sum of angles in a triangle?",
        "answer": "360 degrees",
        "is_correct": False,  # 180
    },
    {
        # Correct answer that sounds wrong
        "question": "What percentage of the Earth's surface is covered by water?",
        "answer": "About 71%",
        "is_correct": True,
    },
    {
        # Plausible wrong — off by a common confusion
        "question": "How many chromosomes do humans have?",
        "answer": "24",
        "is_correct": False,  # 46 (24 pairs is wrong framing too)
    },
    {
        # Correct but model might second-guess
        "question": "Is zero an even number?",
        "answer": "Yes",
        "is_correct": True,
    },
    {
        # Sounds right but wrong — needs actual calculation
        "question": "What is 15% of 250?",
        "answer": "35",
        "is_correct": False,  # 37.5
    },
    {
        # Right answer, unusual phrasing
        "question": "How many sides does a hexagon have?",
        "answer": "Half a dozen",
        "is_correct": True,
    },
    {
        # Classic trick — most models fall for this
        "question": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
        "answer": "$0.10",
        "is_correct": False,  # $0.05
    },
    {
        # Correct but requires careful reading
        "question": "If you have 3 apples and take away 2, how many do you have?",
        "answer": "2",
        "is_correct": True,  # You took 2, so you have 2
    },
]


def score_judgement(response: str, is_correct: bool) -> float:
    """Score a correct/incorrect judgement."""
    r = response.strip().lower()
    expected = "correct" if is_correct else "incorrect"
    # Check "incorrect" before "correct" (substring issue)
    if "incorrect" in r or "wrong" in r or "no" == r:
        got = "incorrect"
    elif "correct" in r or "right" in r or "yes" == r:
        got = "correct"
    else:
        return 0.0
    return 1.0 if got == expected else 0.0


@register_probe
class JudgementProbe(BaseProbe):
    name = "judgement"
    description = "Answer correctness evaluation — self-monitoring circuits"

    def run(self, model) -> dict:
        easy_scores = []
        hard_scores = []
        item_results = [] if self.log_responses else None

        for difficulty, items in [("easy", self._limit(EASY_ITEMS)), ("hard", self._limit(HARD_ITEMS))]:
            scores = easy_scores if difficulty == "easy" else hard_scores
            for item in items:
                prompt = (
                    f"Question: {item['question']}\n"
                    f"Answer given: {item['answer']}\n"
                    f"{_INSTRUCTION}"
                )
                response = model.generate_short(
                    prompt, max_new_tokens=5, temperature=0.0
                )
                score = score_judgement(response, item["is_correct"])
                scores.append(score)
                if item_results is not None:
                    item_results.append({
                        "difficulty": difficulty,
                        "question": item["question"][:100],
                        "given_answer": item["answer"],
                        "is_correct": item["is_correct"],
                        "response": response[:100],
                        "score": score,
                    })

        return self._make_result(easy_scores, hard_scores, item_results)
