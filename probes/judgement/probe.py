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

# Hard: built from the model's ACTUAL wrong answers on other probes.
# These are answers the model itself generated — testing if it can
# recognize its own mistakes.
HARD_ITEMS = [
    {
        # Model's actual wrong answer on reasoning probe (expected 27)
        "question": "A store sells apples for $2 each and oranges for $3 each. Maria buys 4 apples and 5 oranges. She pays with a $50 bill. How much change does she get?",
        "answer": "19",
        "is_correct": False,  # 50 - (8+15) = 27
    },
    {
        # Model's actual wrong answer on reasoning probe (expected 40)
        "question": "A tank fills at 3 liters/min and drains at 1 liter/min. After 20 minutes filling, the fill stops. How many minutes until empty?",
        "answer": "60",
        "is_correct": False,  # 20*(3-1)=40 liters, drain at 1/min = 40 min
    },
    {
        # Model's actual wrong answer on reasoning probe (expected 20)
        "question": "Alice is twice as old as Bob. In 10 years, Alice will be 1.5 times as old as Bob. How old is Bob now?",
        "answer": "10",
        "is_correct": False,  # 2x+10=1.5(x+10), x=20
    },
    {
        # Model's actual wrong answer on consistency — CoT said Friday, direct said Wednesday
        "question": "If today is Wednesday, what day will it be 100 days from now?",
        "answer": "Wednesday",
        "is_correct": False,  # 100/7=14r2, Wed+2=Friday
    },
    {
        # Correct answer the model DID get right — control item
        "question": "A snail climbs 3m up each day but slides back 2m each night. The wall is 10m tall. On which day does the snail reach the top?",
        "answer": "8",
        "is_correct": True,
    },
    {
        # Common misconception the model fell for
        "question": "What is the largest desert in the world?",
        "answer": "The Sahara",
        "is_correct": False,  # Antarctica
    },
    {
        # Model classified this as MATH instead of REASONING
        "question": "Three friends split a bill. Alice pays twice what Bob pays. Carol pays $10 more than Bob. Total is $70. How much does Bob pay?",
        "answer": "$15",
        "is_correct": True,  # 2x + x + (x+10) = 70, 4x=60, x=15
    },
    {
        # Classic trick — model got it right but most don't
        "question": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
        "answer": "$0.10",
        "is_correct": False,  # $0.05
    },
    {
        # Model's actual wrong answer on consistency — direct said 0 degrees
        "question": "A clock shows 3:15. What is the angle between the hour and minute hands?",
        "answer": "0 degrees",
        "is_correct": False,  # 7.5 degrees
    },
    {
        # Subtly wrong — right direction, wrong value
        "question": "What is 15% of 250?",
        "answer": "35",
        "is_correct": False,  # 37.5
    },
    {
        # Correct but counterintuitive
        "question": "Is zero an even number?",
        "answer": "Yes",
        "is_correct": True,
    },
    {
        # Model's actual wrong answer on reasoning (expected 57%)
        "question": "Box A has 2 red and 3 blue balls. Box B has 4 red and 1 blue. Box C has 1 red and 4 blue. You pick a random box and draw red. What's the probability it came from Box B (nearest percent)?",
        "answer": "71%",
        "is_correct": False,  # P(B|red) = (4/5)/(2/5+4/5+1/5) = 4/7 ≈ 57%
    },
    # --- Added items for convergence reliability ---
    {
        "question": "What is the square root of 64?",
        "answer": "8",
        "is_correct": True,
    },
    {
        "question": "What is the capital of Japan?",
        "answer": "Beijing",
        "is_correct": False,  # Tokyo
    },
    {
        "question": "How many sides does a hexagon have?",
        "answer": "6",
        "is_correct": True,
    },
    {
        "question": "What is the chemical symbol for sodium?",
        "answer": "So",
        "is_correct": False,  # Na
    },
    {
        "question": "If a train travels 90 km in 1.5 hours, what is its speed in km/h?",
        "answer": "60",
        "is_correct": True,
    },
    {
        "question": "What is 3^4?",
        "answer": "64",
        "is_correct": False,  # 81
    },
    {
        "question": "How many vertices does a cube have?",
        "answer": "8",
        "is_correct": True,
    },
    {
        "question": "What planet is known as the Red Planet?",
        "answer": "Jupiter",
        "is_correct": False,  # Mars
    },
    {
        "question": "What is the derivative of x^3?",
        "answer": "3x^2",
        "is_correct": True,
    },
    {
        "question": "A recipe calls for doubling 3/4 cup. How much is that?",
        "answer": "1 and 1/4 cups",
        "is_correct": False,  # 1 and 1/2 cups
    },
    {
        "question": "What is 17 * 6?",
        "answer": "102",
        "is_correct": True,
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "answer": "Charles Dickens",
        "is_correct": False,  # Shakespeare
    },
    {
        "question": "What is the sum of angles in a triangle?",
        "answer": "180 degrees",
        "is_correct": True,
    },
    {
        "question": "In what year did World War I end?",
        "answer": "1917",
        "is_correct": False,  # 1918
    },
    {
        "question": "What is the value of the golden ratio (to 3 decimal places)?",
        "answer": "1.618",
        "is_correct": True,
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
