"""
Metacognition probe -- self-monitoring and calibration.

20 questions: model answers AND gives confidence 0-9.
Score = calibration quality: confident when right, uncertain when wrong.

Prompt format: "Answer this question, then on a new line write only a digit
0-9 for your confidence (0=guessing, 9=certain):\n{question}"

Calibration score per question:
  confidence_normalized = confidence / 9.0
  if correct: score = confidence_normalized
  if wrong: score = 1.0 - confidence_normalized

Question mix: 5 definitely-knows + 5 often-wrong + 5 obscure + 5 trick.
Overall = mean across 20.

Output: answer + confidence digit.
Maps to: prefrontal cortex / self-monitoring circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

QUESTIONS = [
    # --- Definitely-knows (5) ---
    {
        "question": "What is 2 + 2?",
        "accept": ["4", "four"],
        "category": "easy",
    },
    {
        "question": "What color is the sky on a clear day?",
        "accept": ["blue"],
        "category": "easy",
    },
    {
        "question": "How many legs does a dog have?",
        "accept": ["4", "four"],
        "category": "easy",
    },
    {
        "question": "What planet do we live on?",
        "accept": ["earth"],
        "category": "easy",
    },
    {
        "question": "What is the boiling point of water in Celsius?",
        "accept": ["100"],
        "category": "easy",
    },
    # --- Often-wrong (5) ---
    {
        "question": "What is the square root of 169?",
        "accept": ["13"],
        "category": "medium",
    },
    {
        "question": "How many countries are in Africa?",
        "accept": ["54"],
        "category": "medium",
    },
    {
        "question": "What is the atomic number of iron?",
        "accept": ["26"],
        "category": "medium",
    },
    {
        "question": "In what year did the Berlin Wall fall?",
        "accept": ["1989"],
        "category": "medium",
    },
    {
        "question": "What is the chemical formula for sulfuric acid?",
        "accept": ["h2so4"],
        "category": "medium",
    },
    # --- Obscure (5) ---
    {
        "question": "What is the capital of Burkina Faso?",
        "accept": ["ouagadougou"],
        "category": "obscure",
    },
    {
        "question": "What year was the Treaty of Westphalia signed?",
        "accept": ["1648"],
        "category": "obscure",
    },
    {
        "question": "What is the half-life of Bismuth-209 in years (order of magnitude)?",
        "accept": ["10^19", "1e19", "19"],
        "category": "obscure",
    },
    {
        "question": "Who was the 14th president of the United States?",
        "accept": ["pierce", "franklin pierce"],
        "category": "obscure",
    },
    {
        "question": "What is the deepest point in the ocean in meters (approximate)?",
        "accept": ["10994", "10935", "11000", "10900", "11034"],
        "category": "obscure",
    },
    # --- Trick questions (5) ---
    {
        "question": "How many times can you subtract 5 from 25?",
        "accept": ["1", "one", "once"],
        "category": "trick",
    },
    {
        "question": "If there are 3 apples and you take away 2, how many do you have?",
        "accept": ["2", "two"],
        "category": "trick",
    },
    {
        "question": "A farmer has 15 sheep. All but 8 die. How many are left?",
        "accept": ["8", "eight"],
        "category": "trick",
    },
    {
        "question": "What weighs more: a pound of feathers or a pound of bricks?",
        "accept": ["same", "equal", "neither", "they weigh the same"],
        "category": "trick",
    },
    {
        "question": "How many months have 28 days?",
        "accept": ["12", "all", "twelve"],
        "category": "trick",
    },
]

PROMPT_TEMPLATE = (
    "Answer this question, then on a new line write only a digit 0-9 "
    "for your confidence (0=guessing, 9=certain):\n{question}"
)


def extract_answer_and_confidence(response: str) -> tuple[str, int | None]:
    """
    Extract the answer text and confidence digit from a response.
    Expected format: answer text followed by a line with a single digit.
    The LAST standalone digit line is treated as confidence.
    """
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]

    if not lines:
        return "", None

    # Find the LAST line that is just a single digit -- that is the confidence
    confidence = None
    confidence_idx = None

    for i in range(len(lines) - 1, -1, -1):
        if re.match(r'^[0-9]$', lines[i]):
            confidence = int(lines[i])
            confidence_idx = i
            break

    if confidence_idx is not None:
        answer_lines = lines[:confidence_idx]
        # If there are lines after the confidence digit, include them in answer
        # (unlikely but defensive)
        answer_lines.extend(lines[confidence_idx + 1:])
    else:
        # No standalone digit line found -- look for trailing digit on last line
        last_line = lines[-1]
        trailing = re.search(r'(\d)\s*$', last_line)
        if trailing:
            confidence = int(trailing.group(1))
            answer_part = last_line[:trailing.start()].strip()
            if len(lines) > 1:
                answer_lines = lines[:-1]
                if answer_part:
                    answer_lines.append(answer_part)
            else:
                answer_lines = [answer_part] if answer_part else []
        else:
            answer_lines = lines

    answer = " ".join(answer_lines).strip()
    return answer, confidence


def score_metacognition(response: str, question: dict) -> float:
    """
    Score a metacognition response based on calibration.
    Correct + high confidence = good. Wrong + low confidence = good.
    """
    answer, confidence = extract_answer_and_confidence(response)

    if confidence is None:
        return 0.0

    confidence = max(0, min(9, confidence))
    confidence_normalized = confidence / 9.0

    # Check correctness
    answer_lower = answer.lower()
    correct = any(acc.lower() in answer_lower for acc in question["accept"])

    if correct:
        return confidence_normalized
    else:
        return 1.0 - confidence_normalized


@register_probe
class MetacognitionProbe(BaseProbe):
    name = "metacognition"
    description = "Self-monitoring calibration -- prefrontal cortex / metacognitive circuits"

    def run(self, model) -> float:
        scores = []
        for q in self._limit(QUESTIONS):
            prompt = PROMPT_TEMPLATE.format(question=q["question"])
            response = model.generate_short(prompt, max_new_tokens=30, temperature=0.0)
            score = score_metacognition(response, q)
            scores.append(score)
        return sum(scores) / len(scores)
