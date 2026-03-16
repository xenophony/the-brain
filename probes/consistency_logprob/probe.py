"""
Consistency logprob proxy — reasoning-output alignment circuits.

The full consistency probe generates CoT then direct answers (expensive).
This proxy measures whether adding reasoning context changes P(correct_answer).

For each question, we measure P(correct) in two conditions:
  - Direct: "Answer directly: {question}"
  - With context: "Given that {reasoning_hint}, answer: {question}"

Score = P(correct | with_context) - P(correct | direct).
Positive = reasoning helps (coherent circuits).
Negative = reasoning hurts (misaligned circuits).

One forward pass per condition, batched. No decode steps.
Maps to: internal consistency / reasoning-output alignment circuits.
"""

import math
from probes.registry import BaseProbe, register_probe


ITEMS = [
    # (question, correct_answer_token, reasoning_hint, difficulty)
    {
        "question": "If a train travels 60 km in 30 minutes, what is its speed in km/h?",
        "answer": "120",
        "hint": "distance = 60km, time = 0.5 hours, speed = distance/time",
        "difficulty": "easy",
    },
    {
        "question": "What is 15% of 200?",
        "answer": "30",
        "hint": "15% means multiply by 0.15, so 200 × 0.15",
        "difficulty": "easy",
    },
    {
        "question": "If 3 workers can build a wall in 12 hours, how many hours for 6 workers?",
        "answer": "6",
        "hint": "workers × hours = constant, so 3×12 = 6×h, h = 36/6",
        "difficulty": "easy",
    },
    {
        "question": "A shirt costs $80 after a 20% discount. What was the original price?",
        "answer": "100",
        "hint": "80 = original × 0.8, so original = 80/0.8",
        "difficulty": "easy",
    },
    {
        "question": "If you flip a fair coin 3 times, how many possible outcomes are there?",
        "answer": "8",
        "hint": "each flip has 2 outcomes, so 2^3 total",
        "difficulty": "hard",
    },
    {
        "question": "What is the sum of interior angles of a hexagon in degrees?",
        "answer": "720",
        "hint": "(n-2) × 180 where n=6, so 4 × 180",
        "difficulty": "hard",
    },
    {
        "question": "A car depreciates 15% per year. After 1 year a $20000 car is worth how much?",
        "answer": "17000",
        "hint": "20000 × (1 - 0.15) = 20000 × 0.85",
        "difficulty": "hard",
    },
    {
        "question": "How many prime numbers are there between 1 and 20?",
        "answer": "8",
        "hint": "primes: 2, 3, 5, 7, 11, 13, 17, 19",
        "difficulty": "hard",
    },
]

# Target tokens: the first token of each correct answer
# We measure P(answer_token) in both conditions
ANSWER_TOKENS = list(set(item["answer"] for item in ITEMS))


@register_probe
class ConsistencyLogprobProbe(BaseProbe):
    name = "consistency_logprob"
    description = "Reasoning-output consistency via logprobs — alignment circuits"
    max_items: int | None = 8

    def run(self, model) -> dict:
        items = self._limit(ITEMS)

        # Build two prompt sets: direct and with reasoning context
        direct_prompts = []
        context_prompts = []
        for item in items:
            direct_prompts.append(
                f"Answer with just the number: {item['question']}")
            context_prompts.append(
                f"Given: {item['hint']}. Answer with just the number: {item['question']}")

        # Get all answer tokens we need to check
        all_targets = ANSWER_TOKENS

        # Batch logprobs for both conditions
        if hasattr(model, 'get_logprobs_batch') and len(direct_prompts) > 1:
            direct_logprobs = model.get_logprobs_batch(direct_prompts, all_targets)
            context_logprobs = model.get_logprobs_batch(context_prompts, all_targets)
        else:
            direct_logprobs = [model.get_logprobs(p, all_targets) for p in direct_prompts]
            context_logprobs = [model.get_logprobs(p, all_targets) for p in context_prompts]

        easy_scores = []
        hard_scores = []
        easy_pcorrect = []
        hard_pcorrect = []
        item_results = []

        for item, d_lp, c_lp in zip(items, direct_logprobs, context_logprobs):
            answer = item["answer"]
            difficulty = item["difficulty"]

            # P(correct) in each condition
            d_lp_val = d_lp.get(answer, -100)
            c_lp_val = c_lp.get(answer, -100)
            p_direct = math.exp(d_lp_val) if d_lp_val > -100 else 0.0
            p_context = math.exp(c_lp_val) if c_lp_val > -100 else 0.0

            # Score: does reasoning context help?
            # Normalized to [0, 1]: 0.5 = no change, >0.5 = context helps
            delta = p_context - p_direct
            score = max(0.0, min(1.0, 0.5 + delta))

            # p_correct = average of both conditions
            p_correct = (p_direct + p_context) / 2

            if difficulty == "easy":
                easy_scores.append(score)
                easy_pcorrect.append(p_correct)
            else:
                hard_scores.append(score)
                hard_pcorrect.append(p_correct)

            if self.log_responses:
                item_results.append({
                    "difficulty": difficulty,
                    "question": item["question"][:80],
                    "answer": answer,
                    "p_direct": round(p_direct, 4),
                    "p_context": round(p_context, 4),
                    "delta": round(delta, 4),
                    "score": round(score, 4),
                })

        all_scores = easy_scores + hard_scores
        all_pcorrect = easy_pcorrect + hard_pcorrect

        return {
            "score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "easy_score": sum(easy_scores) / len(easy_scores) if easy_scores else 0.0,
            "hard_score": sum(hard_scores) / len(hard_scores) if hard_scores else 0.0,
            "p_correct": sum(all_pcorrect) / len(all_pcorrect) if all_pcorrect else 0.0,
            "p_correct_easy": sum(easy_pcorrect) / len(easy_pcorrect) if easy_pcorrect else 0.0,
            "p_correct_hard": sum(hard_pcorrect) / len(hard_pcorrect) if hard_pcorrect else 0.0,
            "n_easy": len(easy_scores),
            "n_hard": len(hard_scores),
            "item_results": item_results if item_results else None,
        }
