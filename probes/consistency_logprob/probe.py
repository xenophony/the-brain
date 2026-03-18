"""
Consistency logprob proxy — answer verification circuits.

Measures P("yes") when the model is asked to verify correct vs incorrect
answers to its own questions. Tests self-monitoring: can the model
distinguish right from wrong answers?

Two items per question:
  - Correct claim: "A train goes 60km in 30min. Is its speed 120 km/h?" → yes
  - Wrong claim:   "A train goes 60km in 30min. Is its speed 90 km/h?" → no

Score = P(correct_response) averaged across all items.
Layers that increase P("yes") on correct claims AND P("no") on wrong
claims = verification/consistency circuits.

Maps to: self-monitoring / reasoning verification circuits.
"""

from probes.registry import BaseLogprobProbe, register_probe

ITEMS = [
    # Easy: correct claims (answer = yes)
    {"prompt": "A train travels 60 km in 30 minutes. Is its speed 120 km/h?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "15% of 200 is 30. Is this correct?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "If 3 workers build a wall in 12 hours, 6 workers take 6 hours. Is this correct?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "A shirt costs $80 after a 20% discount. The original price was $100. Is this correct?", "answer": "yes", "difficulty": "easy"},
    # Easy: wrong claims (answer = no)
    {"prompt": "A train travels 60 km in 30 minutes. Is its speed 90 km/h?", "answer": "no", "difficulty": "easy"},
    {"prompt": "15% of 200 is 45. Is this correct?", "answer": "no", "difficulty": "easy"},
    {"prompt": "If 3 workers build a wall in 12 hours, 6 workers take 9 hours. Is this correct?", "answer": "no", "difficulty": "easy"},
    {"prompt": "A shirt costs $80 after a 20% discount. The original price was $120. Is this correct?", "answer": "no", "difficulty": "easy"},
    # Hard: correct claims with tricky numbers
    {"prompt": "Flipping a fair coin 3 times gives 8 possible outcomes. Is this correct?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "The interior angles of a hexagon sum to 720 degrees. Is this correct?", "answer": "yes", "difficulty": "hard"},
    # Hard: wrong claims with plausible-looking wrong answers
    {"prompt": "Flipping a fair coin 3 times gives 6 possible outcomes. Is this correct?", "answer": "no", "difficulty": "hard"},
    {"prompt": "The interior angles of a hexagon sum to 600 degrees. Is this correct?", "answer": "no", "difficulty": "hard"},
    # --- Added items for convergence reliability ---
    # Easy: correct claims
    {"prompt": "A rectangle with sides 5 and 3 has an area of 15. Is this correct?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "Half of 80 is 40. Is this correct?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "If a car travels 60 mph for 2 hours, it covers 120 miles. Is this correct?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "A dozen eggs is 12. Is this correct?", "answer": "yes", "difficulty": "easy"},
    {"prompt": "25% of 200 is 50. Is this correct?", "answer": "yes", "difficulty": "easy"},
    # Easy: wrong claims
    {"prompt": "A rectangle with sides 5 and 3 has an area of 8. Is this correct?", "answer": "no", "difficulty": "easy"},
    {"prompt": "Half of 80 is 45. Is this correct?", "answer": "no", "difficulty": "easy"},
    {"prompt": "If a car travels 60 mph for 2 hours, it covers 180 miles. Is this correct?", "answer": "no", "difficulty": "easy"},
    {"prompt": "A dozen eggs is 10. Is this correct?", "answer": "no", "difficulty": "easy"},
    {"prompt": "25% of 200 is 75. Is this correct?", "answer": "no", "difficulty": "easy"},
    # Hard: correct claims
    {"prompt": "The square root of 144 is 12. Is this correct?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "If you roll two dice, there are 36 possible outcomes. Is this correct?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "The sum of the first 10 positive integers is 55. Is this correct?", "answer": "yes", "difficulty": "hard"},
    {"prompt": "A circle with radius 7 has an area of 49*pi. Is this correct?", "answer": "yes", "difficulty": "hard"},
    # Hard: wrong claims
    {"prompt": "The square root of 144 is 14. Is this correct?", "answer": "no", "difficulty": "hard"},
    {"prompt": "If you roll two dice, there are 12 possible outcomes. Is this correct?", "answer": "no", "difficulty": "hard"},
    {"prompt": "The sum of the first 10 positive integers is 50. Is this correct?", "answer": "no", "difficulty": "hard"},
    {"prompt": "A circle with radius 7 has an area of 14*pi. Is this correct?", "answer": "no", "difficulty": "hard"},
]


@register_probe
class ConsistencyLogprobProbe(BaseLogprobProbe):
    name = "consistency_logprob"
    description = "Answer verification via logprobs — self-monitoring circuits"
    ITEMS = ITEMS
    CHOICES = ["yes", "no"]
