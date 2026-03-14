"""
Spatial Pong Simple probe — pure trajectory prediction, no speed constraint.

The paddle can always reach the ball. Tests whether the model can predict
where the ball will arrive after bouncing off walls and determine which
direction the paddle should move.

8 easy (direct path, no wall bounces) + 8 hard (1-2 wall bounces).

Output: 'up', 'down', or 'stay'.
Maps to: parietal lobe / visual-spatial trajectory prediction.
"""

import random
from probes.registry import BaseProbe, register_probe
from probes.pong_oracle import pong_oracle


COURT_W = 60
COURT_H = 30
PADDLE_X = 40
PADDLE_H = 6


def _generate_scenarios(seed=42):
    """Generate all 16 scenarios deterministically.

    Returns (easy_items, hard_items) where each item is a dict with
    ball/paddle parameters, oracle answer, steps, and final_y.
    """
    rng = random.Random(seed)

    # --- Easy: no wall bounces, ball_dy=0 ---
    easy_items = []
    attempts = 0
    while len(easy_items) < 8 and attempts < 500:
        attempts += 1
        ball_x = rng.randint(5, 25)
        ball_y = rng.randint(5, 25)
        ball_dx = rng.randint(2, 5)
        ball_dy = rng.choice([0, 0, 0, 1, -1])
        paddle_cy = rng.randint(10, 20)

        action, steps, final_y = pong_oracle(
            ball_x, ball_y, ball_dx, ball_dy,
            PADDLE_X, paddle_cy, PADDLE_H, 999, COURT_H)
        if steps == 0:
            continue

        # Verify no bounce
        x, y, dy = float(ball_x), float(ball_y), float(ball_dy)
        dx = float(ball_dx)
        bounced = False
        while x < PADDLE_X:
            x += dx
            y += dy
            if y <= 0 or y >= COURT_H:
                bounced = True
                break
        if bounced:
            continue

        easy_items.append({
            "ball_x": ball_x, "ball_y": ball_y,
            "ball_dx": ball_dx, "ball_dy": ball_dy,
            "paddle_cy": paddle_cy,
            "answer": action, "steps": steps,
            "final_y": round(final_y, 1),
        })

    # --- Hard: 1-2 wall bounces ---
    hard_items = []
    attempts = 0
    while len(hard_items) < 8 and attempts < 1000:
        attempts += 1
        ball_x = rng.randint(5, 25)
        ball_y = rng.randint(5, 25)
        ball_dx = rng.randint(2, 5)
        ball_dy = rng.choice([-3, -2, 2, 3])
        paddle_cy = rng.randint(10, 20)

        action, steps, final_y = pong_oracle(
            ball_x, ball_y, ball_dx, ball_dy,
            PADDLE_X, paddle_cy, PADDLE_H, 999, COURT_H)
        if steps == 0:
            continue

        # Count bounces
        x, y, dy = float(ball_x), float(ball_y), float(ball_dy)
        dx = float(ball_dx)
        bounces = 0
        while x < PADDLE_X:
            x += dx
            y += dy
            if y <= 0:
                y = abs(y)
                dy = abs(dy)
                bounces += 1
            elif y >= COURT_H:
                y = 2 * COURT_H - y
                dy = -abs(dy)
                bounces += 1
        if 1 <= bounces <= 2:
            hard_items.append({
                "ball_x": ball_x, "ball_y": ball_y,
                "ball_dx": ball_dx, "ball_dy": ball_dy,
                "paddle_cy": paddle_cy,
                "answer": action, "steps": steps,
                "final_y": round(final_y, 1),
                "bounces": bounces,
            })

    return easy_items, hard_items


EASY_ITEMS, HARD_ITEMS = _generate_scenarios(seed=42)

# Oracle answers for manual verification (seed=42):
# Easy 0: ball=(25,8) vel=(2,0) paddle_cy=13 -> down (steps=8, final_y=8.0)
# Easy 1: ball=(23,18) vel=(2,0) paddle_cy=11 -> up (steps=9, final_y=18.0)
# Easy 2: ball=(25,22) vel=(5,0) paddle_cy=17 -> up (steps=3, final_y=22.0)
# Easy 3: ball=(23,13) vel=(2,0) paddle_cy=16 -> stay (steps=9, final_y=13.0)
# Easy 4: ball=(15,13) vel=(3,0) paddle_cy=15 -> stay (steps=9, final_y=13.0)
# Easy 5: ball=(8,7) vel=(5,0) paddle_cy=15 -> down (steps=7, final_y=7.0)
# Easy 6: ball=(16,24) vel=(4,0) paddle_cy=17 -> up (steps=6, final_y=24.0)
# Easy 7: ball=(22,8) vel=(5,0) paddle_cy=18 -> down (steps=4, final_y=8.0)
# Hard 0: ball=(6,12) vel=(4,-3) paddle_cy=13 -> stay (steps=9, final_y=15.0, bounces=1)
# Hard 1: ball=(8,17) vel=(4,3) paddle_cy=20 -> stay (steps=8, final_y=19.0, bounces=1)
# Hard 2: ball=(13,25) vel=(2,-2) paddle_cy=18 -> down (steps=14, final_y=3.0, bounces=1)
# Hard 3: ball=(25,22) vel=(3,2) paddle_cy=10 -> up (steps=5, final_y=28.0, bounces=1)
# Hard 4: ball=(7,11) vel=(4,-2) paddle_cy=20 -> down (steps=9, final_y=7.0, bounces=1)
# Hard 5: ball=(9,12) vel=(4,3) paddle_cy=19 -> up (steps=8, final_y=24.0, bounces=1)
# Hard 6: ball=(17,16) vel=(3,-2) paddle_cy=18 -> down (steps=8, final_y=0.0, bounces=1)
# Hard 7: ball=(20,7) vel=(2,-3) paddle_cy=12 -> up (steps=10, final_y=23.0, bounces=1)

PROMPT_TEMPLATE = """\
Ball: position=({ball_x}, {ball_y}), velocity=(dx={ball_dx}, dy={ball_dy})
Paddle: x={paddle_x}, center_y={paddle_cy}, height=6
Court: width=60, height=30 (walls at y=0 and y=30)
Ball arrives in {steps} steps.

The ball reflects off top and bottom walls.
The paddle spans center_y \u00b1 3.

Should the paddle move up, down, or stay?
Answer with only: up, down, or stay"""


def score_pong_simple(response: str, expected: str) -> float:
    """Exact match scoring: 1.0 if correct, 0.0 otherwise."""
    cleaned = response.strip().lower()
    # Extract the action word
    for word in ["stay", "up", "down"]:
        if word in cleaned:
            return 1.0 if word == expected else 0.0
    return 0.0


@register_probe
class SpatialPongSimpleProbe(BaseProbe):
    name = "spatial_pong_simple"
    description = "Pong trajectory prediction (no speed constraint) - parietal lobe / visual-spatial"

    def run(self, model) -> dict:
        easy_scores = []
        hard_scores = []
        item_results = [] if self.log_responses else None

        for item in EASY_ITEMS:
            prompt = PROMPT_TEMPLATE.format(
                ball_x=item["ball_x"], ball_y=item["ball_y"],
                ball_dx=item["ball_dx"], ball_dy=item["ball_dy"],
                paddle_x=PADDLE_X, paddle_cy=item["paddle_cy"],
                steps=item["steps"])
            response = model.generate_short(prompt, max_new_tokens=10, temperature=0.0)
            score = score_pong_simple(response, item["answer"])
            easy_scores.append(score)
            if item_results is not None:
                item_results.append({
                    "difficulty": "easy",
                    "expected": item["answer"],
                    "response": response[:200],
                    "score": score,
                })

        for item in HARD_ITEMS:
            prompt = PROMPT_TEMPLATE.format(
                ball_x=item["ball_x"], ball_y=item["ball_y"],
                ball_dx=item["ball_dx"], ball_dy=item["ball_dy"],
                paddle_x=PADDLE_X, paddle_cy=item["paddle_cy"],
                steps=item["steps"])
            response = model.generate_short(prompt, max_new_tokens=10, temperature=0.0)
            score = score_pong_simple(response, item["answer"])
            hard_scores.append(score)
            if item_results is not None:
                item_results.append({
                    "difficulty": "hard",
                    "expected": item["answer"],
                    "response": response[:200],
                    "score": score,
                })

        return self._make_result(easy_scores, hard_scores, item_results)
