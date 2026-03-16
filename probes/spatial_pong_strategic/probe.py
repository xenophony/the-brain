"""
Spatial Pong Strategic probe — motion planning with paddle speed constraint.

Tests trajectory prediction AND reachability reasoning. The paddle has a limited
speed, so some ball positions are unreachable and the correct answer is 'stay'.
Hard items omit steps_to_arrival from the prompt.

8 easy (high paddle speed, always reachable) + 8 hard (low speed, some unreachable).

Output: 'up', 'down', or 'stay'.
Maps to: parietal lobe + prefrontal executive / motion planning.
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
    Hard items include at least 3 where the ball is unreachable.
    """
    rng = random.Random(seed)

    # --- Easy: paddle_speed=5, always reachable ---
    easy_items = []
    attempts = 0
    while len(easy_items) < 8 and attempts < 500:
        attempts += 1
        ball_x = rng.randint(5, 25)
        ball_y = rng.randint(5, 25)
        ball_dx = rng.randint(2, 5)
        ball_dy = rng.choice([-3, -2, -1, 0, 1, 2, 3])
        paddle_cy = rng.randint(10, 20)
        paddle_speed = 5

        action, steps, final_y = pong_oracle(
            ball_x, ball_y, ball_dx, ball_dy,
            PADDLE_X, paddle_cy, PADDLE_H, paddle_speed, COURT_H)
        if steps == 0:
            continue

        # Want a mix; prefer non-stay for easy items to test movement
        if action != "stay" or len(easy_items) >= 6:
            easy_items.append({
                "ball_x": ball_x, "ball_y": ball_y,
                "ball_dx": ball_dx, "ball_dy": ball_dy,
                "paddle_cy": paddle_cy, "paddle_speed": paddle_speed,
                "answer": action, "steps": steps,
                "final_y": round(final_y, 1),
            })

    # Supplement with stay scenarios if needed
    for _ in range(20):
        ball_x = rng.randint(5, 25)
        ball_y = rng.randint(5, 25)
        ball_dx = rng.randint(2, 5)
        ball_dy = rng.choice([-1, 0, 1])
        paddle_cy = rng.randint(10, 20)
        paddle_speed = 5
        action, steps, final_y = pong_oracle(
            ball_x, ball_y, ball_dx, ball_dy,
            PADDLE_X, paddle_cy, PADDLE_H, paddle_speed, COURT_H)
        if action == "stay" and steps > 0 and len(easy_items) < 8:
            easy_items.append({
                "ball_x": ball_x, "ball_y": ball_y,
                "ball_dx": ball_dx, "ball_dy": ball_dy,
                "paddle_cy": paddle_cy, "paddle_speed": paddle_speed,
                "answer": action, "steps": steps,
                "final_y": round(final_y, 1),
            })
    easy_items = easy_items[:8]

    # --- Hard: paddle_speed=1-2, at least 3 unreachable ---
    hard_items = []
    unreachable_count = 0
    attempts = 0
    while len(hard_items) < 8 and attempts < 2000:
        attempts += 1
        ball_x = rng.randint(5, 25)
        ball_y = rng.randint(5, 25)
        ball_dx = rng.randint(2, 5)
        ball_dy = rng.choice([-3, -2, -1, 0, 1, 2, 3])
        paddle_cy = rng.randint(10, 20)
        paddle_speed = rng.choice([1, 1, 2])

        action, steps, final_y = pong_oracle(
            ball_x, ball_y, ball_dx, ball_dy,
            PADDLE_X, paddle_cy, PADDLE_H, paddle_speed, COURT_H)
        if steps == 0:
            continue

        distance = abs(final_y - paddle_cy)
        is_within_paddle = distance <= PADDLE_H / 2
        max_reach = paddle_speed * steps
        is_unreachable = (not is_within_paddle) and (max_reach < distance - PADDLE_H / 2)

        if is_unreachable and unreachable_count < 3:
            hard_items.append({
                "ball_x": ball_x, "ball_y": ball_y,
                "ball_dx": ball_dx, "ball_dy": ball_dy,
                "paddle_cy": paddle_cy, "paddle_speed": paddle_speed,
                "answer": action, "steps": steps,
                "final_y": round(final_y, 1),
                "unreachable": True,
            })
            unreachable_count += 1
        elif not is_unreachable and not is_within_paddle and len(hard_items) - unreachable_count < 5:
            hard_items.append({
                "ball_x": ball_x, "ball_y": ball_y,
                "ball_dx": ball_dx, "ball_dy": ball_dy,
                "paddle_cy": paddle_cy, "paddle_speed": paddle_speed,
                "answer": action, "steps": steps,
                "final_y": round(final_y, 1),
                "unreachable": False,
            })

    return easy_items, hard_items


EASY_ITEMS, HARD_ITEMS = _generate_scenarios(seed=42)

# Oracle answers for manual verification (seed=42):
# Easy 0: ball=(25,8) vel=(2,2) paddle_cy=14 speed=5 -> up (steps=8, final_y=24.0)
# Easy 1: ball=(12,12) vel=(3,2) paddle_cy=11 speed=5 -> up (steps=10, final_y=28.0)
# Easy 2: ball=(22,7) vel=(5,-3) paddle_cy=10 speed=5 -> down (steps=4, final_y=5.0)
# Easy 3: ball=(5,22) vel=(3,2) paddle_cy=20 speed=5 -> down (steps=12, final_y=14.0)
# Easy 4: ball=(13,5) vel=(3,2) paddle_cy=16 speed=5 -> up (steps=9, final_y=23.0)
# Easy 5: ball=(15,13) vel=(3,-2) paddle_cy=15 speed=5 -> down (steps=9, final_y=5.0)
# Easy 6: ball=(16,24) vel=(4,3) paddle_cy=10 speed=5 -> up (steps=6, final_y=18.0)
# Easy 7: ball=(19,22) vel=(2,0) paddle_cy=11 speed=5 -> up (steps=11, final_y=22.0)
# Hard 0: ball=(25,14) vel=(3,-2) paddle_cy=15 speed=1 -> stay (steps=5, final_y=4.0) UNREACHABLE
# Hard 1: ball=(22,21) vel=(2,1) paddle_cy=15 speed=1 -> stay (steps=9, final_y=30.0) UNREACHABLE
# Hard 2: ball=(5,8) vel=(4,3) paddle_cy=14 speed=1 -> up (steps=9, final_y=25.0)
# Hard 3: ball=(6,12) vel=(2,-3) paddle_cy=17 speed=1 -> up (steps=17, final_y=21.0)
# Hard 4: ball=(22,9) vel=(3,2) paddle_cy=17 speed=2 -> up (steps=6, final_y=21.0)
# Hard 5: ball=(10,13) vel=(5,-2) paddle_cy=18 speed=2 -> stay (steps=6, final_y=1.0) UNREACHABLE
# Hard 6: ball=(11,14) vel=(5,2) paddle_cy=20 speed=1 -> up (steps=6, final_y=26.0)
# Hard 7: ball=(19,21) vel=(5,-3) paddle_cy=13 speed=1 -> down (steps=5, final_y=6.0)

PROMPT_TEMPLATE_EASY = """\
Ball: position=({ball_x}, {ball_y}), velocity=(dx={ball_dx}, dy={ball_dy})
Paddle: x={paddle_x}, center_y={paddle_cy}, height=6, speed={paddle_speed}
Court: width=60, height=30

The paddle moves at {paddle_speed} units per step.
The ball moves every step and reflects off walls.
The paddle spans center_y \u00b1 3.

Should the paddle move up, down, or stay?
You MUST respond with exactly one word: up, down, or stay. Nothing else."""

# Hard items omit steps_to_arrival
PROMPT_TEMPLATE_HARD = """\
Ball: position=({ball_x}, {ball_y}), velocity=(dx={ball_dx}, dy={ball_dy})
Paddle: x={paddle_x}, center_y={paddle_cy}, height=6, speed={paddle_speed}
Court: width=60, height=30

The paddle moves at {paddle_speed} units per step.
The ball moves every step and reflects off walls.
The paddle spans center_y \u00b1 3.

Should the paddle move up, down, or stay?
You MUST respond with exactly one word: up, down, or stay. Nothing else."""


def score_pong_strategic(response: str, expected: str) -> float:
    """Extract last directional word from response and compare to expected."""
    cleaned = response.strip().lower()
    last_word = None
    last_pos = -1
    for word in ["stay", "up", "down"]:
        pos = cleaned.rfind(word)
        if pos > last_pos:
            last_pos = pos
            last_word = word
    if last_word is None:
        return 0.0
    return 1.0 if last_word == expected else 0.0


@register_probe
class SpatialPongStrategicProbe(BaseProbe):
    name = "spatial_pong_strategic"
    description = "Pong motion planning with speed constraint - parietal + prefrontal executive"

    def run(self, model) -> dict:
        def _prompt_easy(item):
            return PROMPT_TEMPLATE_EASY.format(
                ball_x=item["ball_x"], ball_y=item["ball_y"],
                ball_dx=item["ball_dx"], ball_dy=item["ball_dy"],
                paddle_x=PADDLE_X, paddle_cy=item["paddle_cy"],
                paddle_speed=item["paddle_speed"])

        def _prompt_hard(item):
            return PROMPT_TEMPLATE_HARD.format(
                ball_x=item["ball_x"], ball_y=item["ball_y"],
                ball_dx=item["ball_dx"], ball_dy=item["ball_dy"],
                paddle_x=PADDLE_X, paddle_cy=item["paddle_cy"],
                paddle_speed=item["paddle_speed"])

        easy_scores, easy_results = self._run_items(
            model, self._limit(EASY_ITEMS),
            prompt_fn=_prompt_easy,
            score_fn=lambda resp, item: score_pong_strategic(resp, item["answer"]),
            max_new_tokens=5, difficulty="easy")

        hard_scores, hard_results = self._run_items(
            model, self._limit(HARD_ITEMS),
            prompt_fn=_prompt_hard,
            score_fn=lambda resp, item: score_pong_strategic(resp, item["answer"]),
            max_new_tokens=5, difficulty="hard")

        item_results = (easy_results + hard_results) if self.log_responses else None
        return self._make_result(easy_scores, hard_scores, item_results)
