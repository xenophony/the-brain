"""
Pong strategic logprob probe — motion planning via logprobs.

Logprob variant of the spatial_pong_strategic probe. Zero decode steps.
"""

from probes.registry import BaseLogprobProbe, register_probe
from probes.spatial_pong_strategic.probe import (
    EASY_ITEMS, HARD_ITEMS, PROMPT_TEMPLATE_EASY, PROMPT_TEMPLATE_HARD, PADDLE_X,
)

CHOICES = ["up", "down", "stay"]


def _format_easy(item):
    return PROMPT_TEMPLATE_EASY.format(
        ball_x=item["ball_x"], ball_y=item["ball_y"],
        ball_dx=item["ball_dx"], ball_dy=item["ball_dy"],
        paddle_x=PADDLE_X, paddle_cy=item["paddle_cy"],
        paddle_speed=item["paddle_speed"],
    )


def _format_hard(item):
    return PROMPT_TEMPLATE_HARD.format(
        ball_x=item["ball_x"], ball_y=item["ball_y"],
        ball_dx=item["ball_dx"], ball_dy=item["ball_dy"],
        paddle_x=PADDLE_X, paddle_cy=item["paddle_cy"],
        paddle_speed=item["paddle_speed"],
    )


ITEMS = [
    {
        "prompt": _format_easy(item),
        "answer": item["answer"].lower(),
        "difficulty": "easy",
    }
    for item in EASY_ITEMS
] + [
    {
        "prompt": _format_hard(item),
        "answer": item["answer"].lower(),
        "difficulty": "hard",
    }
    for item in HARD_ITEMS
]


@register_probe
class PongStrategicLogprobProbe(BaseLogprobProbe):
    name = "pong_strategic_logprob"
    description = "Pong motion planning via logprobs — parietal + prefrontal executive"
    ITEMS = ITEMS
    CHOICES = CHOICES
