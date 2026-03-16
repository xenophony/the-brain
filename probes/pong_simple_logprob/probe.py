"""
Pong simple logprob probe — trajectory prediction via logprobs.

Logprob variant of the spatial_pong_simple probe. Zero decode steps.
"""

from probes.registry import BaseLogprobProbe, register_probe
from probes.spatial_pong_simple.probe import (
    EASY_ITEMS, HARD_ITEMS, PROMPT_TEMPLATE, PADDLE_X,
)

CHOICES = ["up", "down", "stay"]


def _format_prompt(item):
    return PROMPT_TEMPLATE.format(
        ball_x=item["ball_x"], ball_y=item["ball_y"],
        ball_dx=item["ball_dx"], ball_dy=item["ball_dy"],
        paddle_x=PADDLE_X, paddle_cy=item["paddle_cy"],
        steps=item["steps"],
    )


ITEMS = [
    {
        "prompt": _format_prompt(item),
        "answer": item["answer"].lower(),
        "difficulty": "easy",
    }
    for item in EASY_ITEMS
] + [
    {
        "prompt": _format_prompt(item),
        "answer": item["answer"].lower(),
        "difficulty": "hard",
    }
    for item in HARD_ITEMS
]


@register_probe
class PongSimpleLogprobProbe(BaseLogprobProbe):
    name = "pong_simple_logprob"
    description = "Pong trajectory prediction via logprobs — parietal/visual-spatial"
    ITEMS = ITEMS
    CHOICES = CHOICES
