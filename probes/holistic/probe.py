"""
Holistic/analogy probe — complete conceptual analogies.

Output: single word. Scored by exact/semantic match.
Maps to: default mode network / associative thinking circuits.
"""

from probes.registry import BaseProbe, register_probe


@register_probe
class HolisticProbe(BaseProbe):
    name = "holistic"
    description = "Analogy completion — default mode network circuits"

    def run(self, model) -> float:
        raise NotImplementedError("Holistic probe not yet implemented")
