"""
Emotional intelligence probe — EQ-Bench style intensity estimation.

Output: single digit 0-9. Scored with partial credit.
Maps to: limbic system / emotional processing circuits.
"""

from probes.registry import BaseProbe, register_probe


@register_probe
class EQProbe(BaseProbe):
    name = "eq"
    description = "Emotional intensity estimation — limbic system circuits"

    def run(self, model) -> float:
        raise NotImplementedError("EQ probe not yet implemented")
