"""
Planning probe — order steps to achieve a goal.

Output: ordered letter sequence. Scored by pairwise ordering correctness.
Maps to: prefrontal executive / planning circuits.
"""

from probes.registry import BaseProbe, register_probe


@register_probe
class PlanningProbe(BaseProbe):
    name = "planning"
    description = "Step ordering / planning — prefrontal executive circuits"

    def run(self, model) -> float:
        raise NotImplementedError("Planning probe not yet implemented")
