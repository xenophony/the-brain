"""
Factual recall probe — obscure but verifiable facts.

Output: number or single word. Scored by exact match.
Maps to: hippocampus / factual memory circuits.
"""

from probes.registry import BaseProbe, register_probe


@register_probe
class FactualProbe(BaseProbe):
    name = "factual"
    description = "Obscure fact recall — hippocampus circuits"

    def run(self, model) -> float:
        raise NotImplementedError("Factual probe not yet implemented")
