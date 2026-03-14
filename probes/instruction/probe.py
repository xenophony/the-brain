"""
Instruction following probe — comply with multiple explicit constraints.

Output: short text. Scored by fraction of constraints satisfied.
Maps to: PFC / working memory circuits.
"""

from probes.registry import BaseProbe, register_probe


@register_probe
class InstructionProbe(BaseProbe):
    name = "instruction"
    description = "Multi-constraint instruction following — working memory circuits"

    def run(self, model) -> float:
        raise NotImplementedError("Instruction probe not yet implemented")
