"""
Tool selection probe — pick the best tool for a task.

Output: single tool name. Scored by exact match.
Maps to: frontal lobe / executive function circuits.
"""

from probes.registry import BaseProbe, register_probe


@register_probe
class ToolUseProbe(BaseProbe):
    name = "tool_use"
    description = "Tool selection routing — frontal lobe circuits"

    def run(self, model) -> float:
        raise NotImplementedError("Tool use probe not yet implemented")
