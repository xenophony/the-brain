"""Layerwise code probe."""
from probes.code_logprob.probe import CodeLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class CodeLayerwiseProbe(BaseLayerwiseProbe):
    name = "code_layerwise"
    description = "Per-layer code understanding circuits"
    ITEMS = CodeLogprobProbe.ITEMS
    CHOICES = CodeLogprobProbe.CHOICES
