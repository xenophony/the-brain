"""Layerwise causal reasoning probe."""
from probes.causal_logprob.probe import CausalLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class CausalLayerwiseProbe(BaseLayerwiseProbe):
    name = "causal_layerwise"
    description = "Per-layer causal reasoning — world model circuits"
    ITEMS = CausalLogprobProbe.ITEMS
    CHOICES = CausalLogprobProbe.CHOICES
