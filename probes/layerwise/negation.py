"""Layerwise negation probe."""
from probes.negation_logprob.probe import NegationLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class NegationLayerwiseProbe(BaseLayerwiseProbe):
    name = "negation_layerwise"
    description = "Per-layer negation/inhibition circuits"
    ITEMS = NegationLogprobProbe.ITEMS
    CHOICES = NegationLogprobProbe.CHOICES
