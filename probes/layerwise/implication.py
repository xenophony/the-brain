"""Layerwise implication probe."""
from probes.implication_logprob.probe import ImplicationLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class ImplicationLayerwiseProbe(BaseLayerwiseProbe):
    name = "implication_layerwise"
    description = "Per-layer logical implication circuits"
    ITEMS = ImplicationLogprobProbe.ITEMS
    CHOICES = ImplicationLogprobProbe.CHOICES
