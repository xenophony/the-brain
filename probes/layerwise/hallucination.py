"""Layerwise hallucination probe."""
from probes.hallucination_logprob.probe import HallucinationLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class HallucinationLayerwiseProbe(BaseLayerwiseProbe):
    name = "hallucination_layerwise"
    description = "Per-layer hallucination/confabulation circuits"
    ITEMS = HallucinationLogprobProbe.ITEMS
    CHOICES = HallucinationLogprobProbe.CHOICES
