"""Layerwise temporal probe."""
from probes.temporal_logprob.probe import TemporalLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class TemporalLayerwiseProbe(BaseLayerwiseProbe):
    name = "temporal_layerwise"
    description = "Per-layer temporal reasoning circuits"
    ITEMS = TemporalLogprobProbe.ITEMS
    CHOICES = TemporalLogprobProbe.CHOICES
