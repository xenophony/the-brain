"""Layerwise consistency probe."""
from probes.consistency_logprob.probe import ConsistencyLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class ConsistencyLayerwiseProbe(BaseLayerwiseProbe):
    name = "consistency_layerwise"
    description = "Per-layer internal consistency circuits"
    ITEMS = ConsistencyLogprobProbe.ITEMS
    CHOICES = ConsistencyLogprobProbe.CHOICES
