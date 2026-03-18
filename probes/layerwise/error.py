"""Layerwise error detection probe."""
from probes.error_logprob.probe import ErrorLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class ErrorLayerwiseProbe(BaseLayerwiseProbe):
    name = "error_layerwise"
    description = "Per-layer error detection circuits"
    ITEMS = ErrorLogprobProbe.ITEMS
    CHOICES = ErrorLogprobProbe.CHOICES
