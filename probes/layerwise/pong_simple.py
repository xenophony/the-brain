"""Layerwise pong simple probe."""
from probes.pong_simple_logprob.probe import PongSimpleLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class PongSimpleLayerwiseProbe(BaseLayerwiseProbe):
    name = "pong_simple_layerwise"
    description = "Per-layer spatial trajectory prediction circuits"
    ITEMS = PongSimpleLogprobProbe.ITEMS
    CHOICES = PongSimpleLogprobProbe.CHOICES
