"""Layerwise pong strategic probe."""
from probes.pong_strategic_logprob.probe import PongStrategicLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class PongStrategicLayerwiseProbe(BaseLayerwiseProbe):
    name = "pong_strategic_layerwise"
    description = "Per-layer strategic spatial reasoning circuits"
    ITEMS = PongStrategicLogprobProbe.ITEMS
    CHOICES = PongStrategicLogprobProbe.CHOICES
