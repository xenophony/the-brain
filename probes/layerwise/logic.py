"""Layerwise logic probe."""
from probes.logic_logprob.probe import LogicLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class LogicLayerwiseProbe(BaseLayerwiseProbe):
    name = "logic_layerwise"
    description = "Per-layer logical reasoning circuits"
    ITEMS = LogicLogprobProbe.ITEMS
    CHOICES = LogicLogprobProbe.CHOICES
