"""Layerwise routing probe."""
from probes.routing_logprob.probe import RoutingLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class RoutingLayerwiseProbe(BaseLayerwiseProbe):
    name = "routing_layerwise"
    description = "Per-layer domain routing circuits"
    ITEMS = RoutingLogprobProbe.ITEMS
    CHOICES = RoutingLogprobProbe.CHOICES
