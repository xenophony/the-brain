"""Layerwise language probe."""
from probes.language_logprob.probe import LanguageLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class LanguageLayerwiseProbe(BaseLayerwiseProbe):
    name = "language_layerwise"
    description = "Per-layer grammaticality circuits"
    ITEMS = LanguageLogprobProbe.ITEMS
    CHOICES = LanguageLogprobProbe.CHOICES
