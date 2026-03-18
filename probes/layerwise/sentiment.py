"""Layerwise sentiment probe."""
from probes.sentiment_logprob.probe import SentimentLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class SentimentLayerwiseProbe(BaseLayerwiseProbe):
    name = "sentiment_layerwise"
    description = "Per-layer sentiment classification circuits"
    ITEMS = SentimentLogprobProbe.ITEMS
    CHOICES = SentimentLogprobProbe.CHOICES
