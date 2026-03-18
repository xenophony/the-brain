"""Layerwise judgement probe."""
from probes.judgement_logprob.probe import JudgementLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class JudgementLayerwiseProbe(BaseLayerwiseProbe):
    name = "judgement_layerwise"
    description = "Per-layer judgement/evaluation circuits"
    ITEMS = JudgementLogprobProbe.ITEMS
    CHOICES = JudgementLogprobProbe.CHOICES
