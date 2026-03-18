"""Layerwise psych urgency probe."""
from probes.psych_urgency_logprob.probe import PsychUrgencyLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class PsychUrgencyLayerwiseProbe(BaseLayerwiseProbe):
    name = "psych_urgency_layerwise"
    description = "Per-layer psycholinguistic urgency circuits"
    ITEMS = PsychUrgencyLogprobProbe.ITEMS
    CHOICES = PsychUrgencyLogprobProbe.CHOICES
