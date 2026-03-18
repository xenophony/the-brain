"""Layerwise psych conflict probe."""
from probes.psych_conflict_logprob.probe import PsychConflictLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class PsychConflictLayerwiseProbe(BaseLayerwiseProbe):
    name = "psych_conflict_layerwise"
    description = "Per-layer psycholinguistic conflict circuits"
    ITEMS = PsychConflictLogprobProbe.ITEMS
    CHOICES = PsychConflictLogprobProbe.CHOICES
