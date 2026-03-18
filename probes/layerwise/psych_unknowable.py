"""Layerwise psych unknowable probe."""
from probes.psych_unknowable_logprob.probe import PsychUnknowableLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class PsychUnknowableLayerwiseProbe(BaseLayerwiseProbe):
    name = "psych_unknowable_layerwise"
    description = "Per-layer psycholinguistic unknowable circuits"
    ITEMS = PsychUnknowableLogprobProbe.ITEMS
    CHOICES = PsychUnknowableLogprobProbe.CHOICES
