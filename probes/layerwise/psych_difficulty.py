"""Layerwise psych difficulty probe."""
from probes.psych_difficulty_logprob.probe import PsychDifficultyLogprobProbe
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class PsychDifficultyLayerwiseProbe(BaseLayerwiseProbe):
    name = "psych_difficulty_layerwise"
    description = "Per-layer psycholinguistic difficulty circuits"
    ITEMS = PsychDifficultyLogprobProbe.ITEMS
    CHOICES = PsychDifficultyLogprobProbe.CHOICES
