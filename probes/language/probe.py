"""
Language/syntax probe — grammaticality judgment.

Output: single word "grammatical" or "ungrammatical".
Maps to: Broca/Wernicke language circuits.
"""

from probes.registry import BaseProbe, register_probe


@register_probe
class LanguageProbe(BaseProbe):
    name = "language"
    description = "Syntactic anomaly detection — Broca/Wernicke circuits"

    def run(self, model) -> float:
        raise NotImplementedError("Language probe not yet implemented")
