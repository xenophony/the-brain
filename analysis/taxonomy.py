"""
Probe taxonomy — hierarchical groupings of probes into cognitive domains.

Each probe can belong to multiple domains (e.g. sycophancy is both SOCIAL and SAFETY).
Domains map loosely to brain region clusters identified via (i,j) sweep heatmaps.
"""

REASONING = ["math", "planning", "counterfactual", "temporal"]
SOCIAL = ["eq", "sycophancy", "holistic"]
LANGUAGE = ["language", "abstraction", "instruction"]
SPATIAL = ["spatial"]
MEMORY = ["factual", "metacognition"]
EXECUTION = ["code", "tool_use"]
SAFETY = ["hallucination", "sycophancy", "consistency", "instruction"]
STABILITY = ["noise_robustness"]

ALL_DOMAINS = {
    "REASONING": REASONING,
    "SOCIAL": SOCIAL,
    "LANGUAGE": LANGUAGE,
    "SPATIAL": SPATIAL,
    "MEMORY": MEMORY,
    "EXECUTION": EXECUTION,
    "SAFETY": SAFETY,
    "STABILITY": STABILITY,
}


def get_domain(probe_name: str) -> list[str]:
    """Return which domains a probe belongs to.

    Args:
        probe_name: Name of the probe (e.g. "math", "sycophancy").

    Returns:
        List of domain names the probe belongs to. Empty if not found.
    """
    return [domain for domain, probes in ALL_DOMAINS.items() if probe_name in probes]


def get_probes_in_domain(domain: str) -> list[str]:
    """Return probes in a domain.

    Args:
        domain: Domain name (e.g. "REASONING", "SAFETY").

    Returns:
        List of probe names. Empty if domain not found.
    """
    return list(ALL_DOMAINS.get(domain, []))


def get_all_probe_names() -> list[str]:
    """Return flat unique list of all probe names across all domains."""
    seen = set()
    result = []
    for probes in ALL_DOMAINS.values():
        for p in probes:
            if p not in seen:
                seen.add(p)
                result.append(p)
    return result
