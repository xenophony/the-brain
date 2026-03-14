"""
Calibration analysis for baseline probe scores.

Reads baseline_scores.json and flags:
  - Ceiling effects (any model > threshold on a probe)
  - Floor effects (all models < threshold on a probe)
  - Narrow dynamic range (max - min across models < threshold)
  - High inter-probe correlation (non-orthogonal probes)

Generates CALIBRATION_REPORT.md with all findings.
"""

import json
import math
from pathlib import Path


def _get_all_probe_names(baseline_scores: dict) -> list[str]:
    """Collect all unique probe names across all models."""
    probes = set()
    for model_data in baseline_scores.values():
        probes.update(model_data.keys())
    return sorted(probes)


def _get_scores_vector(baseline_scores: dict, probe_name: str) -> list[float]:
    """Get scores for one probe across all models (skipping errors/missing)."""
    scores = []
    for model_data in baseline_scores.values():
        entry = model_data.get(probe_name)
        if entry and entry.get("error") is None:
            scores.append(entry["score"])
    return scores


def check_ceiling_effects(baseline_scores: dict, threshold: float = 0.85) -> list[dict]:
    """Flag probes where ANY model scores above threshold.

    Returns list of {"probe": str, "model": str, "score": float, "threshold": float}.
    """
    findings = []
    for model_name, model_data in baseline_scores.items():
        for probe_name, entry in model_data.items():
            if entry.get("error") is not None:
                continue
            if entry["score"] > threshold:
                findings.append({
                    "probe": probe_name,
                    "model": model_name,
                    "score": entry["score"],
                    "threshold": threshold,
                })
    return findings


def check_floor_effects(baseline_scores: dict, threshold: float = 0.20) -> list[dict]:
    """Flag probes where ALL models score below threshold.

    Returns list of {"probe": str, "max_score": float, "threshold": float}.
    """
    probes = _get_all_probe_names(baseline_scores)
    findings = []
    for probe_name in probes:
        scores = _get_scores_vector(baseline_scores, probe_name)
        if not scores:
            continue
        max_score = max(scores)
        if max_score < threshold:
            findings.append({
                "probe": probe_name,
                "max_score": max_score,
                "threshold": threshold,
            })
    return findings


def check_dynamic_range(baseline_scores: dict, min_range: float = 0.25) -> list[dict]:
    """For each probe: max_score - min_score across models. Want > min_range.

    Returns list of {"probe": str, "range": float, "min": float, "max": float,
                      "ok": bool, "min_range": float}.
    """
    probes = _get_all_probe_names(baseline_scores)
    results = []
    for probe_name in probes:
        scores = _get_scores_vector(baseline_scores, probe_name)
        if len(scores) < 2:
            continue
        score_min = min(scores)
        score_max = max(scores)
        score_range = score_max - score_min
        results.append({
            "probe": probe_name,
            "range": round(score_range, 4),
            "min": round(score_min, 4),
            "max": round(score_max, 4),
            "ok": score_range >= min_range,
            "min_range": min_range,
        })
    return results


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation between two equal-length lists."""
    n = len(x)
    if n < 3:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def check_orthogonality(
    baseline_scores: dict, threshold: float = 0.7
) -> tuple[list[dict], list[dict]]:
    """Pairwise Pearson correlation of probe scores across models.

    Returns (all_pairs, flagged_pairs) where flagged pairs have |r| > threshold.
    Each pair: {"probe_a": str, "probe_b": str, "correlation": float, "flagged": bool}.
    """
    probes = _get_all_probe_names(baseline_scores)
    model_names = list(baseline_scores.keys())

    all_pairs = []
    flagged = []

    for i in range(len(probes)):
        for j in range(i + 1, len(probes)):
            pa, pb = probes[i], probes[j]
            # Build aligned score vectors (only models that have both probes)
            xs, ys = [], []
            for model in model_names:
                ea = baseline_scores.get(model, {}).get(pa)
                eb = baseline_scores.get(model, {}).get(pb)
                if (ea and ea.get("error") is None and
                        eb and eb.get("error") is None):
                    xs.append(ea["score"])
                    ys.append(eb["score"])
            if len(xs) < 3:
                continue
            r = _pearson_correlation(xs, ys)
            entry = {
                "probe_a": pa,
                "probe_b": pb,
                "correlation": round(r, 4),
                "flagged": abs(r) > threshold,
            }
            all_pairs.append(entry)
            if abs(r) > threshold:
                flagged.append(entry)

    return all_pairs, flagged


def generate_calibration_report(
    baseline_scores: dict, output_path: str
) -> str:
    """Generate CALIBRATION_REPORT.md with all findings.

    Returns the report text.
    """
    lines = ["# Calibration Report\n"]
    lines.append(f"Models evaluated: {', '.join(baseline_scores.keys())}\n")

    probes = _get_all_probe_names(baseline_scores)
    lines.append(f"Probes evaluated: {len(probes)}\n")

    # --- Score summary table ---
    lines.append("\n## Score Summary\n")
    lines.append("| Probe | " + " | ".join(baseline_scores.keys()) + " |")
    lines.append("|-------|" + "|".join(["-------"] * len(baseline_scores)) + "|")
    for probe in probes:
        row = [probe]
        for model in baseline_scores:
            entry = baseline_scores[model].get(probe)
            if entry and entry.get("error") is None:
                row.append(f"{entry['score']:.3f}")
            elif entry and entry.get("error"):
                row.append("ERR")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    # --- Ceiling effects ---
    ceilings = check_ceiling_effects(baseline_scores)
    lines.append("\n## Ceiling Effects (score > 0.85)\n")
    if ceilings:
        lines.append("These probes may be too easy -- consider harder items:\n")
        for c in ceilings:
            lines.append(f"- **{c['probe']}**: {c['model']} scored {c['score']:.3f}")
    else:
        lines.append("No ceiling effects detected.\n")

    # --- Floor effects ---
    floors = check_floor_effects(baseline_scores)
    lines.append("\n## Floor Effects (all scores < 0.20)\n")
    if floors:
        lines.append("These probes may be too hard or broken:\n")
        for f in floors:
            lines.append(f"- **{f['probe']}**: max score = {f['max_score']:.3f}")
    else:
        lines.append("No floor effects detected.\n")

    # --- Dynamic range ---
    ranges = check_dynamic_range(baseline_scores)
    lines.append("\n## Dynamic Range (want > 0.25)\n")
    narrow = [r for r in ranges if not r["ok"]]
    if narrow:
        lines.append("These probes have narrow range across models (may not differentiate):\n")
        for r in narrow:
            lines.append(f"- **{r['probe']}**: range = {r['range']:.3f} "
                         f"(min={r['min']:.3f}, max={r['max']:.3f})")
    else:
        lines.append("All probes have sufficient dynamic range.\n")
    lines.append("\nFull range table:\n")
    lines.append("| Probe | Min | Max | Range | OK |")
    lines.append("|-------|-----|-----|-------|----|")
    for r in ranges:
        ok_str = "Yes" if r["ok"] else "NO"
        lines.append(f"| {r['probe']} | {r['min']:.3f} | {r['max']:.3f} | "
                     f"{r['range']:.3f} | {ok_str} |")

    # --- Orthogonality ---
    all_pairs, flagged_pairs = check_orthogonality(baseline_scores)
    lines.append("\n## Orthogonality (|r| > 0.70 flagged)\n")
    if flagged_pairs:
        lines.append("These probe pairs are highly correlated and may be redundant:\n")
        for p in flagged_pairs:
            lines.append(f"- **{p['probe_a']}** vs **{p['probe_b']}**: r = {p['correlation']:.3f}")
    else:
        lines.append("No highly correlated probe pairs detected.\n")

    report = "\n".join(lines) + "\n"

    # Write report
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)

    return report
