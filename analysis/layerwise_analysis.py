"""
Layerwise analysis module — post-processing for per-layer probe results.

Provides:
1. Convergence detection: where p(correct) stabilizes
2. Answer computation region: steepest p(correct) rise
3. Psych-correctness correlation
4. Psych convergence: where each psych signal peaks/troughs
5. Emotional signature of correctness: correct vs incorrect psych profiles
6. Entropy tracking: per-layer choice entropy
7. Surprise detection: dramatic layer-to-layer changes
8. Cross-probe comparison: identify shared circuit regions
"""

import json
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class LayerwiseReport:
    """Summary report for one probe's layerwise analysis."""
    probe_name: str
    n_items: int
    n_layers: int
    mean_convergence_layer: float
    computation_region: tuple
    surprise_layers: list
    psych_peak_layers: dict  # {category: peak_layer_idx}
    psych_trough_layers: dict  # {category: trough_layer_idx}


def analyze_convergence(mean_p_correct: list[float], threshold: float = 0.5,
                        stability_window: int = 3) -> dict:
    """Analyze where the answer converges across layers.

    Args:
        mean_p_correct: Per-layer mean p(correct) values.
        threshold: p(correct) threshold for convergence.
        stability_window: Number of consecutive layers above threshold-0.1.

    Returns:
        dict with convergence_layer, is_converged, stability_score
    """
    n = len(mean_p_correct)
    convergence_layer = None

    for k in range(n):
        if mean_p_correct[k] > threshold:
            stable = True
            for j in range(k, min(k + stability_window, n)):
                if mean_p_correct[j] < threshold - 0.1:
                    stable = False
                    break
            if stable:
                convergence_layer = k
                break

    # Stability score: fraction of post-convergence layers that stay above threshold
    stability_score = 0.0
    if convergence_layer is not None and convergence_layer < n:
        post = mean_p_correct[convergence_layer:]
        stability_score = sum(1 for p in post if p > threshold - 0.1) / len(post)

    return {
        "convergence_layer": convergence_layer,
        "is_converged": convergence_layer is not None,
        "stability_score": round(stability_score, 4),
    }


def find_computation_region(mean_p_correct: list[float]) -> dict:
    """Find the layer range with steepest p(correct) rise.

    Uses Kadane's algorithm on the difference sequence.

    Returns:
        dict with start, end, total_rise
    """
    if len(mean_p_correct) < 2:
        return {"start": 0, "end": 0, "total_rise": 0.0}

    diffs = [mean_p_correct[i] - mean_p_correct[i - 1]
             for i in range(1, len(mean_p_correct))]

    best_start = 0
    best_end = 0
    best_sum = 0.0
    cur_start = 0
    cur_sum = 0.0

    for i, d in enumerate(diffs):
        if d > 0:
            if cur_sum <= 0:
                cur_start = i
                cur_sum = d
            else:
                cur_sum += d
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_start = cur_start
                best_end = i + 1
        else:
            cur_sum = 0.0

    return {
        "start": best_start,
        "end": best_end,
        "total_rise": round(best_sum, 6),
    }


def detect_surprises(mean_p_correct: list[float], threshold: float = 0.10) -> list[dict]:
    """Find layers where p(correct) changes dramatically.

    Args:
        mean_p_correct: Per-layer mean p(correct).
        threshold: Minimum absolute change to count as surprise.

    Returns:
        List of {layer, delta, direction} dicts.
    """
    surprises = []
    for k in range(1, len(mean_p_correct)):
        delta = mean_p_correct[k] - mean_p_correct[k - 1]
        if abs(delta) > threshold:
            surprises.append({
                "layer": k,
                "delta": round(delta, 6),
                "direction": "rise" if delta > 0 else "drop",
            })
    return surprises


def find_psych_peaks(psych_by_layer: dict) -> dict:
    """Find peak and trough layers for each psych category.

    Args:
        psych_by_layer: {category: [value_per_layer, ...]}

    Returns:
        {category: {"peak_layer": int, "peak_value": float,
                     "trough_layer": int, "trough_value": float}}
    """
    result = {}
    for cat, values in psych_by_layer.items():
        if not values:
            continue
        peak_idx = max(range(len(values)), key=lambda i: values[i])
        trough_idx = min(range(len(values)), key=lambda i: values[i])
        result[cat] = {
            "peak_layer": peak_idx,
            "peak_value": round(values[peak_idx], 6),
            "trough_layer": trough_idx,
            "trough_value": round(values[trough_idx], 6),
        }
    return result


def compute_psych_correctness_correlation(correct_vs_incorrect: dict,
                                          n_layers: int) -> dict:
    """Compute correlation between psych signals and correctness.

    For each psych category, measures the area between correct and incorrect
    mean profiles. Positive = higher psych signal when correct.

    Args:
        correct_vs_incorrect: output from BaseLayerwiseProbe._compare_correct_incorrect
        n_layers: number of layers

    Returns:
        {category: {"area_difference": float, "divergence_layer": int,
                     "correlation_direction": "positive"|"negative"|"neutral"}}
    """
    result = {}
    for cat, data in correct_vs_incorrect.items():
        correct_mean = data.get("correct_mean_by_layer", [])
        incorrect_mean = data.get("incorrect_mean_by_layer", [])

        if not correct_mean or not incorrect_mean:
            continue

        # Area between curves
        n = min(len(correct_mean), len(incorrect_mean))
        area_diff = sum(correct_mean[k] - incorrect_mean[k] for k in range(n)) / n

        direction = "neutral"
        if area_diff > 0.001:
            direction = "positive"
        elif area_diff < -0.001:
            direction = "negative"

        result[cat] = {
            "area_difference": round(area_diff, 6),
            "divergence_layer": data.get("divergence_layer", 0),
            "max_divergence": data.get("max_divergence", 0.0),
            "correlation_direction": direction,
        }
    return result


def compare_probes(probe_results: dict[str, dict]) -> dict:
    """Cross-probe comparison to find shared circuit regions.

    Args:
        probe_results: {probe_name: layerwise_result_dict, ...}

    Returns:
        dict with shared computation regions, divergence points, etc.
    """
    if not probe_results:
        return {}

    # Extract computation regions
    regions = {}
    for name, result in probe_results.items():
        region = result.get("computation_region", (0, 0))
        if isinstance(region, (list, tuple)) and len(region) == 2:
            regions[name] = tuple(region)

    # Find overlapping regions
    overlaps = []
    names = list(regions.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r1 = regions[names[i]]
            r2 = regions[names[j]]
            overlap_start = max(r1[0], r2[0])
            overlap_end = min(r1[1], r2[1])
            if overlap_start < overlap_end:
                overlaps.append({
                    "probes": [names[i], names[j]],
                    "overlap_region": (overlap_start, overlap_end),
                    "overlap_size": overlap_end - overlap_start,
                })

    # Convergence comparison
    convergence = {}
    for name, result in probe_results.items():
        convergence[name] = result.get("mean_convergence_layer", -1)

    # Shared surprise layers
    all_surprises = {}
    for name, result in probe_results.items():
        for s in result.get("surprise_layers", []):
            layer = s["layer"]
            if layer not in all_surprises:
                all_surprises[layer] = []
            all_surprises[layer].append({
                "probe": name,
                "delta": s["delta"],
                "direction": s["direction"],
            })

    # Filter to layers that surprise 2+ probes
    shared_surprises = {
        layer: probes for layer, probes in all_surprises.items()
        if len(probes) >= 2
    }

    return {
        "computation_regions": regions,
        "region_overlaps": overlaps,
        "convergence_layers": convergence,
        "shared_surprise_layers": shared_surprises,
    }


def generate_layerwise_report(probe_results: dict[str, dict],
                              output_path: str | Path | None = None) -> dict:
    """Generate a comprehensive cross-probe layerwise analysis report.

    Args:
        probe_results: {probe_name: layerwise_result_dict, ...}
        output_path: Optional path to write JSON report.

    Returns:
        Complete report dict.
    """
    report = {
        "n_probes": len(probe_results),
        "probes": {},
        "cross_probe": compare_probes(probe_results),
    }

    for name, result in probe_results.items():
        mean_p = result.get("mean_p_correct_by_layer", [])
        psych = result.get("psych_by_layer", {})
        cvi = result.get("correct_vs_incorrect", {})

        probe_report = {
            "n_items": result.get("n_items", 0),
            "n_layers": result.get("n_layers", 0),
            "final_score": result.get("score", 0.0),
            "final_p_correct": result.get("p_correct", 0.0),
            "convergence": analyze_convergence(mean_p),
            "computation_region": find_computation_region(mean_p),
            "surprise_layers": detect_surprises(mean_p),
            "psych_peaks": find_psych_peaks(psych),
            "psych_correctness_correlation": compute_psych_correctness_correlation(
                cvi, result.get("n_layers", 0)),
            "mean_convergence_layer": result.get("mean_convergence_layer", -1),
        }

        report["probes"][name] = probe_report

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    return report
