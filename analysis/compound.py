"""
Compound circuit analysis — finds multi-probe circuit interactions.

Operates on delta matrices (numpy arrays from build_delta_matrix) to identify:
- Synergistic circuits: regions where many probes improve simultaneously
- Antagonistic circuits: improving one probe significantly degrades another
- Cascade candidates: overlapping beneficial regions for pipeline circuits
- Inhibitory circuits: single-probe gains that vanish in combination
"""

import json
from pathlib import Path

import numpy as np

from analysis.heatmap import load_results, build_delta_matrix


def find_synergistic_circuits(
    matrices: dict[str, np.ndarray],
    min_probes: int = 3,
    threshold: float = 0.03,
) -> list[dict]:
    """Find (i,j) regions where min_probes or more improve above threshold.

    A synergistic circuit is a layer configuration that benefits multiple
    cognitive functions simultaneously — these are high-value targets for
    the router since a single path adjustment improves multiple domains.

    Args:
        matrices: {probe_name: NxM delta matrix} from build_delta_matrix.
        min_probes: Minimum number of probes that must improve.
        threshold: Minimum delta to count as improvement.

    Returns:
        List of dicts with keys: i, j, improving_probes, deltas, n_improving.
        Sorted by n_improving descending, then by mean delta.
    """
    if not matrices:
        return []

    probe_names = list(matrices.keys())
    ref = next(iter(matrices.values()))
    n_rows, n_cols = ref.shape

    results = []
    for i in range(n_rows):
        for j in range(i + 1, n_cols):
            improving = {}
            for name in probe_names:
                val = matrices[name][i, j]
                if not np.isnan(val) and val > threshold:
                    improving[name] = float(val)

            if len(improving) >= min_probes:
                results.append({
                    "i": i,
                    "j": j,
                    "improving_probes": list(improving.keys()),
                    "deltas": improving,
                    "n_improving": len(improving),
                    "mean_delta": float(np.mean(list(improving.values()))),
                    "description": (
                        f"({i},{j}): {len(improving)} probes improve "
                        f"({', '.join(improving.keys())})"
                    ),
                })

    results.sort(key=lambda x: (-x["n_improving"], -x["mean_delta"]))
    return results


def find_antagonistic_circuits(
    matrices: dict[str, np.ndarray],
    threshold: float = 0.03,
) -> list[dict]:
    """Find (i,j) where improving probe A significantly degrades probe B.

    Antagonistic circuits reveal fundamental trade-offs in the model's
    architecture — improving one capability necessarily hurts another.
    These are critical for router design: the router must choose which
    capability to prioritize.

    Args:
        matrices: {probe_name: NxM delta matrix}.
        threshold: Minimum magnitude for both improvement and degradation.

    Returns:
        List of dicts with keys: i, j, improved, degraded, deltas, description.
        Sorted by conflict magnitude (sum of absolute deltas).
    """
    if not matrices:
        return []

    probe_names = list(matrices.keys())
    ref = next(iter(matrices.values()))
    n_rows, n_cols = ref.shape

    results = []
    for i in range(n_rows):
        for j in range(i + 1, n_cols):
            improved = {}
            degraded = {}
            for name in probe_names:
                val = matrices[name][i, j]
                if np.isnan(val):
                    continue
                if val > threshold:
                    improved[name] = float(val)
                elif val < -threshold:
                    degraded[name] = float(val)

            if improved and degraded:
                all_deltas = {**improved, **degraded}
                conflict_mag = sum(abs(v) for v in all_deltas.values())
                results.append({
                    "i": i,
                    "j": j,
                    "improved": list(improved.keys()),
                    "degraded": list(degraded.keys()),
                    "deltas": all_deltas,
                    "conflict_magnitude": conflict_mag,
                    "description": (
                        f"({i},{j}): improves {', '.join(improved.keys())} "
                        f"but degrades {', '.join(degraded.keys())}"
                    ),
                })

    results.sort(key=lambda x: -x["conflict_magnitude"])
    return results


def find_cascade_candidates(
    matrices: dict[str, np.ndarray],
    threshold: float = 0.03,
) -> list[dict]:
    """Find overlapping beneficial regions as pipeline circuit candidates.

    A cascade candidate is a pair of (i,j) configs that share overlapping
    layer ranges and both improve (possibly different) probes. This suggests
    that running layers in the overlapping region multiple times in a pipeline
    could compound benefits.

    Args:
        matrices: {probe_name: NxM delta matrix}.
        threshold: Minimum delta for a region to be considered beneficial.

    Returns:
        List of dicts with keys: region1, region2, overlap, probes, description.
        Sorted by overlap size descending.
    """
    if not matrices:
        return []

    probe_names = list(matrices.keys())
    ref = next(iter(matrices.values()))
    n_rows, n_cols = ref.shape

    # First, find beneficial regions per probe
    beneficial = []
    for name in probe_names:
        for i in range(n_rows):
            for j in range(i + 1, n_cols):
                val = matrices[name][i, j]
                if not np.isnan(val) and val > threshold:
                    beneficial.append({
                        "probe": name,
                        "i": i,
                        "j": j,
                        "delta": float(val),
                    })

    # Find pairs with overlapping layer ranges from different probes
    results = []
    seen = set()
    for a_idx, a in enumerate(beneficial):
        for b in beneficial[a_idx + 1:]:
            if a["probe"] == b["probe"]:
                continue
            # Compute overlap
            ov_start = max(a["i"], b["i"])
            ov_end = min(a["j"], b["j"])
            if ov_start >= ov_end:
                continue

            key = (
                min(a["i"], b["i"]), min(a["j"], b["j"]),
                max(a["i"], b["i"]), max(a["j"], b["j"]),
                tuple(sorted([a["probe"], b["probe"]])),
            )
            if key in seen:
                continue
            seen.add(key)

            results.append({
                "region1": {"i": a["i"], "j": a["j"], "probe": a["probe"], "delta": a["delta"]},
                "region2": {"i": b["i"], "j": b["j"], "probe": b["probe"], "delta": b["delta"]},
                "overlap": {"i": ov_start, "j": ov_end, "size": ov_end - ov_start},
                "probes": sorted([a["probe"], b["probe"]]),
                "description": (
                    f"Overlap layers {ov_start}-{ov_end-1}: "
                    f"{a['probe']}({a['i']},{a['j']}) + "
                    f"{b['probe']}({b['i']},{b['j']})"
                ),
            })

    results.sort(key=lambda x: -x["overlap"]["size"])
    return results


def find_inhibitory_circuits(
    matrices: dict[str, np.ndarray],
    threshold: float = 0.03,
) -> list[dict]:
    """Find regions where single-probe improvement is high but combined is lower.

    An inhibitory circuit is an (i,j) config where one probe improves strongly
    but the average across all probes is much lower than that single probe's
    improvement. This suggests the circuit helps one function at the expense
    of others — an inhibitory interaction.

    Args:
        matrices: {probe_name: NxM delta matrix}.
        threshold: Minimum single-probe delta to consider.

    Returns:
        List of dicts with keys: i, j, best_probe, best_delta, mean_delta,
        inhibition_ratio, deltas, description.
        Sorted by inhibition_ratio descending.
    """
    if not matrices:
        return []

    probe_names = list(matrices.keys())
    ref = next(iter(matrices.values()))
    n_rows, n_cols = ref.shape

    results = []
    for i in range(n_rows):
        for j in range(i + 1, n_cols):
            deltas = {}
            for name in probe_names:
                val = matrices[name][i, j]
                if not np.isnan(val):
                    deltas[name] = float(val)

            if len(deltas) < 2:
                continue

            best_probe = max(deltas, key=deltas.get)
            best_delta = deltas[best_probe]
            if best_delta < threshold:
                continue

            mean_delta = float(np.mean(list(deltas.values())))

            # Inhibition: best probe is strong but average is weak or negative
            if best_delta > 0 and mean_delta < best_delta * 0.5:
                inhibition_ratio = 1.0 - (mean_delta / best_delta) if best_delta > 0 else 0.0
                results.append({
                    "i": i,
                    "j": j,
                    "best_probe": best_probe,
                    "best_delta": best_delta,
                    "mean_delta": mean_delta,
                    "inhibition_ratio": float(inhibition_ratio),
                    "deltas": deltas,
                    "description": (
                        f"({i},{j}): {best_probe} improves by {best_delta:.4f} "
                        f"but mean across probes is {mean_delta:.4f} "
                        f"(inhibition={inhibition_ratio:.2f})"
                    ),
                })

    results.sort(key=lambda x: -x["inhibition_ratio"])
    return results


def generate_compound_report(results_path: str, output_dir: str) -> dict:
    """Run all compound analyses and save results.

    Args:
        results_path: Path to sweep_results.json.
        output_dir: Directory for output files.

    Returns:
        Dict with keys: synergistic, antagonistic, cascade, inhibitory,
        each containing the analysis results.
    """
    results = load_results(results_path)
    if not results:
        return {}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Infer dimensions
    max_j = max(r["j"] for r in results if r["j"] > 0) if any(r["j"] > 0 for r in results) else 1
    n_layers = max_j
    probe_names = list(results[0]["probe_scores"].keys()) if results else []

    # Build delta matrices for all probes
    matrices = {p: build_delta_matrix(results, p, n_layers) for p in probe_names}

    report = {
        "synergistic": find_synergistic_circuits(matrices),
        "antagonistic": find_antagonistic_circuits(matrices),
        "cascade": find_cascade_candidates(matrices),
        "inhibitory": find_inhibitory_circuits(matrices),
    }

    # Save report
    with open(out / "compound_analysis.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n=== Compound Circuit Analysis ===")
    print(f"  Synergistic circuits: {len(report['synergistic'])}")
    print(f"  Antagonistic circuits: {len(report['antagonistic'])}")
    print(f"  Cascade candidates: {len(report['cascade'])}")
    print(f"  Inhibitory circuits: {len(report['inhibitory'])}")
    print(f"Saved compound_analysis.json to {output_dir}")

    return report
