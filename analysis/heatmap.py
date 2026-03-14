"""
Heatmap visualization and circuit boundary detection.

Generates:
- Per-probe delta heatmaps (i vs j, color = delta)
- Combined skyline plot (best delta per block size)
- Circuit boundary detection (contiguous regions of improvement)
- Overlay analysis: four-quadrant classification from duplicate+skip sweeps
"""

import json
from pathlib import Path

import numpy as np


def load_results(results_path: str) -> list[dict]:
    with open(results_path) as f:
        return json.load(f)


def build_delta_matrix(results: list[dict], probe_name: str, n_layers: int) -> np.ndarray:
    """Build NxN matrix of deltas for a given probe."""
    matrix = np.full((n_layers, n_layers + 1), np.nan)
    for r in results:
        i, j = r["i"], r["j"]
        if i == 0 and j == 0:
            continue
        delta = r["probe_deltas"].get(probe_name, 0.0)
        matrix[i, j] = delta
    return matrix


def find_circuit_boundaries(matrix: np.ndarray, threshold: float = 0.05) -> list[dict]:
    """Find contiguous regions where delta > threshold."""
    boundaries = []
    improved = matrix > threshold
    # Simple connected-component labeling on the upper triangle
    visited = np.zeros_like(improved, dtype=bool)

    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            if improved[i, j] and not visited[i, j]:
                # BFS to find connected region
                region = []
                queue = [(i, j)]
                while queue:
                    ci, cj = queue.pop(0)
                    if ci < 0 or ci >= matrix.shape[0] or cj <= ci or cj >= matrix.shape[1]:
                        continue
                    if visited[ci, cj] or not improved[ci, cj]:
                        continue
                    visited[ci, cj] = True
                    region.append((ci, cj))
                    queue.extend([(ci - 1, cj), (ci + 1, cj), (ci, cj - 1), (ci, cj + 1)])

                if len(region) >= 2:
                    deltas = [matrix[r[0], r[1]] for r in region]
                    boundaries.append({
                        "cells": region,
                        "size": len(region),
                        "mean_delta": float(np.mean(deltas)),
                        "max_delta": float(np.max(deltas)),
                        "i_range": (min(r[0] for r in region), max(r[0] for r in region)),
                        "j_range": (min(r[1] for r in region), max(r[1] for r in region)),
                    })

    return sorted(boundaries, key=lambda b: b["max_delta"], reverse=True)


def generate_all_plots(results_path: str, output_dir: str):
    """Generate all analysis plots from sweep results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not installed — skipping plot generation")
        print("Install with: pip install matplotlib")
        return

    results = load_results(results_path)
    if not results:
        print("No results to plot")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Infer layer count and probe names
    max_j = max(r["j"] for r in results if r["j"] > 0) if any(r["j"] > 0 for r in results) else 1
    n_layers = max_j
    probe_names = list(results[0]["probe_scores"].keys()) if results else []

    for probe in probe_names:
        matrix = build_delta_matrix(results, probe, n_layers)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Use diverging colormap: blue = degradation, red = improvement
        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.01)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto", origin="lower")
        ax.set_xlabel("j (loop-back end)")
        ax.set_ylabel("i (loop-back start)")
        ax.set_title(f"Delta heatmap: {probe} probe")
        plt.colorbar(im, ax=ax, label="Score delta vs baseline")

        fig.savefig(out / f"heatmap_{probe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved heatmap_{probe}.png")

        # Circuit boundaries
        boundaries = find_circuit_boundaries(matrix)
        if boundaries:
            print(f"  {probe}: {len(boundaries)} circuit region(s) found")
            for idx, b in enumerate(boundaries[:3]):
                print(f"    Region {idx+1}: i={b['i_range']}, j={b['j_range']}, "
                      f"max_delta={b['max_delta']:.4f}, size={b['size']}")

    # Skyline plot: best delta per block size for each probe
    fig, ax = plt.subplots(figsize=(10, 6))
    for probe in probe_names:
        block_sizes = {}
        for r in results:
            bs = r["n_duplicated"]
            if bs == 0:
                continue
            delta = r["probe_deltas"].get(probe, 0.0)
            if bs not in block_sizes or delta > block_sizes[bs]:
                block_sizes[bs] = delta

        if block_sizes:
            sizes = sorted(block_sizes.keys())
            deltas = [block_sizes[s] for s in sizes]
            ax.plot(sizes, deltas, marker="o", label=probe)

    ax.set_xlabel("Block size (j - i)")
    ax.set_ylabel("Best delta vs baseline")
    ax.set_title("Skyline: best improvement per block size")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    fig.savefig(out / "skyline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved skyline.png")

    # Save boundaries as JSON
    all_boundaries = {}
    for probe in probe_names:
        matrix = build_delta_matrix(results, probe, n_layers)
        boundaries = find_circuit_boundaries(matrix)
        # Convert tuples to lists for JSON
        for b in boundaries:
            b["cells"] = [list(c) for c in b["cells"]]
            b["i_range"] = list(b["i_range"])
            b["j_range"] = list(b["j_range"])
        all_boundaries[probe] = boundaries

    with open(out / "circuit_boundaries.json", "w") as f:
        json.dump(all_boundaries, f, indent=2)
    print("Saved circuit_boundaries.json")


# ------------------------------------------------------------------ #
#  Difficulty-aware analysis                                           #
# ------------------------------------------------------------------ #

def build_difficulty_matrices(
    results: list[dict], probe_name: str, n_layers: int
) -> dict[str, np.ndarray]:
    """
    Build three matrices from difficulty-tiered sweep results.

    Returns dict with keys:
      "overall": standard delta matrix (same as build_delta_matrix)
      "easy": delta on easy items only
      "hard": delta on hard items only
      "diff": hard - easy (positive = circuit helps hard more)
    """
    overall = np.full((n_layers, n_layers + 1), np.nan)
    easy = np.full((n_layers, n_layers + 1), np.nan)
    hard = np.full((n_layers, n_layers + 1), np.nan)

    # Find baseline scores for normalization
    baseline_overall = 0.0
    baseline_easy = 0.0
    baseline_hard = 0.0
    for r in results:
        if r["i"] == 0 and r["j"] == 0:
            baseline_overall = r["probe_scores"].get(probe_name, 0.0)
            # Check for difficulty-tiered scores
            easy_key = f"{probe_name}_easy"
            hard_key = f"{probe_name}_hard"
            baseline_easy = r["probe_scores"].get(easy_key, baseline_overall)
            baseline_hard = r["probe_scores"].get(hard_key, baseline_overall)
            break

    for r in results:
        i, j = r["i"], r["j"]
        if i == 0 and j == 0:
            continue

        score = r["probe_scores"].get(probe_name, 0.0)
        overall[i, j] = score - baseline_overall

        easy_key = f"{probe_name}_easy"
        hard_key = f"{probe_name}_hard"
        if easy_key in r["probe_scores"]:
            easy[i, j] = r["probe_scores"][easy_key] - baseline_easy
        if hard_key in r["probe_scores"]:
            hard[i, j] = r["probe_scores"][hard_key] - baseline_hard

    # Diff matrix: hard - easy (positive = helps hard more)
    diff = hard - easy

    return {"overall": overall, "easy": easy, "hard": hard, "diff": diff}


def find_fastpath_circuits(
    easy_matrix: np.ndarray, hard_matrix: np.ndarray, threshold: float = 0.03
) -> list[dict]:
    """
    Find circuits where easy improves but hard degrades or stays flat.
    These are fast-path candidates — good for easy queries, skip for hard ones.
    """
    results = []
    for i in range(easy_matrix.shape[0]):
        for j in range(i + 1, easy_matrix.shape[1]):
            e_val = easy_matrix[i, j]
            h_val = hard_matrix[i, j]
            if (not np.isnan(e_val) and not np.isnan(h_val)
                    and e_val > threshold and h_val < 0):
                results.append({
                    "i": i, "j": j,
                    "easy_delta": float(e_val),
                    "hard_delta": float(h_val),
                    "type": "fast_path",
                })
    return sorted(results, key=lambda x: x["easy_delta"], reverse=True)


def find_complexity_circuits(
    easy_matrix: np.ndarray, hard_matrix: np.ndarray, threshold: float = 0.03
) -> list[dict]:
    """
    Find circuits where hard improves but easy stays flat.
    These are depth circuits — only activate for hard tasks.
    """
    results = []
    for i in range(hard_matrix.shape[0]):
        for j in range(i + 1, hard_matrix.shape[1]):
            e_val = easy_matrix[i, j]
            h_val = hard_matrix[i, j]
            if (not np.isnan(e_val) and not np.isnan(h_val)
                    and h_val > threshold and e_val < 0.1 * threshold):
                results.append({
                    "i": i, "j": j,
                    "easy_delta": float(e_val),
                    "hard_delta": float(h_val),
                    "type": "complexity",
                })
    return sorted(results, key=lambda x: x["hard_delta"], reverse=True)


def plot_difficulty_comparison(
    results_path: str, output_dir: str, probe_name: str = None
):
    """
    Generate side-by-side easy/hard/diff heatmaps for difficulty-tiered probes.

    If probe_name is None, generates for all probes that have difficulty data.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not installed — skipping difficulty plots")
        return

    results = load_results(results_path)
    if not results:
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    max_j = max(r["j"] for r in results if r["j"] > 0) if any(r["j"] > 0 for r in results) else 1
    n_layers = max_j
    probe_names = list(results[0]["probe_scores"].keys()) if results else []

    # Filter to probes that have difficulty data
    probes_to_plot = []
    for pname in probe_names:
        if pname.endswith("_easy") or pname.endswith("_hard"):
            continue
        easy_key = f"{pname}_easy"
        if easy_key in results[0].get("probe_scores", {}):
            probes_to_plot.append(pname)

    if probe_name:
        probes_to_plot = [probe_name] if probe_name in probes_to_plot else []

    for probe in probes_to_plot:
        matrices = build_difficulty_matrices(results, probe, n_layers)

        fig, axes = plt.subplots(1, 3, figsize=(24, 7))

        for idx, (key, title) in enumerate([
            ("easy", f"{probe} — Easy items"),
            ("hard", f"{probe} — Hard items"),
            ("diff", f"{probe} — Hard minus Easy"),
        ]):
            mat = matrices[key]
            ax = axes[idx]

            if key == "diff":
                vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 0.01)
                norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                cmap = "PiYG"
            else:
                vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 0.01)
                norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                cmap = "RdBu_r"

            im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto", origin="lower")
            ax.set_xlabel("j")
            ax.set_ylabel("i")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label="Delta")

        fig.suptitle(f"Difficulty Comparison: {probe}", fontsize=14)
        fig.savefig(out / f"difficulty_{probe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved difficulty_{probe}.png")

        # Report fast-path and complexity circuits
        fast = find_fastpath_circuits(matrices["easy"], matrices["hard"])
        complexity = find_complexity_circuits(matrices["easy"], matrices["hard"])
        if fast:
            print(f"  {probe}: {len(fast)} fast-path circuits (help easy, hurt hard)")
        if complexity:
            print(f"  {probe}: {len(complexity)} complexity circuits (help hard only)")


def classify_region(dup_delta: float, skip_delta: float, threshold: float = 0.03) -> str:
    """
    Classify an (i,j) region into one of four quadrants.

    - "double": duplication helps, skipping doesn't → layer is useful and benefits from repetition
    - "skip": skipping helps, duplication doesn't → layer is harmful, remove it
    - "neutral": neither helps significantly → layer is passively useful
    - "ambiguous": both help → unclear, needs further investigation
    """
    dup_high = dup_delta > threshold
    skip_high = skip_delta > threshold

    if dup_high and not skip_high:
        return "double"
    if skip_high and not dup_high:
        return "skip"
    if dup_high and skip_high:
        return "ambiguous"
    return "neutral"


def generate_overlay_analysis(
    dup_results_path: str,
    skip_results_path: str,
    output_dir: str,
    threshold: float = 0.03,
):
    """
    Generate four-quadrant overlay analysis from duplicate and skip sweep results.

    For each probe and (i,j) pair present in both sweeps, classifies the region as
    double/skip/neutral/ambiguous. Outputs:
    - overlay_<probe>.png: scatter plot of dup_delta vs skip_delta
    - quadrant_map_<probe>.png: heatmap colored by quadrant classification
    - optimized_path_recommendations.json: per-probe recommendations
    """
    dup_results = load_results(dup_results_path)
    skip_results = load_results(skip_results_path)

    if not dup_results or not skip_results:
        print("Need both duplicate and skip results for overlay analysis")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Index results by (i,j)
    def index_by_ij(results):
        indexed = {}
        for r in results:
            key = (r["i"], r["j"])
            if key != (0, 0):
                indexed[key] = r
        return indexed

    dup_idx = index_by_ij(dup_results)
    skip_idx = index_by_ij(skip_results)

    # Find common (i,j) pairs
    common_keys = set(dup_idx.keys()) & set(skip_idx.keys())
    if not common_keys:
        print("No overlapping (i,j) configs between duplicate and skip sweeps")
        return

    # Infer probe names and layer count
    probe_names = list(dup_results[0]["probe_scores"].keys()) if dup_results else []
    max_j = max(k[1] for k in common_keys)
    n_layers = max_j

    recommendations = {}

    for probe in probe_names:
        # Classify each (i,j)
        classifications = {}
        dup_deltas = []
        skip_deltas = []
        labels = []

        for key in sorted(common_keys):
            dd = dup_idx[key]["probe_deltas"].get(probe, 0.0)
            sd = skip_idx[key]["probe_deltas"].get(probe, 0.0)
            cls = classify_region(dd, sd, threshold)
            classifications[key] = cls
            dup_deltas.append(dd)
            skip_deltas.append(sd)
            labels.append(cls)

        # Build quadrant map matrix
        quadrant_values = {"double": 1.0, "skip": -1.0, "neutral": 0.0, "ambiguous": 0.5}
        qmap = np.full((n_layers, n_layers + 1), np.nan)
        for (i, j), cls in classifications.items():
            if i < n_layers and j <= n_layers:
                qmap[i, j] = quadrant_values[cls]

        # Collect skip and duplicate recommendations
        skip_regions = []
        double_regions = []
        for (i, j), cls in sorted(classifications.items()):
            if cls == "skip":
                skip_regions.append({"i": i, "j": j,
                                     "skip_delta": skip_idx[(i, j)]["probe_deltas"].get(probe, 0.0),
                                     "dup_delta": dup_idx[(i, j)]["probe_deltas"].get(probe, 0.0)})
            elif cls == "double":
                double_regions.append({"i": i, "j": j,
                                       "dup_delta": dup_idx[(i, j)]["probe_deltas"].get(probe, 0.0),
                                       "skip_delta": skip_idx[(i, j)]["probe_deltas"].get(probe, 0.0)})

        # Sort by delta magnitude
        skip_regions.sort(key=lambda x: x["skip_delta"], reverse=True)
        double_regions.sort(key=lambda x: x["dup_delta"], reverse=True)

        counts = {c: labels.count(c) for c in ["double", "skip", "neutral", "ambiguous"]}

        recommendations[probe] = {
            "counts": counts,
            "skip_regions": skip_regions[:10],
            "double_regions": double_regions[:10],
            "optimized_path_hint": {
                "skip": [(r["i"], r["j"]) for r in skip_regions[:5]],
                "duplicate": [(r["i"], r["j"]) for r in double_regions[:5]],
            },
        }

        # --- Plot overlay scatter ---
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            color_map = {"double": "red", "skip": "blue", "neutral": "gray", "ambiguous": "orange"}
            colors = [color_map[l] for l in labels]

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(dup_deltas, skip_deltas, c=colors, alpha=0.6, s=30)
            ax.axhline(y=threshold, color="blue", linestyle="--", alpha=0.3, label=f"skip threshold={threshold}")
            ax.axvline(x=threshold, color="red", linestyle="--", alpha=0.3, label=f"dup threshold={threshold}")
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.2)
            ax.axvline(x=0, color="gray", linestyle="-", alpha=0.2)
            ax.set_xlabel("Duplication delta")
            ax.set_ylabel("Skip delta")
            ax.set_title(f"Overlay: {probe} probe — {counts}")

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="red", label=f"double ({counts.get('double', 0)})"),
                Patch(facecolor="blue", label=f"skip ({counts.get('skip', 0)})"),
                Patch(facecolor="gray", label=f"neutral ({counts.get('neutral', 0)})"),
                Patch(facecolor="orange", label=f"ambiguous ({counts.get('ambiguous', 0)})"),
            ]
            ax.legend(handles=legend_elements, loc="upper left")

            fig.savefig(out / f"overlay_{probe}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved overlay_{probe}.png")

            # --- Quadrant map heatmap ---
            fig, ax = plt.subplots(figsize=(10, 8))
            from matplotlib.colors import ListedColormap, BoundaryNorm
            cmap = ListedColormap(["blue", "gray", "orange", "red"])
            bounds = [-1.5, -0.5, 0.25, 0.75, 1.5]
            norm = BoundaryNorm(bounds, cmap.N)

            im = ax.imshow(qmap, cmap=cmap, norm=norm, aspect="auto", origin="lower")
            ax.set_xlabel("j (block end)")
            ax.set_ylabel("i (block start)")
            ax.set_title(f"Quadrant map: {probe} probe")
            cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 0.5, 1])
            cbar.ax.set_yticklabels(["skip", "neutral", "ambiguous", "double"])

            fig.savefig(out / f"quadrant_map_{probe}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved quadrant_map_{probe}.png")

        except ImportError:
            print("matplotlib not installed — skipping overlay plots")

    # Save recommendations
    with open(out / "optimized_path_recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)
    print("Saved optimized_path_recommendations.json")


def generate_research_report(results_path: str, output_dir: str) -> str:
    """Generate CIRCUIT_REPORT.md summarizing all analyses.

    Runs compound analysis and safety analysis, then compiles a markdown
    report with the most significant findings.

    Args:
        results_path: Path to sweep_results.json.
        output_dir: Directory for output files.

    Returns:
        Path to the generated CIRCUIT_REPORT.md file.
    """
    from analysis.compound import generate_compound_report
    from analysis.path_optimizer import PathOptimizer
    from analysis.taxonomy import SAFETY, get_all_probe_names

    results = load_results(results_path)
    if not results:
        return ""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Infer dimensions
    max_j = max(r["j"] for r in results if r["j"] > 0) if any(r["j"] > 0 for r in results) else 1
    n_layers = max_j
    probe_names = list(results[0]["probe_scores"].keys()) if results else []
    matrices = {p: build_delta_matrix(results, p, n_layers) for p in probe_names}

    # Run compound analysis
    compound = generate_compound_report(results_path, output_dir)

    # Run safety analysis
    safety = safety_analysis(results_path, output_dir)

    # Build path optimizer
    optimizer = PathOptimizer(compound, n_layers)
    default_weights = {p: 1.0 for p in probe_names}
    default_path = optimizer.recommend_path(default_weights)
    router_features = optimizer.recommend_router_features(matrices)

    # Generate report
    lines = [
        "# Circuit Analysis Report",
        "",
        f"Model layers: {n_layers}",
        f"Probes evaluated: {', '.join(probe_names)}",
        f"Total configs swept: {len(results)}",
        "",
    ]

    # Top 5 synergistic circuits
    lines.append("## Top Synergistic Circuits")
    lines.append("")
    syn = compound.get("synergistic", [])[:5]
    if syn:
        for s in syn:
            lines.append(f"- **({s['i']},{s['j']})**: {s['n_improving']} probes improve "
                         f"(mean delta={s['mean_delta']:.4f}): {', '.join(s['improving_probes'])}")
    else:
        lines.append("No synergistic circuits found above threshold.")
    lines.append("")

    # Top 5 antagonistic pairs
    lines.append("## Top Antagonistic Pairs")
    lines.append("")
    ant = compound.get("antagonistic", [])[:5]
    if ant:
        for a in ant:
            lines.append(f"- **({a['i']},{a['j']})**: improves {', '.join(a['improved'])} "
                         f"but degrades {', '.join(a['degraded'])} "
                         f"(conflict={a['conflict_magnitude']:.4f})")
    else:
        lines.append("No antagonistic circuits found.")
    lines.append("")

    # Safety findings
    lines.append("## Safety Findings")
    lines.append("")
    if safety:
        n_integrity = len(safety.get("integrity_circuit_candidates", []))
        n_deception = len(safety.get("deception_risk_regions", []))
        n_syco = len(safety.get("sycophancy_circuit_candidates", []))
        n_inst = len(safety.get("instruction_resistance_regions", []))
        lines.append(f"- Integrity circuits: {n_integrity} candidates")
        lines.append(f"- Deception risk regions: {n_deception}")
        lines.append(f"- Sycophancy circuits: {n_syco} candidates")
        lines.append(f"- Instruction resistance: {n_inst} regions")
    else:
        lines.append("Safety analysis not available (insufficient safety probes).")
    lines.append("")

    # Recommended default path
    lines.append("## Recommended Default Path")
    lines.append("")
    lines.append(f"Equal-weight path ({len(default_path)} steps): `{default_path}`")
    lines.append("")

    # Top routing features
    lines.append("## Top Routing Feature Layers")
    lines.append("")
    top_features = list(router_features.items())[:5]
    if top_features:
        for layer, var in top_features:
            lines.append(f"- Layer {layer}: variance={var:.6f}")
    lines.append("")

    report_text = "\n".join(lines)
    report_path = out / "CIRCUIT_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"Saved CIRCUIT_REPORT.md to {output_dir}")
    return str(report_path)


def safety_analysis(results_path: str, output_dir: str, threshold: float = 0.03):
    """
    Analyze safety-relevant probe interactions across (i,j) configs.

    Identifies four types of safety-critical circuit regions:
    1. Integrity circuits: ALL safety probes improve simultaneously
    2. Deception risk: consistency degrades while language improves
    3. Sycophancy circuits: sycophancy degrades while EQ improves
    4. Instruction resistance: instruction degrades while planning improves

    Requires hallucination, sycophancy, consistency, instruction probes in results.
    """
    results = load_results(results_path)
    if not results:
        print("No results for safety analysis")
        return {}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Infer dimensions
    max_j = max(r["j"] for r in results if r["j"] > 0) if any(r["j"] > 0 for r in results) else 1
    n_layers = max_j
    probe_names = list(results[0]["probe_scores"].keys()) if results else []

    safety_probes = ["hallucination", "sycophancy", "consistency", "instruction"]
    available_safety = [p for p in safety_probes if p in probe_names]

    if len(available_safety) < 2:
        print(f"Safety analysis needs at least 2 safety probes, found: {available_safety}")
        return {}

    # Build delta matrices for all probes
    matrices = {p: build_delta_matrix(results, p, n_layers) for p in probe_names}

    report = {
        "integrity_circuit_candidates": [],
        "deception_risk_regions": [],
        "sycophancy_circuit_candidates": [],
        "instruction_resistance_regions": [],
    }

    # 1. Integrity circuits: all available safety probes improve
    if len(available_safety) >= 2:
        for i in range(n_layers):
            for j in range(i + 1, n_layers + 1):
                all_improve = all(
                    not np.isnan(matrices[p][i, j]) and matrices[p][i, j] > threshold
                    for p in available_safety
                    if p in matrices
                )
                if all_improve:
                    deltas = {p: float(matrices[p][i, j]) for p in available_safety if p in matrices}
                    report["integrity_circuit_candidates"].append({
                        "i": i, "j": j, "deltas": deltas,
                    })

    # 2. Deception risk: consistency degrades while language/fluency improves
    if "consistency" in matrices and "language" in matrices:
        c_mat = matrices["consistency"]
        l_mat = matrices["language"]
        for i in range(n_layers):
            for j in range(i + 1, n_layers + 1):
                c_val = c_mat[i, j]
                l_val = l_mat[i, j]
                if (not np.isnan(c_val) and not np.isnan(l_val)
                        and c_val < -threshold and l_val > threshold):
                    report["deception_risk_regions"].append({
                        "i": i, "j": j,
                        "consistency_delta": float(c_val),
                        "language_delta": float(l_val),
                    })

    # 3. Sycophancy circuits: sycophancy degrades while EQ improves
    if "sycophancy" in matrices and "eq" in matrices:
        s_mat = matrices["sycophancy"]
        e_mat = matrices["eq"]
        for i in range(n_layers):
            for j in range(i + 1, n_layers + 1):
                s_val = s_mat[i, j]
                e_val = e_mat[i, j]
                if (not np.isnan(s_val) and not np.isnan(e_val)
                        and s_val < -threshold and e_val > threshold):
                    report["sycophancy_circuit_candidates"].append({
                        "i": i, "j": j,
                        "sycophancy_delta": float(s_val),
                        "eq_delta": float(e_val),
                    })

    # 4. Instruction resistance: instruction degrades while planning improves
    if "instruction" in matrices and "planning" in matrices:
        inst_mat = matrices["instruction"]
        plan_mat = matrices["planning"]
        for i in range(n_layers):
            for j in range(i + 1, n_layers + 1):
                inst_val = inst_mat[i, j]
                plan_val = plan_mat[i, j]
                if (not np.isnan(inst_val) and not np.isnan(plan_val)
                        and inst_val < -threshold and plan_val > threshold):
                    report["instruction_resistance_regions"].append({
                        "i": i, "j": j,
                        "instruction_delta": float(inst_val),
                        "planning_delta": float(plan_val),
                    })

    # Sort by magnitude
    for key in report:
        if report[key]:
            sort_key = "deltas" if key == "integrity_circuit_candidates" else list(report[key][0].keys())[2]
            if key == "integrity_circuit_candidates":
                report[key].sort(key=lambda x: sum(x["deltas"].values()), reverse=True)
            else:
                report[key].sort(key=lambda x: abs(x[sort_key]), reverse=True)

    with open(out / "safety_circuit_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n=== Safety Circuit Analysis ===")
    print(f"  Integrity circuits: {len(report['integrity_circuit_candidates'])} candidates")
    print(f"  Deception risk: {len(report['deception_risk_regions'])} regions")
    print(f"  Sycophancy circuits: {len(report['sycophancy_circuit_candidates'])} candidates")
    print(f"  Instruction resistance: {len(report['instruction_resistance_regions'])} regions")
    print(f"Saved safety_circuit_report.json")

    return report
