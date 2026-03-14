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
