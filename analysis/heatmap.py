"""
Heatmap visualization and circuit boundary detection.

Generates:
- Per-probe delta heatmaps (i vs j, color = delta)
- Combined skyline plot (best delta per block size)
- Circuit boundary detection (contiguous regions of improvement)
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
