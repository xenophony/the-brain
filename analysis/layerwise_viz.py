"""
Layerwise probe visualization — publication-quality charts from per-layer data.

Generates:
1. p(correct) curves with convergence markers and computation regions
2. Psycholinguistic signal heatmaps across layers
3. Correct vs incorrect psych divergence plots
4. Multi-category convergence comparison (convergence probe)
5. Multi-probe convergence bar chart
6. Computation region Gantt chart
7. Batch generation from a results directory
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# --------------------------------------------------------------------------- #
#  Color palette — consistent across all plots                                 #
# --------------------------------------------------------------------------- #

# Psycholinguistic categories split into functional groups for heatmap ordering
_EMOTIONAL_CATS = [
    "hedging", "confidence", "epistemic_uncertain", "epistemic_certain",
    "causation", "approximators", "negation", "absolutes",
    "first_person", "distancing", "urgency", "stress",
    "multilingual_yes", "multilingual_no",
]
_FUNCTION_WORD_CATS = [
    "articles", "conjunctions", "prepositions",
    "auxiliaries", "modals", "quantifiers",
]

# Probe color cycle for multi-probe charts
_PROBE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def _load_json(path: str | Path) -> dict:
    """Load a JSON file, returning empty dict on error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        print(f"Warning: could not load {path}: {e}")
        return {}


def _style_ax(ax, xlabel: str, ylabel: str, title: str):
    """Apply consistent styling to an axes."""
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.2, linewidth=0.5)


def _save_fig(fig, output_path: str | Path, dpi: int = 300):
    """Save figure and close."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {output_path}")


# --------------------------------------------------------------------------- #
#  1. p(correct) curve                                                         #
# --------------------------------------------------------------------------- #

def plot_p_correct_curve(result: dict, output_path: str | Path):
    """Plot p(correct) across layers for a single probe.

    Line plot with std deviation band, convergence marker, computation region
    shading, and surprise layer arrows.

    Args:
        result: Single probe layerwise result dict.
        output_path: Where to save the PNG.
    """
    mean_p = result.get("mean_p_correct_by_layer", [])
    if not mean_p:
        print(f"Warning: no mean_p_correct_by_layer data, skipping p_correct curve")
        return

    n_layers = len(mean_p)
    layers = np.arange(n_layers)
    mean_arr = np.array(mean_p)
    std_p = result.get("std_p_correct_by_layer", [])
    std_arr = np.array(std_p) if std_p and len(std_p) == n_layers else np.zeros(n_layers)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Computation region shading
    comp_region = result.get("computation_region", (0, 0))
    if isinstance(comp_region, (list, tuple)) and len(comp_region) == 2:
        r_start, r_end = comp_region
        if r_end > r_start:
            ax.axvspan(r_start, r_end, alpha=0.10, color="#2ca02c",
                       label=f"Computation region ({r_start}-{r_end})")

    # Std deviation band
    ax.fill_between(layers, mean_arr - std_arr, mean_arr + std_arr,
                    alpha=0.15, color="#1f77b4")

    # Main line
    ax.plot(layers, mean_arr, color="#1f77b4", linewidth=1.8, zorder=3)

    # Convergence layer
    conv_layer = result.get("mean_convergence_layer")
    if conv_layer is not None and isinstance(conv_layer, (int, float)):
        conv_val = conv_layer
        if 0 <= conv_val < n_layers:
            ax.axvline(conv_val, color="#d62728", linestyle="--", linewidth=1.2,
                       alpha=0.7, label=f"Convergence ({conv_val:.1f})")

    # Surprise layers
    surprises = result.get("surprise_layers", [])
    for s in surprises:
        layer_idx = s.get("layer", 0)
        delta = s.get("delta", 0)
        direction = s.get("direction", "rise")
        if 0 <= layer_idx < n_layers:
            marker = "^" if direction == "rise" else "v"
            color = "#2ca02c" if direction == "rise" else "#d62728"
            ax.plot(layer_idx, mean_arr[layer_idx], marker=marker, color=color,
                    markersize=10, zorder=5, markeredgecolor="black",
                    markeredgewidth=0.5)
            ax.annotate(f"{delta:+.2f}", (layer_idx, mean_arr[layer_idx]),
                        textcoords="offset points", xytext=(5, 10 if direction == "rise" else -15),
                        fontsize=7, color=color, fontweight="bold")

    # Chance line
    n_choices = len(result.get("items", [{}])[0].get("layers", [{}])[0].get("psych", {})) or 2
    # Use 0.5 as generic chance line
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5,
               label="Chance (0.5)")

    probe_name = result.get("probe_name", "unknown")
    n_items = result.get("n_items", 0)
    score = result.get("score", 0.0)
    _style_ax(ax, "Layer", "p(correct)",
              f"{probe_name} — p(correct) by layer  [n={n_items}, score={score:.3f}]")
    ax.set_xlim(0, n_layers - 1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9)

    _save_fig(fig, output_path)


# --------------------------------------------------------------------------- #
#  2. Psycholinguistic heatmap                                                 #
# --------------------------------------------------------------------------- #

def plot_psych_heatmap(result: dict, output_path: str | Path):
    """Plot heatmap of psycholinguistic signals across layers.

    x = layer, y = psych category (grouped: function words then emotional).
    Color = probability mass.

    Args:
        result: Single probe layerwise result dict.
        output_path: Where to save the PNG.
    """
    psych = result.get("psych_by_layer", {})
    if not psych:
        print("Warning: no psych_by_layer data, skipping psych heatmap")
        return

    # Order categories: function words first, then emotional, then any extras
    ordered_cats = []
    for cat in _FUNCTION_WORD_CATS:
        if cat in psych:
            ordered_cats.append(cat)
    for cat in _EMOTIONAL_CATS:
        if cat in psych:
            ordered_cats.append(cat)
    for cat in psych:
        if cat not in ordered_cats:
            ordered_cats.append(cat)

    if not ordered_cats:
        return

    n_layers = len(psych[ordered_cats[0]])
    data = np.array([psych[cat] for cat in ordered_cats])

    fig, ax = plt.subplots(figsize=(14, max(4, len(ordered_cats) * 0.35)))

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(ordered_cats)))
    ax.set_yticklabels(ordered_cats, fontsize=8)

    # Draw separator between function words and emotional categories
    n_func = sum(1 for c in ordered_cats if c in _FUNCTION_WORD_CATS)
    if 0 < n_func < len(ordered_cats):
        ax.axhline(n_func - 0.5, color="white", linewidth=2)

    # X-axis: show every 5th layer
    xticks = list(range(0, n_layers, max(1, n_layers // 15)))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=8)

    probe_name = result.get("probe_name", "unknown")
    _style_ax(ax, "Layer", "Psych category",
              f"{probe_name} — psycholinguistic signal heatmap")
    plt.colorbar(im, ax=ax, label="Mean probability mass", shrink=0.8)

    _save_fig(fig, output_path)


# --------------------------------------------------------------------------- #
#  3. Correct vs incorrect divergence                                          #
# --------------------------------------------------------------------------- #

def plot_correct_vs_incorrect(result: dict, category: str, output_path: str | Path):
    """Plot correct vs incorrect psych profiles for one category.

    Two lines with divergence region shaded.

    Args:
        result: Single probe layerwise result dict.
        category: Psych category name (e.g. "hedging").
        output_path: Where to save the PNG.
    """
    cvi = result.get("correct_vs_incorrect", {})
    cat_data = cvi.get(category)
    if not cat_data:
        print(f"Warning: no correct_vs_incorrect data for '{category}', skipping")
        return

    correct_mean = cat_data.get("correct_mean_by_layer", [])
    incorrect_mean = cat_data.get("incorrect_mean_by_layer", [])
    if not correct_mean or not incorrect_mean:
        return

    n_layers = min(len(correct_mean), len(incorrect_mean))
    layers = np.arange(n_layers)
    c_arr = np.array(correct_mean[:n_layers])
    i_arr = np.array(incorrect_mean[:n_layers])

    fig, ax = plt.subplots(figsize=(12, 5))

    # Shade divergence region
    div_layer = cat_data.get("divergence_layer", 0)
    max_div = cat_data.get("max_divergence", 0.0)

    # Find region where divergence is > half of max
    if max_div > 0:
        diff = np.abs(c_arr - i_arr)
        div_mask = diff > max_div * 0.3
        in_region = False
        region_start = 0
        for k in range(n_layers):
            if div_mask[k] and not in_region:
                region_start = k
                in_region = True
            elif not div_mask[k] and in_region:
                ax.axvspan(region_start, k, alpha=0.08, color="#ff7f0e")
                in_region = False
        if in_region:
            ax.axvspan(region_start, n_layers - 1, alpha=0.08, color="#ff7f0e")

    ax.plot(layers, c_arr, color="#2ca02c", linewidth=1.5, label="Correct items")
    ax.plot(layers, i_arr, color="#d62728", linewidth=1.5, label="Incorrect items")

    # Mark peak divergence
    if 0 <= div_layer < n_layers:
        ax.axvline(div_layer, color="#ff7f0e", linestyle="--", linewidth=1.0,
                   alpha=0.6, label=f"Peak divergence (layer {div_layer})")

    probe_name = result.get("probe_name", "unknown")
    _style_ax(ax, "Layer", f"Mean {category} signal",
              f"{probe_name} — {category}: correct vs incorrect  "
              f"[max div={max_div:.4f}]")
    ax.set_xlim(0, n_layers - 1)
    ax.legend(loc="best", fontsize=9)

    _save_fig(fig, output_path)


# --------------------------------------------------------------------------- #
#  4. Convergence comparison (convergence probe)                               #
# --------------------------------------------------------------------------- #

def plot_convergence_comparison(convergence_result: dict, output_path: str | Path):
    """Plot per-category convergence for the convergence probe.

    One line per function word category, legend shows convergence layer.

    Args:
        convergence_result: Result dict from convergence_layerwise probe.
        output_path: Where to save the PNG.
    """
    cat_conv = convergence_result.get("category_convergence", {})
    if not cat_conv:
        print("Warning: no category_convergence data, skipping convergence comparison")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (cat, data) in enumerate(sorted(cat_conv.items())):
        mean_p = data.get("mean_p_correct_by_layer", [])
        if not mean_p:
            continue
        conv_layer = data.get("mean_convergence_layer", -1)
        color = _PROBE_COLORS[idx % len(_PROBE_COLORS)]
        layers = np.arange(len(mean_p))

        label = f"{cat} (conv={conv_layer:.1f})"
        ax.plot(layers, mean_p, color=color, linewidth=1.4, label=label)

        # Mark convergence point
        conv_int = int(round(conv_layer))
        if 0 <= conv_int < len(mean_p):
            ax.plot(conv_int, mean_p[conv_int], "o", color=color,
                    markersize=6, zorder=4, markeredgecolor="black",
                    markeredgewidth=0.5)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    _style_ax(ax, "Layer", "p(correct)",
              "Function word convergence by category")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    _save_fig(fig, output_path)


# --------------------------------------------------------------------------- #
#  5. Multi-probe convergence bar chart                                        #
# --------------------------------------------------------------------------- #

def plot_multi_probe_convergence(report: dict, output_path: str | Path):
    """Horizontal bar chart of convergence layer per probe.

    Sorted earliest to latest.

    Args:
        report: Cross-probe report dict (layerwise_report.json).
        output_path: Where to save the PNG.
    """
    cross = report.get("cross_probe", {})
    conv_layers = cross.get("convergence_layers", {})
    if not conv_layers:
        print("Warning: no convergence_layers data, skipping multi-probe convergence")
        return

    # Filter out invalid entries and sort
    valid = {k: v for k, v in conv_layers.items()
             if isinstance(v, (int, float)) and v >= 0}
    if not valid:
        return

    sorted_items = sorted(valid.items(), key=lambda x: x[1])
    names = [item[0].replace("_layerwise", "") for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.45)))

    colors = [_PROBE_COLORS[i % len(_PROBE_COLORS)] for i in range(len(names))]
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="black",
                   linewidth=0.3, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8)

    _style_ax(ax, "Convergence layer", "Probe",
              "Answer convergence layer by probe (earlier = faster)")
    ax.invert_yaxis()

    _save_fig(fig, output_path)


# --------------------------------------------------------------------------- #
#  6. Computation region Gantt chart                                           #
# --------------------------------------------------------------------------- #

def plot_computation_regions(report: dict, output_path: str | Path):
    """Gantt-style chart of computation regions across probes.

    Each probe is a horizontal bar spanning its computation region.

    Args:
        report: Cross-probe report dict (layerwise_report.json).
        output_path: Where to save the PNG.
    """
    cross = report.get("cross_probe", {})
    regions = cross.get("computation_regions", {})
    if not regions:
        print("Warning: no computation_regions data, skipping Gantt chart")
        return

    # Filter valid regions
    valid = {}
    for name, region in regions.items():
        if isinstance(region, (list, tuple)) and len(region) == 2:
            start, end = region
            if end > start:
                valid[name] = (start, end)

    if not valid:
        return

    # Sort by start layer
    sorted_items = sorted(valid.items(), key=lambda x: x[1][0])
    names = [item[0].replace("_layerwise", "") for item in sorted_items]
    starts = [item[1][0] for item in sorted_items]
    durations = [item[1][1] - item[1][0] for item in sorted_items]

    # Find max layer for x-axis
    max_layer = max(item[1][1] for item in sorted_items)

    fig, ax = plt.subplots(figsize=(12, max(3, len(names) * 0.5)))

    colors = [_PROBE_COLORS[i % len(_PROBE_COLORS)] for i in range(len(names))]
    y_pos = np.arange(len(names))

    ax.barh(y_pos, durations, left=starts, color=colors, edgecolor="black",
            linewidth=0.4, height=0.6, alpha=0.85)

    # Region labels
    for i, (start, dur) in enumerate(zip(starts, durations)):
        mid = start + dur / 2
        ax.text(mid, i, f"{start}-{start + dur}",
                ha="center", va="center", fontsize=7, fontweight="bold",
                color="white" if dur > 2 else "black")

    # Highlight overlapping regions
    overlaps = cross.get("region_overlaps", [])
    for ov in overlaps:
        ov_region = ov.get("overlap_region", (0, 0))
        if isinstance(ov_region, (list, tuple)) and len(ov_region) == 2:
            ax.axvspan(ov_region[0], ov_region[1], alpha=0.06, color="red")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(-0.5, max_layer + 1)
    ax.invert_yaxis()
    _style_ax(ax, "Layer", "Probe",
              "Computation regions — where each probe's answer forms")

    _save_fig(fig, output_path)


# --------------------------------------------------------------------------- #
#  7. Batch generation                                                         #
# --------------------------------------------------------------------------- #

def generate_all_viz(results_dir: str | Path, output_dir: str | Path):
    """Load all JSONs from results_dir and generate all applicable plots.

    Args:
        results_dir: Directory containing layerwise probe JSON files.
        output_dir: Directory for output PNGs.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        return

    # Load individual probe results
    probe_results = {}
    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.name == "layerwise_report.json":
            continue
        data = _load_json(json_file)
        if not data:
            continue

        probe_name = data.get("probe_name", json_file.stem)
        probe_results[probe_name] = data

    if not probe_results:
        print(f"No probe result files found in {results_dir}")
        return

    print(f"Found {len(probe_results)} probe results: {list(probe_results.keys())}")

    # Per-probe plots
    for probe_name, result in probe_results.items():
        safe_name = probe_name.replace("/", "_")

        # 1. p(correct) curve
        if result.get("mean_p_correct_by_layer"):
            plot_p_correct_curve(result, output_dir / f"p_correct_{safe_name}.png")

        # 2. Psych heatmap
        if result.get("psych_by_layer"):
            plot_psych_heatmap(result, output_dir / f"psych_heatmap_{safe_name}.png")

        # 3. Correct vs incorrect for top divergent categories
        cvi = result.get("correct_vs_incorrect", {})
        if cvi:
            # Pick top 5 most divergent categories
            ranked = sorted(cvi.items(),
                            key=lambda x: x[1].get("max_divergence", 0),
                            reverse=True)
            for cat, cat_data in ranked[:5]:
                if cat_data.get("max_divergence", 0) > 0:
                    plot_correct_vs_incorrect(
                        result, cat,
                        output_dir / f"cvi_{safe_name}_{cat}.png")

        # 4. Convergence comparison (convergence probe only)
        if result.get("category_convergence"):
            plot_convergence_comparison(
                result, output_dir / f"convergence_comparison_{safe_name}.png")

    # Cross-probe plots from report
    report_path = results_dir / "layerwise_report.json"
    if report_path.exists():
        report = _load_json(report_path)
        if report:
            # 5. Multi-probe convergence
            plot_multi_probe_convergence(
                report, output_dir / "multi_probe_convergence.png")

            # 6. Computation regions
            plot_computation_regions(
                report, output_dir / "computation_regions.png")
    else:
        print(f"No layerwise_report.json found — skipping cross-probe plots")

    print(f"\nAll plots saved to {output_dir}/")
