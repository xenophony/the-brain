"""
Joint mechanistic + behavioral analysis: overlay residual traces on sweep heatmaps.

Combines two complementary views:
- Behavioral: which (i,j) configs improve/degrade performance (from sweep)
- Mechanistic: where in the residual stream the answer is computed (from traces)

When a sweep heatmap shows improvement at (i,j), the residual trace tells us
*why* — because the duplicated layers overlap with the answer computation region.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class ValidationResult:
    """Result of mechanistic validation of a circuit region."""
    circuit_region: tuple  # (i, j)
    computation_region: tuple  # from residual trace
    overlap: bool
    confidence: float  # 0-1, how much the regions overlap
    probe_name: str


def overlay_trace_on_heatmap(domain_trace, sweep_matrix: np.ndarray, output_path: str = None):
    """Overlay answer computation region on sweep heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not installed — skipping trace overlay")
        return

    comp_region = domain_trace.answer_computation_region
    n_layers = sweep_matrix.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: sweep heatmap with computation region overlay
    vmax = max(abs(np.nanmin(sweep_matrix)), abs(np.nanmax(sweep_matrix)), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax1.imshow(sweep_matrix, cmap="RdBu_r", norm=norm, aspect="auto", origin="lower")
    ax1.axhline(y=comp_region[0], color="lime", linestyle="--", linewidth=2, label="Computation start")
    ax1.axhline(y=comp_region[1], color="lime", linestyle="-", linewidth=2, label="Computation end")
    ax1.set_xlabel("j")
    ax1.set_ylabel("i")
    ax1.set_title(f"{domain_trace.probe_name} — Sweep + Computation Region")
    ax1.legend()
    plt.colorbar(im, ax=ax1)

    # Right: residual stream probability evolution
    if domain_trace.mean_probabilities:
        layers = list(range(len(domain_trace.mean_probabilities)))
        ax2.plot(layers, domain_trace.mean_probabilities, 'b-', linewidth=2, label="Mean p(correct)")
        if domain_trace.std_probabilities:
            mean = np.array(domain_trace.mean_probabilities)
            std = np.array(domain_trace.std_probabilities)
            ax2.fill_between(layers, mean - std, mean + std, alpha=0.2, color="blue")
        # Mark computation region
        ax2.axvspan(comp_region[0], comp_region[1], alpha=0.2, color="green", label="Computation region")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("P(correct answer)")
        ax2.set_title(f"{domain_trace.probe_name} — Residual Stream Evolution")
        ax2.legend()

    fig.suptitle(f"Mechanistic Analysis: {domain_trace.probe_name}", fontsize=14)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")
    else:
        plt.close(fig)


def validate_circuit_mechanistically(circuit_region, domain_trace, sweep_matrix, threshold=0.03):
    """Check if circuit region overlaps with answer computation region."""
    comp = domain_trace.answer_computation_region
    ci, cj = circuit_region

    # Check overlap between circuit row range and computation region
    circuit_layers = set(range(ci, cj))
    comp_layers = set(range(comp[0], comp[1] + 1))

    overlap = circuit_layers & comp_layers
    total = circuit_layers | comp_layers

    confidence = len(overlap) / len(total) if total else 0.0

    return ValidationResult(
        circuit_region=circuit_region,
        computation_region=comp,
        overlap=len(overlap) > 0,
        confidence=confidence,
        probe_name=domain_trace.probe_name,
    )


def generate_mechanistic_report(all_traces: dict, sweep_results_path: str, output_dir: str):
    """Generate MECHANISTIC_REPORT.md from traces and sweep results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines = ["# Mechanistic Analysis Report", ""]

    for probe_name, domain_trace in all_traces.items():
        lines.append(f"## {probe_name}")
        lines.append(f"- Questions traced: {domain_trace.n_questions}")
        lines.append(f"- Mean peak layer: {domain_trace.mean_peak_layer:.1f}")
        lines.append(f"- Answer computation region: layers {domain_trace.answer_computation_region}")
        if domain_trace.suppression_regions:
            lines.append(f"- Suppression regions: {domain_trace.suppression_regions}")
        lines.append("")

    report_path = out / "MECHANISTIC_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {report_path}")
    return str(report_path)
