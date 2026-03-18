#!/usr/bin/env python
"""
CLI for layerwise probe visualization.

Usage:
    python scripts/viz_layerwise.py --results-dir results/layerwise/ --output-dir results/layerwise/plots/

Options:
    --results-dir DIR   Directory with layerwise probe JSON files (required)
    --output-dir DIR    Directory for output PNGs (default: <results-dir>/plots/)
    --probe NAME        Generate plots for one specific probe only
    --category CAT      Generate correct-vs-incorrect for one specific psych category
    --dpi N             DPI for saved images (default: 300)
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.layerwise_viz import (
    generate_all_viz,
    plot_p_correct_curve,
    plot_psych_heatmap,
    plot_correct_vs_incorrect,
    plot_convergence_comparison,
    plot_multi_probe_convergence,
    plot_computation_regions,
    _load_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate layerwise probe visualization plots")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory with layerwise probe JSON files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for PNGs (default: <results-dir>/plots/)")
    parser.add_argument("--probe", type=str, default=None,
                        help="Generate plots for one specific probe only")
    parser.add_argument("--category", type=str, default=None,
                        help="Correct-vs-incorrect for one psych category")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for saved images")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"

    if not results_dir.exists():
        print(f"Error: results directory does not exist: {results_dir}")
        sys.exit(1)

    # Archive previous plots if output dir has PNGs
    if output_dir.exists() and any(output_dir.glob("*.png")):
        from scripts.archive_utils import archive_previous_run
        archive_previous_run(output_dir, file_glob="*.png")

    if args.probe:
        # Single probe mode
        probe_file = results_dir / f"{args.probe}.json"
        if not probe_file.exists():
            # Try with _layerwise suffix
            probe_file = results_dir / f"{args.probe}_layerwise.json"
        if not probe_file.exists():
            print(f"Error: probe file not found: {probe_file}")
            sys.exit(1)

        result = _load_json(probe_file)
        if not result:
            sys.exit(1)

        output_dir.mkdir(parents=True, exist_ok=True)
        name = args.probe

        if args.category:
            plot_correct_vs_incorrect(result, args.category,
                                      output_dir / f"cvi_{name}_{args.category}.png")
        else:
            plot_p_correct_curve(result, output_dir / f"p_correct_{name}.png")
            plot_psych_heatmap(result, output_dir / f"psych_heatmap_{name}.png")
            if result.get("category_convergence"):
                plot_convergence_comparison(result,
                                           output_dir / f"convergence_{name}.png")
    else:
        # Full batch mode
        generate_all_viz(results_dir, output_dir)


if __name__ == "__main__":
    main()
