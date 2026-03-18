#!/usr/bin/env python
"""
CLI runner for layerwise probe analysis.

Usage:
    python scripts/run_layerwise.py --model PATH [--probes NAMES] [--layer-path i,j] [--output-dir DIR]
    python scripts/run_layerwise.py --mock [--probes NAMES] [--output-dir DIR]

Options:
    --model PATH        Model path (required unless --mock)
    --probes NAMES      Comma-separated probe names (default: all layerwise probes)
    --layer-path i,j    Optional (i,j) config to test (default: baseline 0..N-1)
    --output-dir DIR    Output directory (default: results/layerwise/)
    --mock              Use MockAdapter for testing
    --max-items N       Max items per probe (default: 8)
    --compare i,j       Compare baseline vs this config (runs both, outputs diff)
    --no-psych          Disable psycholinguistic signal capture
    --mode MODE         MockAdapter mode: random, perfect, terrible, sycophantic (default: random)
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_ij(s: str) -> tuple[int, int]:
    """Parse 'i,j' string to (int, int)."""
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected i,j format, got: {s}")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="Run layerwise probe analysis for fMRI-style circuit mapping")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--probes", type=str, default=None,
                        help="Comma-separated probe names (default: all)")
    parser.add_argument("--layer-path", type=parse_ij, default=None,
                        help="(i,j) config to test")
    parser.add_argument("--output-dir", type=str, default="results/layerwise/")
    parser.add_argument("--mock", action="store_true",
                        help="Use MockAdapter for testing")
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--compare", type=parse_ij, default=None,
                        help="Compare baseline vs this (i,j) config")
    parser.add_argument("--no-psych", action="store_true",
                        help="Disable psycholinguistic capture")
    parser.add_argument("--mode", type=str, default="random",
                        help="MockAdapter mode (default: random)")

    args = parser.parse_args()

    if not args.model and not args.mock:
        parser.error("Either --model or --mock is required")

    # Load model
    if args.mock:
        from sweep.mock_adapter import MockAdapter
        model = MockAdapter(mode=args.mode, seed=42)
        print(f"Using MockAdapter (mode={args.mode}, {model.num_layers} layers)")
    else:
        from sweep.exllama_adapter import ExLlamaV2LayerAdapter
        model = ExLlamaV2LayerAdapter(args.model)
        print(f"Loaded model: {model.num_layers} layers")

    # Import and discover layerwise probes
    from probes.layerwise_registry import list_layerwise_probes, get_layerwise_probe
    import probes.layerwise  # trigger auto-discovery

    available = list_layerwise_probes()
    if args.probes:
        probe_names = [p.strip() for p in args.probes.split(",")]
        # Allow shorthand: "causal" -> "causal_layerwise"
        resolved = []
        for name in probe_names:
            if name in available:
                resolved.append(name)
            elif f"{name}_layerwise" in available:
                resolved.append(f"{name}_layerwise")
            else:
                print(f"Warning: probe '{name}' not found. Available: {available}")
        probe_names = resolved
    else:
        probe_names = available

    if not probe_names:
        print("No probes to run. Available layerwise probes:")
        for p in available:
            print(f"  - {p}")
        return

    print(f"\nRunning {len(probe_names)} layerwise probes: {probe_names}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build layer path
    n_layers = model.num_layers
    if args.layer_path:
        i, j = args.layer_path
        # Build duplicate path: 0..j-1, i..N-1
        layer_path = list(range(j)) + list(range(i, n_layers))
        print(f"Layer path: dup({i},{j}) = {len(layer_path)} steps "
              f"(layers {i}-{j-1} duplicated)")
        model.set_layer_path(layer_path)
    else:
        layer_path = None  # baseline

    # Run probes
    all_results = {}
    t0 = time.time()

    for probe_name in probe_names:
        probe = get_layerwise_probe(probe_name)
        probe.max_items = args.max_items
        probe.capture_psych = not args.no_psych

        print(f"\n--- {probe_name} ---")
        tp = time.time()

        try:
            result = probe.run(model)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - tp
        score = result.get("score", 0.0)
        p_correct = result.get("p_correct", 0.0)
        conv = result.get("mean_convergence_layer", "N/A")
        region = result.get("computation_region", (0, 0))

        print(f"  Score: {score:.4f} | p_correct: {p_correct:.4f}")
        print(f"  Convergence layer: {conv}")
        print(f"  Computation region: layers {region[0]}-{region[1]}")
        print(f"  Surprises: {len(result.get('surprise_layers', []))}")
        print(f"  Time: {elapsed:.1f}s")

        all_results[probe_name] = result

        # Save individual probe result
        probe_file = output_dir / f"{probe_name}.json"
        with open(probe_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Completed {len(all_results)}/{len(probe_names)} probes in {total_time:.1f}s")

    # Cross-probe analysis
    if len(all_results) > 1:
        from analysis.layerwise_analysis import generate_layerwise_report
        report_path = output_dir / "layerwise_report.json"
        report = generate_layerwise_report(all_results, report_path)
        print(f"\nCross-probe report saved to {report_path}")

        # Print cross-probe highlights
        cross = report.get("cross_probe", {})
        overlaps = cross.get("region_overlaps", [])
        if overlaps:
            print(f"\nShared computation regions ({len(overlaps)} overlaps):")
            for ov in overlaps[:5]:
                print(f"  {ov['probes'][0]} + {ov['probes'][1]}: "
                      f"layers {ov['overlap_region'][0]}-{ov['overlap_region'][1]}")

        shared_surprises = cross.get("shared_surprise_layers", {})
        if shared_surprises:
            print(f"\nShared surprise layers ({len(shared_surprises)}):")
            for layer, probes_list in sorted(shared_surprises.items()):
                probe_str = ", ".join(p["probe"] for p in probes_list)
                print(f"  Layer {layer}: {probe_str}")

    # Comparison mode
    if args.compare:
        ci, cj = args.compare
        print(f"\n{'='*60}")
        print(f"Comparing baseline vs dup({ci},{cj})...")

        comp_path = list(range(cj)) + list(range(ci, n_layers))
        model.set_layer_path(comp_path)

        comp_results = {}
        for probe_name in probe_names:
            if probe_name not in all_results:
                continue
            probe = get_layerwise_probe(probe_name)
            probe.max_items = args.max_items
            probe.capture_psych = not args.no_psych

            try:
                comp_result = probe.run(model)
                comp_results[probe_name] = comp_result

                base = all_results[probe_name]
                delta_score = comp_result["score"] - base["score"]
                delta_p = comp_result["p_correct"] - base["p_correct"]
                print(f"  {probe_name}: score {delta_score:+.4f} | "
                      f"p_correct {delta_p:+.4f}")
            except Exception as e:
                print(f"  {probe_name}: ERROR {e}")

        # Save comparison
        comp_file = output_dir / f"comparison_{ci}_{cj}.json"
        with open(comp_file, "w") as f:
            json.dump({
                "baseline": {k: {"score": v["score"], "p_correct": v["p_correct"],
                                  "mean_convergence_layer": v.get("mean_convergence_layer")}
                             for k, v in all_results.items()},
                "config": {"i": ci, "j": cj},
                "comparison": {k: {"score": v["score"], "p_correct": v["p_correct"],
                                    "mean_convergence_layer": v.get("mean_convergence_layer")}
                               for k, v in comp_results.items()},
            }, f, indent=2, default=str)
        print(f"\nComparison saved to {comp_file}")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
