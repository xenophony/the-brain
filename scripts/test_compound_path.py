#!/usr/bin/env python3
"""
Quick test of compound layer paths against specific probes.

Tests whether combining multiple circuit modifications is additive.

Usage:
  python scripts/test_compound_path.py --model models/Qwen3-30B-A3B-exl2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sweep.runner import build_optimized_path
from probes.registry import get_probe


def run_path(model, path, probe_names, label, repeats=1):
    """Run probes on a specific layer path and return scores.

    If repeats > 1, runs multiple times and averages for noise reduction.
    """
    model.set_layer_path(path)
    all_runs = []
    for _ in range(repeats):
        scores = {}
        for name in probe_names:
            probe = get_probe(name)
            result = probe.run(model)
            if isinstance(result, dict):
                scores[name] = result["score"]
                if "p_correct" in result:
                    scores[f"{name}_pcorrect"] = result["p_correct"]
            else:
                scores[name] = result
        all_runs.append(scores)

    if repeats == 1:
        return all_runs[0]

    # Average across runs
    avg = {}
    for key in all_runs[0]:
        vals = [r[key] for r in all_runs if isinstance(r.get(key), (int, float))]
        if vals:
            avg[key] = sum(vals) / len(vals)
    return avg


def main():
    parser = argparse.ArgumentParser(description="Test compound layer paths")
    parser.add_argument("--model", required=True)
    parser.add_argument("--probes", nargs="+", default=["math"],
                        help="Probes to test (default: math)")
    parser.add_argument("--full", action="store_true",
                        help="Use all probe items (no max_items limit) for less noisy results")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Run each path N times and average (reduces noise)")
    parser.add_argument("--regions", nargs="+", default=None,
                        help="Circuit regions as i-j pairs, e.g. --regions 1-3 27-28 44-48")
    args = parser.parse_args()

    # Load model
    from sweep.exllama_adapter import ExLlamaV2LayerAdapter
    model = ExLlamaV2LayerAdapter(args.model)
    N = model.num_layers

    probe_names = args.probes

    # Set full items mode on all probes
    if args.full:
        for name in probe_names:
            probe = get_probe(name)
            probe.max_items = None
            n = len(probe.ITEMS) if hasattr(probe, 'ITEMS') else '?'
            print(f"  {name}: max_items=None ({n} items)")
        print()
    normal_path = list(range(N))

    # Parse regions from CLI or use defaults
    if args.regions:
        regions = {}
        for r in args.regions:
            parts = r.split("-")
            i, j = int(parts[0]), int(parts[1])
            regions[r] = (i, j)
    else:
        # Default: key circuits from logprob sweep findings
        regions = {
            "1-3": (1, 3),
            "27-28": (27, 28),
            "44-48": (44, 48),
        }

    print(f"Regions: {list(regions.keys())}")

    # Build all combinations
    from itertools import combinations
    paths = [("baseline", normal_path, [], [])]

    region_names = sorted(regions.keys())
    for r in range(1, len(region_names) + 1):
        for combo in combinations(region_names, r):
            label = "dup " + " + ".join(combo)
            dup_regions = [regions[c] for c in combo]
            paths.append((label, None, [], dup_regions))

    results = []
    for label, explicit_path, skip_regions, dup_regions in paths:
        if explicit_path is not None:
            path = explicit_path
        else:
            path = build_optimized_path(N, skip_regions, dup_regions)

        print(f"\n{'='*60}")
        print(f"Path: {label}")
        print(f"  Length: {len(path)} layers (normal={N})")
        if dup_regions:
            print(f"  Duplicated: {dup_regions}")
        if skip_regions:
            print(f"  Skipped: {skip_regions}")

        scores = run_path(model, path, probe_names, label, repeats=args.repeats)
        results.append((label, scores))

        for name, score in sorted(scores.items()):
            print(f"  {name}: {score:.4f}")

    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    baseline_scores = results[0][1]

    print(f"{'Path':<30s}", end="")
    for name in sorted(baseline_scores.keys()):
        print(f"  {name:>15s}", end="")
    print()

    for label, scores in results:
        print(f"{label:<30s}", end="")
        for name in sorted(baseline_scores.keys()):
            val = scores.get(name, 0.0)
            base = baseline_scores.get(name, 0.0)
            delta = val - base
            if label == "baseline":
                print(f"  {val:>15.4f}", end="")
            else:
                print(f"  {delta:>+14.4f}", end="")
        print()


if __name__ == "__main__":
    main()
