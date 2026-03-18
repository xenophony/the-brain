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
    parser.add_argument("--skip-regions", nargs="+", default=None,
                        help="Regions to SKIP (remove), e.g. --skip-regions 5-10 8-12")
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

    # Parse skip regions
    skip_regions_map = {}
    if args.skip_regions:
        for r in args.skip_regions:
            parts = r.split("-")
            i, j = int(parts[0]), int(parts[1])
            skip_regions_map[r] = (i, j)
        print(f"Skip regions: {list(skip_regions_map.keys())}")

    # Build all combinations
    from itertools import combinations
    paths = [("baseline", normal_path, [], [])]

    # Duplicate region combinations
    region_names = sorted(regions.keys())
    for r in range(1, len(region_names) + 1):
        for combo in combinations(region_names, r):
            label = "dup " + " + ".join(combo)
            dup_regions = [regions[c] for c in combo]
            paths.append((label, None, [], dup_regions))

    # Skip region paths (each individually, then all combined)
    if skip_regions_map:
        skip_names = sorted(skip_regions_map.keys())
        for r in range(1, len(skip_names) + 1):
            for combo in combinations(skip_names, r):
                label = "skip " + " + ".join(combo)
                sr = [skip_regions_map[c] for c in combo]
                paths.append((label, None, sr, []))

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

    # Save results
    import json
    from datetime import datetime
    output_dir = Path("results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"compound_test_{timestamp}.json"

    save_data = {
        "timestamp": timestamp,
        "probes": probe_names,
        "regions": {k: list(v) for k, v in regions.items()} if isinstance(next(iter(regions.values())), tuple) else regions,
        "repeats": args.repeats,
        "full_items": args.full,
        "baseline": baseline_scores,
        "results": [
            {
                "path": label,
                "scores": scores,
                "deltas": {k: scores.get(k, 0.0) - baseline_scores.get(k, 0.0)
                           for k in baseline_scores
                           if isinstance(scores.get(k, 0.0), (int, float))}
            }
            for label, scores in results
        ],
    }
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
