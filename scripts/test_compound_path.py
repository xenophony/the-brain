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


def run_path(model, path, probe_names, label):
    """Run probes on a specific layer path and return scores."""
    model.set_layer_path(path)
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
    return scores


def main():
    parser = argparse.ArgumentParser(description="Test compound layer paths")
    parser.add_argument("--model", required=True)
    parser.add_argument("--probes", nargs="+", default=["math"],
                        help="Probes to test (default: math)")
    args = parser.parse_args()

    # Load model
    from sweep.exllama_adapter import ExLlamaV2LayerAdapter
    model = ExLlamaV2LayerAdapter(args.model)
    N = model.num_layers

    probe_names = args.probes
    normal_path = list(range(N))

    # Define circuit regions to test (from sweep data)
    regions = {
        "1-3": (1, 3),
        "27-28": (27, 28),
        "44-48": (44, 48),
    }

    # Build all combinations: individual, pairs, and triple
    from itertools import combinations
    paths = [("baseline", normal_path, [], [])]

    region_names = sorted(regions.keys())
    # Singles
    for name in region_names:
        paths.append((f"dup {name}", None, [], [regions[name]]))
    # Pairs
    for combo in combinations(region_names, 2):
        label = " + ".join(f"{c}" for c in combo)
        paths.append((f"dup {label}", None, [], [regions[c] for c in combo]))
    # Triple
    label = " + ".join(region_names)
    paths.append((f"dup {label}", None, [], [regions[c] for c in region_names]))

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

        scores = run_path(model, path, probe_names, label)
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
