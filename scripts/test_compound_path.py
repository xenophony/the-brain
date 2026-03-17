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

    # Multi-pass circuit amplification test
    # Run specific layers multiple times to see if effects scale
    paths = [("baseline", normal_path, [], [])]

    # Layer 27 through spatial circuit 1x, 2x, 3x, 4x, 5x
    for repeats in range(1, 6):
        label = f"27-28 x{repeats}"
        # Build path manually: normal path but layer 27 repeated
        path = list(range(N))
        # Insert extra copies of layer 27 right after its normal position
        insert_pos = 28  # after layer 27 in the normal path
        for _ in range(repeats - 1):  # -1 because it already runs once
            path.insert(insert_pos, 27)
        paths.append((label, path, [], []))

    # Also test layer 1 (reasoning) multi-pass for comparison
    for repeats in range(1, 6):
        label = f"1-2 x{repeats}"
        path = list(range(N))
        insert_pos = 2
        for _ in range(repeats - 1):
            path.insert(insert_pos, 1)
        paths.append((label, path, [], []))

    # Combined: spatial 27 x3 + reasoning 1 x3
    path = list(range(N))
    # Insert layer 1 copies
    for _ in range(2):
        path.insert(2, 1)
    # Find where 27 is now and insert copies after it
    idx_27 = path.index(27)
    for _ in range(2):
        path.insert(idx_27 + 1, 27)
    paths.append(("1 x3 + 27 x3", path, [], []))

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
