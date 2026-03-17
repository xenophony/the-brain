#!/usr/bin/env python3
"""
Build compound path candidates from sweep data.

Reads duplicate and/or skip sweep results, identifies top performing
regions per domain, generates compound paths (combine best duplicates
with best skips), and outputs candidates for logprob verification.

No GPU needed — pure data analysis.

Usage:
  python scripts/build_compound_candidates.py \
    --dup results/analysis/sweep_results_merged.json \
    --skip results/incoming/skip_sweep.json \
    --output results/analysis/compound_candidates.json

  # Or with just one sweep type
  python scripts/build_compound_candidates.py \
    --dup results/analysis/sweep_results_merged.json
"""

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def get_probe_names(results: list[dict]) -> list[str]:
    """Extract probe names (exclude sub-fields)."""
    baseline = next((r for r in results if r["i"] == 0 and r["j"] == 0), None)
    if not baseline:
        return []
    return [k for k in baseline["probe_scores"]
            if not k.startswith("_")
            and not k.endswith("_easy") and not k.endswith("_hard")
            and "_pcorrect" not in k and "_psych_" not in k]


def find_top_regions(results: list[dict], mode: str,
                     top_n: int = 5, min_delta: float = 0.10,
                     ) -> dict:
    """Find top performing (i,j) regions per probe.

    For duplicate mode: positive delta = good (amplify)
    For skip mode: positive delta = good (removing helps)

    Returns {probe_name: [(i, j, delta, pcorrect_delta), ...]}
    """
    probes = get_probe_names(results)
    configs = [r for r in results if not (r["i"] == 0 and r["j"] == 0)]

    top_regions = {}
    for probe in probes:
        pc_key = f"{probe}_pcorrect"
        ranked = []
        for r in configs:
            delta = r["probe_deltas"].get(probe, 0.0)
            pc_delta = r["probe_deltas"].get(pc_key, 0.0)
            if delta >= min_delta:
                ranked.append({
                    "i": r["i"], "j": r["j"],
                    "delta": round(delta, 4),
                    "pcorrect_delta": round(pc_delta, 4) if pc_delta else 0.0,
                })
        ranked.sort(key=lambda x: x["delta"], reverse=True)
        if ranked:
            top_regions[probe] = ranked[:top_n]

    return top_regions


def extract_layer_regions(top_regions: dict, max_regions: int = 3,
                          ) -> list[tuple[int, int]]:
    """Extract unique (i,j) layer regions from top performing configs.

    Deduplicates and returns the most frequently appearing regions
    across all probes.
    """
    region_scores = defaultdict(float)
    for probe, configs in top_regions.items():
        for cfg in configs:
            key = (cfg["i"], cfg["j"])
            region_scores[key] += cfg["delta"]

    # Sort by total delta across probes
    sorted_regions = sorted(region_scores.items(),
                            key=lambda x: x[1], reverse=True)
    return [r[0] for r in sorted_regions[:max_regions]]


def build_compound_paths(
    dup_regions: list[tuple[int, int]],
    skip_regions: list[tuple[int, int]],
    n_layers: int = 48,
    max_compound_size: int = 3,
) -> list[dict]:
    """Generate compound path candidates from top regions.

    Builds all combinations up to max_compound_size of:
      - Duplicate-only paths
      - Skip-only paths
      - Mixed duplicate + skip paths
    """
    from sweep.runner import build_optimized_path

    candidates = []

    # Baseline
    candidates.append({
        "label": "baseline",
        "dup_regions": [],
        "skip_regions": [],
        "path": list(range(n_layers)),
        "n_layers": n_layers,
    })

    # Individual duplicates
    for i, j in dup_regions:
        path = build_optimized_path(n_layers, [], [(i, j)])
        candidates.append({
            "label": f"dup({i},{j})",
            "dup_regions": [(i, j)],
            "skip_regions": [],
            "path": path,
            "n_layers": len(path),
        })

    # Individual skips
    for i, j in skip_regions:
        path = build_optimized_path(n_layers, [(i, j)], [])
        candidates.append({
            "label": f"skip({i},{j})",
            "dup_regions": [],
            "skip_regions": [(i, j)],
            "path": path,
            "n_layers": len(path),
        })

    # Duplicate combinations (2-way)
    for combo in combinations(dup_regions, min(2, len(dup_regions))):
        if len(combo) < 2:
            continue
        # Check for overlapping regions
        all_layers = set()
        overlap = False
        for i, j in combo:
            region = set(range(i, j))
            if region & all_layers:
                overlap = True
                break
            all_layers |= region
        if overlap:
            continue

        path = build_optimized_path(n_layers, [], list(combo))
        label = " + ".join(f"dup({i},{j})" for i, j in combo)
        candidates.append({
            "label": label,
            "dup_regions": list(combo),
            "skip_regions": [],
            "path": path,
            "n_layers": len(path),
        })

    # Skip combinations (2-way)
    for combo in combinations(skip_regions, min(2, len(skip_regions))):
        if len(combo) < 2:
            continue
        all_layers = set()
        overlap = False
        for i, j in combo:
            region = set(range(i, j))
            if region & all_layers:
                overlap = True
                break
            all_layers |= region
        if overlap:
            continue

        path = build_optimized_path(n_layers, list(combo), [])
        label = " + ".join(f"skip({i},{j})" for i, j in combo)
        candidates.append({
            "label": label,
            "dup_regions": [],
            "skip_regions": list(combo),
            "path": path,
            "n_layers": len(path),
        })

    # Mixed: each duplicate with each skip (if non-overlapping)
    for di, dj in dup_regions:
        dup_layers = set(range(di, dj))
        for si, sj in skip_regions:
            skip_layers = set(range(si, sj))
            if dup_layers & skip_layers:
                continue  # overlapping — skip takes priority but avoid confusion
            path = build_optimized_path(n_layers, [(si, sj)], [(di, dj)])
            candidates.append({
                "label": f"dup({di},{dj}) + skip({si},{sj})",
                "dup_regions": [(di, dj)],
                "skip_regions": [(si, sj)],
                "path": path,
                "n_layers": len(path),
            })

    # Triple combos: 2 dup + 1 skip, 1 dup + 2 skip
    if len(dup_regions) >= 2 and skip_regions:
        for dup_combo in combinations(dup_regions, 2):
            dup_layers = set()
            for i, j in dup_combo:
                dup_layers |= set(range(i, j))
            for si, sj in skip_regions:
                skip_layers = set(range(si, sj))
                if dup_layers & skip_layers:
                    continue
                path = build_optimized_path(
                    n_layers, [(si, sj)], list(dup_combo))
                label = " + ".join(f"dup({i},{j})" for i, j in dup_combo)
                label += f" + skip({si},{sj})"
                candidates.append({
                    "label": label,
                    "dup_regions": list(dup_combo),
                    "skip_regions": [(si, sj)],
                    "path": path,
                    "n_layers": len(path),
                })

    if dup_regions and len(skip_regions) >= 2:
        for skip_combo in combinations(skip_regions, 2):
            skip_layers = set()
            overlap = False
            for i, j in skip_combo:
                region = set(range(i, j))
                if region & skip_layers:
                    overlap = True
                    break
                skip_layers |= region
            if overlap:
                continue
            for di, dj in dup_regions:
                dup_layers = set(range(di, dj))
                if dup_layers & skip_layers:
                    continue
                path = build_optimized_path(
                    n_layers, list(skip_combo), [(di, dj)])
                label = f"dup({di},{dj})"
                label += " + " + " + ".join(
                    f"skip({i},{j})" for i, j in skip_combo)
                candidates.append({
                    "label": label,
                    "dup_regions": [(di, dj)],
                    "skip_regions": list(skip_combo),
                    "path": path,
                    "n_layers": len(path),
                })

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Build compound path candidates from sweep data")
    parser.add_argument("--dup", type=str, default=None,
                        help="Duplicate sweep results JSON")
    parser.add_argument("--skip", type=str, default=None,
                        help="Skip sweep results JSON")
    parser.add_argument("--output", type=str,
                        default="results/analysis/compound_candidates.json")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Top N regions per probe to consider")
    parser.add_argument("--max-regions", type=int, default=5,
                        help="Max unique regions to combine")
    parser.add_argument("--min-delta", type=float, default=0.10,
                        help="Minimum delta to consider a region")
    parser.add_argument("--n-layers", type=int, default=48,
                        help="Total model layers")
    args = parser.parse_args()

    if not args.dup and not args.skip:
        print("ERROR: Provide at least one of --dup or --skip")
        sys.exit(1)

    dup_regions = []
    skip_regions = []
    dup_top = {}
    skip_top = {}

    if args.dup:
        print(f"Loading duplicate sweep: {args.dup}")
        dup_results = load_results(args.dup)
        dup_top = find_top_regions(dup_results, "duplicate",
                                   top_n=args.top_n, min_delta=args.min_delta)
        dup_regions = extract_layer_regions(dup_top, args.max_regions)
        print(f"  Top duplicate regions: {dup_regions}")
        for probe, configs in dup_top.items():
            if configs:
                best = configs[0]
                print(f"    {probe}: ({best['i']},{best['j']}) "
                      f"delta={best['delta']:+.4f}")

    if args.skip:
        print(f"Loading skip sweep: {args.skip}")
        skip_results = load_results(args.skip)
        skip_top = find_top_regions(skip_results, "skip",
                                    top_n=args.top_n, min_delta=args.min_delta)
        skip_regions = extract_layer_regions(skip_top, args.max_regions)
        print(f"  Top skip regions: {skip_regions}")
        for probe, configs in skip_top.items():
            if configs:
                best = configs[0]
                print(f"    {probe}: ({best['i']},{best['j']}) "
                      f"delta={best['delta']:+.4f}")

    # Build compound candidates
    print(f"\nBuilding compound candidates...")
    candidates = build_compound_paths(
        dup_regions, skip_regions, n_layers=args.n_layers)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "dup_source": args.dup,
        "skip_source": args.skip,
        "dup_top_regions": {p: c[:3] for p, c in dup_top.items()},
        "skip_top_regions": {p: c[:3] for p, c in skip_top.items()},
        "dup_layer_regions": [list(r) for r in dup_regions],
        "skip_layer_regions": [list(r) for r in skip_regions],
        "n_candidates": len(candidates),
        "candidates": candidates,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{len(candidates)} compound candidates saved to {output_path}")
    print(f"\nBreakdown:")
    dup_only = sum(1 for c in candidates
                   if c["dup_regions"] and not c["skip_regions"])
    skip_only = sum(1 for c in candidates
                    if c["skip_regions"] and not c["dup_regions"])
    mixed = sum(1 for c in candidates
                if c["dup_regions"] and c["skip_regions"])
    print(f"  Baseline: 1")
    print(f"  Duplicate-only: {dup_only}")
    print(f"  Skip-only: {skip_only}")
    print(f"  Mixed (dup+skip): {mixed}")

    # Preview top candidates by path length efficiency
    print(f"\nMost efficient candidates (fewest extra/fewer layers):")
    for c in sorted(candidates, key=lambda x: abs(x["n_layers"] - 48))[:10]:
        diff = c["n_layers"] - 48
        sign = "+" if diff > 0 else ""
        print(f"  {c['label']:40s} {sign}{diff} layers ({c['n_layers']} total)")


if __name__ == "__main__":
    main()
