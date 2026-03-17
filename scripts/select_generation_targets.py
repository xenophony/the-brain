#!/usr/bin/env python3
"""
Select top compound paths for generation verification.

Reads compound logprob results, selects the best paths per domain
and overall, outputs generation_targets.json for final verification.

No GPU needed.

Usage:
  python scripts/select_generation_targets.py \
    --input results/analysis/compound_results_*.json \
    --top-n 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Select top compound paths for generation verification")
    parser.add_argument("--input", required=True,
                        help="compound_results JSON from run_compound_logprobs.py")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top paths to select overall")
    parser.add_argument("--top-per-domain", type=int, default=2,
                        help="Number of top paths per domain")
    parser.add_argument("--output", default="results/analysis/generation_targets.json")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    results = data["results"]
    probes = data["probes"]
    baseline = data["baseline_scores"]

    print(f"Loaded {len(results)} compound results")
    print(f"Probes: {probes}")

    selected = {}  # label -> result (dedup)

    # Top N overall (by combined delta)
    for r in results[:args.top_n]:
        if r["combined_delta"] > 0:
            selected[r["label"]] = r

    # Top per domain
    for probe in probes:
        domain_ranked = sorted(
            results,
            key=lambda r: r["deltas"].get(probe, -999),
            reverse=True)
        for r in domain_ranked[:args.top_per_domain]:
            if r["deltas"].get(probe, 0) > 0:
                selected[r["label"]] = r

    # Also include best "efficient" paths (smallest layer change)
    efficient = sorted(
        [r for r in results if r["combined_delta"] > 0],
        key=lambda r: abs(r["n_layers"] - 48))
    for r in efficient[:3]:
        selected[r["label"]] = r

    targets = list(selected.values())
    targets.sort(key=lambda x: x["combined_delta"], reverse=True)

    # Build output
    output = {
        "source": args.input,
        "n_targets": len(targets),
        "baseline_scores": baseline,
        "targets": [
            {
                "label": t["label"],
                "path": None,  # will be rebuilt from regions
                "dup_regions": t["dup_regions"],
                "skip_regions": t["skip_regions"],
                "n_layers": t["n_layers"],
                "combined_delta": t["combined_delta"],
                "top_domain_deltas": {
                    p: t["deltas"].get(p, 0.0)
                    for p in probes
                    if t["deltas"].get(p, 0.0) != 0
                },
            }
            for t in targets
        ],
        "generation_probes": [
            "eq", "language", "math",
            "spatial_pong_simple", "spatial_pong_strategic",
        ],
        "instructions": (
            "Run generation probes on these paths to verify logprob signal "
            "translates to real output improvement. Use test_compound_path.py "
            "with --full --repeats 3 for statistical confidence."
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GENERATION TARGETS ({len(targets)} paths)")
    print(f"{'='*60}")
    for t in targets:
        print(f"  {t['label']:40s} delta={t['combined_delta']:+.4f} "
              f"({t['n_layers']} layers)")
        # Show top 3 domain improvements
        domain_deltas = [(p, t["deltas"].get(p, 0)) for p in probes]
        domain_deltas.sort(key=lambda x: x[1], reverse=True)
        for p, d in domain_deltas[:3]:
            if d > 0:
                print(f"    {p}: {d:+.4f}")

    print(f"\nSaved to {out_path}")
    print(f"\nNext step: run generation probes on these {len(targets)} paths")


if __name__ == "__main__":
    main()
