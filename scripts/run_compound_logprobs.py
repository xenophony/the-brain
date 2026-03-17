#!/usr/bin/env python3
"""
Run logprob probes on compound path candidates.

Reads compound_candidates.json, runs all logprob probes on each
compound path, captures psych signal, outputs ranked results.

Usage:
  python scripts/run_compound_logprobs.py \
    --model models/Qwen3-30B-A3B-exl2 \
    --candidates results/analysis/compound_candidates.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from probes.registry import get_probe, list_probes, BaseLogprobProbe


DEFAULT_PROBES = [p for p in list_probes() if "logprob" in p]


def run_path_probes(model, path: list[int], probe_names: list[str],
                    capture_psych: bool = True) -> dict:
    """Run all probes on a specific layer path. Returns scores dict."""
    model.set_layer_path(path)
    scores = {}

    for name in probe_names:
        probe = get_probe(name)
        probe.capture_psych = capture_psych
        try:
            result = probe.run(model)
        except Exception as e:
            print(f"    {name} error: {e}")
            result = {"score": 0.0}

        if isinstance(result, dict):
            scores[name] = result.get("score", 0.0)
            for key in ["p_correct", "easy_score", "hard_score",
                        "p_correct_easy", "p_correct_hard"]:
                if key in result:
                    k = key.replace("p_correct", "pcorrect")
                    scores[f"{name}_{k}"] = result[key]
            # Psych fields
            for key, val in result.items():
                if key.startswith("psych_") and isinstance(val, (int, float)):
                    scores[f"{name}_{key}"] = val
        else:
            scores[name] = float(result)

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Run logprob probes on compound path candidates")
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidates", required=True,
                        help="compound_candidates.json from build_compound_candidates.py")
    parser.add_argument("--probes", nargs="+", default=None,
                        help="Specific probes to run (default: all logprob probes)")
    parser.add_argument("--psych", action="store_true", default=True,
                        help="Capture psych signal (default: True)")
    parser.add_argument("--no-psych", action="store_true",
                        help="Disable psych capture")
    args = parser.parse_args()

    capture_psych = not args.no_psych

    # Load candidates
    with open(args.candidates) as f:
        data = json.load(f)
    candidates = data["candidates"]
    print(f"Loaded {len(candidates)} candidates from {args.candidates}")

    # Resolve probes
    probe_names = args.probes or DEFAULT_PROBES
    probe_names = [p for p in probe_names if p in list_probes()]
    print(f"Probes: {probe_names}")

    # Load model
    from sweep.exllama_adapter import ExLlamaV2LayerAdapter
    model = ExLlamaV2LayerAdapter(args.model)
    N = model.num_layers

    # Run baseline first
    baseline_candidate = next(
        (c for c in candidates if c["label"] == "baseline"), None)
    if not baseline_candidate:
        baseline_candidate = {
            "label": "baseline",
            "path": list(range(N)),
            "dup_regions": [],
            "skip_regions": [],
        }

    print(f"\nRunning baseline...")
    t0 = time.time()
    baseline_scores = run_path_probes(
        model, baseline_candidate["path"], probe_names, capture_psych)
    baseline_time = time.time() - t0
    print(f"  Baseline: {baseline_time:.1f}s")

    # Run all candidates
    results = []
    non_baseline = [c for c in candidates if c["label"] != "baseline"]

    for idx, candidate in enumerate(non_baseline):
        t0 = time.time()
        path = candidate["path"]
        scores = run_path_probes(model, path, probe_names, capture_psych)
        elapsed = time.time() - t0

        # Compute deltas
        deltas = {}
        for key in scores:
            if isinstance(scores[key], (int, float)):
                base = baseline_scores.get(key, 0.0)
                if isinstance(base, (int, float)):
                    deltas[key] = round(scores[key] - base, 6)

        # Compute ranking metrics from main probe deltas only
        main_probes = [p for p in probe_names if p in deltas]
        probe_deltas = [deltas.get(p, 0.0) for p in main_probes]
        combined_delta = sum(probe_deltas)
        min_delta = min(probe_deltas) if probe_deltas else 0.0
        n_improved = sum(1 for d in probe_deltas if d > 0.01)
        n_degraded = sum(1 for d in probe_deltas if d < -0.01)

        result = {
            "label": candidate["label"],
            "dup_regions": candidate.get("dup_regions", []),
            "skip_regions": candidate.get("skip_regions", []),
            "n_layers": len(path),
            "combined_delta": round(combined_delta, 4),
            "min_delta": round(min_delta, 4),
            "n_improved": n_improved,
            "n_degraded": n_degraded,
            "scores": scores,
            "deltas": deltas,
            "runtime": round(elapsed, 2),
        }
        results.append(result)

        flag = " ***" if min_delta >= 0 and n_improved >= 3 else ""
        sign = "+" if combined_delta >= 0 else ""
        print(f"  [{idx+1}/{len(non_baseline)}] {candidate['label']:40s} "
              f"delta={sign}{combined_delta:.4f} min={min_delta:+.4f} "
              f"({n_improved}↑ {n_degraded}↓){flag} | {elapsed:.1f}s")

    # Sort by min_delta first (no degradation), then by combined
    results.sort(key=lambda x: (x["min_delta"], x["combined_delta"]), reverse=True)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"compound_results_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "model": args.model,
        "probes": probe_names,
        "capture_psych": capture_psych,
        "n_candidates": len(results),
        "baseline_scores": baseline_scores,
        "results": results,
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    # Print top results — ranked by "no degradation + most improvement"
    print(f"\n{'='*60}")
    print("SAFE PATHS (no probe degraded, ranked by improvement count)")
    print(f"{'='*60}")
    safe = [r for r in results if r["min_delta"] >= -0.01]
    for r in safe[:15]:
        print(f"  {r['label']:40s} min={r['min_delta']:+.4f} "
              f"combined={r['combined_delta']:+.4f} "
              f"({r['n_improved']}↑ {r['n_layers']}L)")
    if not safe:
        print("  None — all paths degrade at least one probe.")

    print(f"\n{'='*60}")
    print("BEST TRADEOFF PATHS (highest combined, accepting some degradation)")
    print(f"{'='*60}")
    by_combined = sorted(results, key=lambda x: x["combined_delta"], reverse=True)
    for r in by_combined[:10]:
        print(f"  {r['label']:40s} combined={r['combined_delta']:+.4f} "
              f"min={r['min_delta']:+.4f} ({r['n_improved']}↑ {r['n_degraded']}↓)")

    # Per-domain best
    print(f"\n{'='*60}")
    print("BEST PATH PER DOMAIN")
    print(f"{'='*60}")
    for probe in probe_names:
        best = max(results, key=lambda r: r["deltas"].get(probe, -999))
        d = best["deltas"].get(probe, 0)
        if d > 0:
            print(f"  {probe:30s} {best['label']:30s} delta={d:+.4f}")

    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
