#!/usr/bin/env python3
"""
Generation verification on selected compound paths.

Runs actual generation probes (not logprobs) on specific paths
identified by the compound analysis pipeline. Captures real model
output to verify logprob signal translates to output improvement.

Usage:
  python scripts/run_generation_verify.py \
    --model models/Qwen3-30B-A3B-exl2 \
    --full --repeats 3
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from probes.registry import get_probe, list_probes
from sweep.runner import build_optimized_path


# Paths to verify — from compound analysis + domain champions
# Each: (label, dup_regions, skip_regions, extra_path_ops)
VERIFICATION_PATHS = [
    # Compound winners
    {
        "label": "dup(0,2)+dup(2,3)+skip(29,34)",
        "dup": [(0, 2), (2, 3)],
        "skip": [(29, 34)],
        "reason": "Overall winner: +1.97, 9 probes up, min=-0.03",
    },
    {
        "label": "dup(2,3)+skip(1,2)",
        "dup": [(2, 3)],
        "skip": [(1, 2)],
        "reason": "Zero-cost: sentiment +0.50, logic +0.12, 48 layers",
    },
    {
        "label": "dup(2,3)+skip(3,5)",
        "dup": [(2, 3)],
        "skip": [(3, 5)],
        "reason": "Minimal tradeoff: causal +0.12, consistency +0.12",
    },
    {
        "label": "dup(2,3)+skip(29,34)",
        "dup": [(2, 3)],
        "skip": [(29, 34)],
        "reason": "Routing +0.50, sentiment +0.50, causal +0.25",
    },
    # Single-layer domain champions
    {
        "label": "dup(3,5) [judgement]",
        "dup": [(3, 5)],
        "skip": [],
        "reason": "Best single for judgement: +0.50",
    },
    {
        "label": "dup(25,26) [spatial]",
        "dup": [(25, 26)],
        "skip": [],
        "reason": "Best single for pong_strategic: +0.42",
    },
    {
        "label": "dup(35,38) [error]",
        "dup": [(35, 38)],
        "skip": [],
        "reason": "Best single for error: +0.54",
    },
    {
        "label": "skip(37,40) [implication]",
        "dup": [],
        "skip": [(37, 40)],
        "reason": "Best skip for implication: +0.38",
    },
    # Stacking / feedback experiments
    {
        "label": "layer2 x3",
        "dup": [],
        "skip": [],
        "custom_path": lambda N: [0, 1, 2, 2, 2] + list(range(3, N)),
        "reason": "Triple reasoning amplification",
    },
    {
        "label": "layer2 x3 + skip(29,34)",
        "dup": [],
        "skip": [],
        "custom_path": lambda N: [0, 1, 2, 2, 2] + list(range(3, 29)) + list(range(34, N)),
        "reason": "Triple reasoning + remove dispensable",
    },
    {
        "label": "feedback: 27-28 → 2",
        "dup": [],
        "skip": [],
        "custom_path": lambda N: list(range(29)) + [27, 28, 2] + list(range(3, N)),
        "reason": "Spatial feeds back into reasoning",
    },
    {
        "label": "feedback: 2→2→27-28",
        "dup": [],
        "skip": [],
        "custom_path": lambda N: list(range(3)) + [2, 2] + list(range(3, 29)) + [27, 28] + list(range(29, N)),
        "reason": "Double reasoning then spatial",
    },
    # Hallucination-specific
    {
        "label": "dup(10,16)+skip(0,5) [hallucination]",
        "dup": [(10, 16)],
        "skip": [(0, 5)],
        "reason": "Best for hallucination: +0.75",
    },
    # Language-specific
    {
        "label": "dup(5,11)+skip(0,5)+skip(29,34) [language]",
        "dup": [(5, 11)],
        "skip": [(0, 5), (29, 34)],
        "reason": "Best for language: +0.875",
    },
]

GENERATION_PROBES = ["math", "eq", "language", "spatial_pong_simple", "spatial_pong_strategic"]

# Logprob probes run alongside generation for psych signal capture
LOGPROB_PROBES = [
    "causal_logprob", "logic_logprob", "sentiment_logprob",
    "error_logprob", "hallucination_logprob", "routing_logprob",
    "judgement_logprob", "consistency_logprob",
]


def build_path(entry, n_layers):
    """Build a layer path from a verification entry."""
    if "custom_path" in entry:
        return entry["custom_path"](n_layers)
    return build_optimized_path(n_layers, entry["skip"], entry["dup"])


def main():
    parser = argparse.ArgumentParser(
        description="Generation verification on selected compound paths")
    parser.add_argument("--model", required=True)
    parser.add_argument("--probes", nargs="+", default=None,
                        help=f"Generation probes (default: {GENERATION_PROBES})")
    parser.add_argument("--full", action="store_true",
                        help="Use all probe items (no max_items limit)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Run each path N times and average")
    parser.add_argument("--output", default=None,
                        help="Output file (default: auto-timestamped)")
    parser.add_argument("--no-psych", action="store_true",
                        help="Skip logprob+psych capture (faster, less data)")
    args = parser.parse_args()

    probe_names = args.probes or GENERATION_PROBES
    run_psych = not args.no_psych

    # Load model
    from sweep.exllama_adapter import ExLlamaV2LayerAdapter
    model = ExLlamaV2LayerAdapter(args.model)
    N = model.num_layers

    # Set full items
    if args.full:
        for name in probe_names:
            probe = get_probe(name)
            probe.max_items = None
            import importlib
            mod = importlib.import_module(f"probes.{name}.probe")
            easy = len(getattr(mod, "EASY_ITEMS", []))
            hard = len(getattr(mod, "HARD_ITEMS", []))
            print(f"  {name}: {easy + hard} items (full)")

    # Build all paths
    paths = [("baseline", list(range(N)), "Control — unmodified model")]
    for entry in VERIFICATION_PATHS:
        path = build_path(entry, N)
        paths.append((entry["label"], path, entry["reason"]))

    print(f"\n{len(paths)} paths to test, {len(probe_names)} probes"
          f"{' x' + str(args.repeats) + ' repeats' if args.repeats > 1 else ''}")

    # Run baseline
    baseline_scores = {}
    print(f"\nRunning baseline...")
    model.set_layer_path(paths[0][1])
    for rep in range(args.repeats):
        for name in probe_names:
            probe = get_probe(name)
            if args.full:
                probe.max_items = None
            result = probe.run(model)
            score = result["score"] if isinstance(result, dict) else float(result)
            if name not in baseline_scores:
                baseline_scores[name] = []
            baseline_scores[name].append(score)

    # Logprob + psych baseline
    baseline_psych = {}
    if run_psych:
        print("  Running logprob+psych baseline...")
        for name in LOGPROB_PROBES:
            probe = get_probe(name)
            probe.capture_psych = True
            result = probe.run(model)
            if isinstance(result, dict):
                baseline_psych[name] = result.get("score", 0.0)
                if "p_correct" in result:
                    baseline_psych[f"{name}_pcorrect"] = result["p_correct"]
                for k, v in result.items():
                    if k.startswith("psych_") and isinstance(v, (int, float)):
                        baseline_psych[f"{name}_{k}"] = v

    baseline_avg = {k: sum(v) / len(v) for k, v in baseline_scores.items()}
    print(f"  Baseline gen: {' | '.join(f'{k}={v:.3f}' for k, v in sorted(baseline_avg.items()))}")
    if baseline_psych:
        logprob_summary = {k: v for k, v in baseline_psych.items() if k in LOGPROB_PROBES}
        print(f"  Baseline logprob: {' | '.join(f'{k.replace(\"_logprob\",\"\")}={v:.3f}' for k, v in sorted(logprob_summary.items()))}")

    # Run all paths
    results = []
    for idx, (label, path, reason) in enumerate(paths[1:], 1):
        print(f"\n[{idx}/{len(paths)-1}] {label} ({len(path)}L)")
        print(f"  Reason: {reason}")

        path_scores = {}
        model.set_layer_path(path)

        # Generation probes — capture responses on first rep for spot-checking
        path_samples = {}  # probe -> [sample responses]
        for rep in range(args.repeats):
            for name in probe_names:
                probe = get_probe(name)
                if args.full:
                    probe.max_items = None
                probe.log_responses = True
                try:
                    result = probe.run(model)
                    score = result["score"] if isinstance(result, dict) else float(result)
                    # Capture all responses from first rep
                    if rep == 0 and isinstance(result, dict) and "item_results" in result:
                        path_samples[name] = result["item_results"]
                except Exception as e:
                    print(f"    {name} error: {e}")
                    score = 0.0
                if name not in path_scores:
                    path_scores[name] = []
                path_scores[name].append(score)

        # Logprob + psych pass (single run, no repeats needed — deterministic)
        path_psych = {}
        if run_psych:
            for name in LOGPROB_PROBES:
                probe = get_probe(name)
                probe.capture_psych = True
                try:
                    result = probe.run(model)
                    if isinstance(result, dict):
                        path_psych[name] = result.get("score", 0.0)
                        if "p_correct" in result:
                            path_psych[f"{name}_pcorrect"] = result["p_correct"]
                        for k, v in result.items():
                            if k.startswith("psych_") and isinstance(v, (int, float)):
                                path_psych[f"{name}_{k}"] = v
                except Exception:
                    pass

        avg_scores = {k: sum(v) / len(v) for k, v in path_scores.items()}
        deltas = {k: avg_scores[k] - baseline_avg.get(k, 0) for k in avg_scores}

        # Psych deltas
        psych_deltas = {}
        if path_psych and baseline_psych:
            for k in path_psych:
                base = baseline_psych.get(k, 0)
                if isinstance(base, (int, float)) and isinstance(path_psych[k], (int, float)):
                    psych_deltas[k] = round(path_psych[k] - base, 6)

        # Print per-probe results
        print("  Generation:")
        for name in sorted(probe_names):
            d = deltas.get(name, 0)
            s = avg_scores.get(name, 0)
            b = baseline_avg.get(name, 0)
            marker = " +++" if d > 0.05 else " ---" if d < -0.05 else ""
            print(f"    {name:28s} {b:.3f} -> {s:.3f} ({d:+.3f}){marker}")

        if psych_deltas:
            # Show logprob score deltas
            logprob_deltas = {k: v for k, v in psych_deltas.items() if k in LOGPROB_PROBES}
            if logprob_deltas:
                print("  Logprob:")
                for k in sorted(logprob_deltas):
                    d = logprob_deltas[k]
                    marker = " +++" if d > 0.05 else " ---" if d < -0.05 else ""
                    print(f"    {k.replace('_logprob',''):28s} {d:+.3f}{marker}")

            # Show top psych signal changes
            psych_only = {k: v for k, v in psych_deltas.items() if "_psych_" in k}
            if psych_only:
                top_psych = sorted(psych_only.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                if any(abs(v) > 1e-4 for _, v in top_psych):
                    print("  Psych signal (top changes):")
                    for k, v in top_psych:
                        if abs(v) > 1e-4:
                            # Extract category name
                            cat = k.split("_psych_")[-1] if "_psych_" in k else k
                            probe_prefix = k.split("_psych_")[0].replace("_logprob", "")
                            print(f"    {probe_prefix}/{cat}: {v:+.6f}")

        # Print sample responses for spot-checking (2 per probe, all saved to JSON)
        if path_samples:
            print("  Sample responses:")
            for probe_name, items in path_samples.items():
                good = next((r for r in items if r.get("score", 0) > 0.5), None)
                bad = next((r for r in items if r.get("score", 0) == 0), None)
                if good:
                    resp = good.get("response", good.get("raw_response", ""))[:80]
                    print(f"    [{probe_name}][OK] {repr(resp)}")
                if bad:
                    resp = bad.get("response", bad.get("raw_response", ""))[:80]
                    print(f"    [{probe_name}][FAIL] {repr(resp)}")

        results.append({
            "label": label,
            "reason": reason,
            "n_layers": len(path),
            "scores": avg_scores,
            "deltas": deltas,
            "psych_scores": path_psych,
            "psych_deltas": psych_deltas,
            "sample_responses": path_samples,
            "combined_delta": round(sum(deltas.values()), 4),
            "min_delta": round(min(deltas.values()), 4),
            "n_improved": sum(1 for d in deltas.values() if d > 0.05),
            "n_degraded": sum(1 for d in deltas.values() if d < -0.05),
        })

    # Sort by safety
    results.sort(key=lambda x: (x["min_delta"], x["combined_delta"]), reverse=True)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = Path(args.output) if args.output else output_dir / f"generation_verify_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "model": args.model,
        "generation_probes": probe_names,
        "logprob_probes": LOGPROB_PROBES if run_psych else [],
        "psych_captured": run_psych,
        "full_items": args.full,
        "repeats": args.repeats,
        "n_paths": len(results),
        "baseline_generation": baseline_avg,
        "baseline_psych": baseline_psych,
        "results": results,
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    # Final summary
    print(f"\n{'='*80}")
    print("GENERATION VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"\nBaseline: {' | '.join(f'{k}={v:.3f}' for k, v in sorted(baseline_avg.items()))}")

    print(f"\nSAFE PATHS (min_delta >= -0.05):")
    safe = [r for r in results if r["min_delta"] >= -0.05]
    for r in safe[:10]:
        print(f"  {r['label']:45s} combined={r['combined_delta']:+.3f} "
              f"min={r['min_delta']:+.3f} ({r['n_improved']}↑ {r['n_degraded']}↓)")

    print(f"\nBEST PER DOMAIN:")
    for probe in probe_names:
        best = max(results, key=lambda r: r["deltas"].get(probe, -99))
        d = best["deltas"].get(probe, 0)
        if d > 0:
            print(f"  {probe:30s} {best['label']:35s} {d:+.3f}")

    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
