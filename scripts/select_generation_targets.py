#!/usr/bin/env python3
"""
Select top compound paths for generation verification.

Reads compound logprob results, selects winning paths using safe
ranking (no probe degraded), and for each path identifies ONLY the
generation probes that showed logprob signal — no wasted GPU time.

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

# Mapping from logprob probes to their generation probe equivalents
LOGPROB_TO_GENERATION = {
    "causal_logprob": None,  # no generation equivalent yet
    "logic_logprob": None,
    "sentiment_logprob": None,
    "error_logprob": None,
    "routing_logprob": None,
    "judgement_logprob": None,
    "sycophancy_logprob": None,
    "consistency_logprob": None,
    "temporal_logprob": None,
    "language_logprob": "language",
    "pong_simple_logprob": "spatial_pong_simple",
    "pong_strategic_logprob": "spatial_pong_strategic",
    "implication_logprob": None,
    "negation_logprob": None,
    "hallucination_logprob": None,
    "psych_conflict_logprob": None,
    "psych_difficulty_logprob": None,
    "psych_unknowable_logprob": None,
    "psych_urgency_logprob": None,
    # Always include these core generation probes if any signal found
    "_always": ["math", "eq"],
}


def get_relevant_generation_probes(deltas: dict, probes: list[str],
                                   threshold: float = 0.05) -> list[str]:
    """Determine which generation probes to run based on logprob signal.

    Returns only generation probes where the corresponding logprob
    probe showed delta > threshold. Always includes math and eq
    if any probe showed signal.
    """
    gen_probes = set()
    has_any_signal = False

    for probe in probes:
        delta = deltas.get(probe, 0.0)
        if abs(delta) > threshold:
            has_any_signal = True
            gen_probe = LOGPROB_TO_GENERATION.get(probe)
            if gen_probe:
                gen_probes.add(gen_probe)

    # Always include core probes if any signal detected
    if has_any_signal:
        gen_probes.update(LOGPROB_TO_GENERATION["_always"])

    return sorted(gen_probes) if gen_probes else []


def main():
    parser = argparse.ArgumentParser(
        description="Select top compound paths for generation verification")
    parser.add_argument("--input", required=True,
                        help="compound_results JSON from run_compound_logprobs.py")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top safe paths to select")
    parser.add_argument("--top-tradeoff", type=int, default=5,
                        help="Number of best-tradeoff paths to include")
    parser.add_argument("--top-per-domain", type=int, default=2,
                        help="Number of top paths per domain")
    parser.add_argument("--signal-threshold", type=float, default=0.05,
                        help="Min logprob delta to consider a probe relevant")
    parser.add_argument("--output", default="results/analysis/generation_targets.json")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    results = data["results"]
    probes = data["probes"]
    baseline = data["baseline_scores"]

    print(f"Loaded {len(results)} compound results")
    print(f"Probes: {probes}")

    # --- Safe paths: no probe degraded ---
    safe = [r for r in results if r.get("min_delta", -1) >= -0.01]
    safe.sort(key=lambda x: x["combined_delta"], reverse=True)

    # --- Best tradeoff: highest combined, accepting degradation ---
    by_combined = sorted(results, key=lambda x: x["combined_delta"], reverse=True)

    # --- Per-domain best ---
    domain_bests = {}
    for probe in probes:
        ranked = sorted(results, key=lambda r: r["deltas"].get(probe, -999), reverse=True)
        for r in ranked[:args.top_per_domain]:
            if r["deltas"].get(probe, 0) > 0:
                domain_bests[f"{probe}:{r['label']}"] = r

    # Collect all selected paths (dedup by label)
    selected = {}
    for r in safe[:args.top_n]:
        selected[r["label"]] = {"result": r, "reason": "safe (no degradation)"}
    for r in by_combined[:args.top_tradeoff]:
        if r["label"] not in selected:
            selected[r["label"]] = {"result": r, "reason": "best tradeoff (highest combined)"}
    for key, r in domain_bests.items():
        if r["label"] not in selected:
            probe_name = key.split(":")[0]
            selected[r["label"]] = {"result": r, "reason": f"best for {probe_name}"}

    # Build targets with per-path relevant generation probes
    targets = []
    for label, info in selected.items():
        r = info["result"]
        gen_probes = get_relevant_generation_probes(
            r["deltas"], probes, args.signal_threshold)

        targets.append({
            "label": label,
            "dup_regions": r["dup_regions"],
            "skip_regions": r["skip_regions"],
            "n_layers": r["n_layers"],
            "combined_delta": r["combined_delta"],
            "min_delta": r.get("min_delta", None),
            "n_improved": r.get("n_improved", 0),
            "n_degraded": r.get("n_degraded", 0),
            "reason": info["reason"],
            "generation_probes": gen_probes,
            "logprob_signal": {
                p: round(r["deltas"].get(p, 0.0), 4)
                for p in probes
                if abs(r["deltas"].get(p, 0.0)) > args.signal_threshold
            },
        })

    targets.sort(key=lambda x: (x.get("min_delta", -1) >= -0.01,
                                 x["combined_delta"]), reverse=True)

    # Save
    output = {
        "source": args.input,
        "signal_threshold": args.signal_threshold,
        "n_targets": len(targets),
        "baseline_scores": baseline,
        "targets": targets,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"GENERATION TARGETS ({len(targets)} paths)")
    print(f"{'='*60}")
    for t in targets:
        safe_flag = "SAFE" if (t.get("min_delta") or 0) >= -0.01 else "TRADE"
        print(f"\n  [{safe_flag}] {t['label']}")
        print(f"    combined={t['combined_delta']:+.4f} min={t.get('min_delta', '?')} "
              f"({t['n_improved']}↑ {t['n_degraded']}↓) | {t['n_layers']}L")
        print(f"    reason: {t['reason']}")
        print(f"    generation probes: {t['generation_probes']}")
        if t["logprob_signal"]:
            top_signals = sorted(t["logprob_signal"].items(),
                                 key=lambda x: abs(x[1]), reverse=True)[:5]
            for p, d in top_signals:
                print(f"      {p}: {d:+.4f}")

    # Estimate generation time
    total_gen_calls = sum(len(t["generation_probes"]) for t in targets)
    est_minutes = total_gen_calls * 8 / 60  # ~8 items per probe, ~1s each
    print(f"\n  Estimated generation time: ~{est_minutes:.0f} minutes")
    print(f"  ({total_gen_calls} probe runs across {len(targets)} paths)")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
