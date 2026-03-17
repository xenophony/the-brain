#!/usr/bin/env python3
"""
Analyze psycholinguistic signal for correlations with performance.

Reads sweep results with psych data and answers:
1. Does hedging increase when the model gets answers wrong?
2. Does confidence predict correct answers?
3. Which psych categories correlate with which probe scores?
4. Do circuit manipulations change psych profiles in meaningful ways?
5. Can psych signal predict answer correctness?

Usage:
  python scripts/analyze_psych_signal.py results/analysis/sweep_results_merged.json
  python scripts/analyze_psych_signal.py results/analysis/generation_verify_*.json --mode verify
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PSYCH_CATEGORIES = [
    "hedging", "confidence", "epistemic_uncertain", "epistemic_certain",
    "causation", "approximators", "negation", "absolutes",
    "first_person", "distancing", "urgency", "stress",
    "multilingual_yes", "multilingual_no",
]


def analyze_sweep_psych(results: list[dict]) -> dict:
    """Analyze psych signal from sweep data."""

    # Get probes and psych categories present
    probes = set()
    psych_keys = set()
    for r in results:
        for k in r.get("probe_scores", {}):
            if "_psych_" in k:
                psych_keys.add(k)
            elif (not k.startswith("_") and "_easy" not in k
                  and "_hard" not in k and "_pcorrect" not in k):
                probes.add(k)

    probes = sorted(probes)
    baseline = next((r for r in results if r["i"] == 0 and r["j"] == 0), None)

    findings = {}

    # 1. Correlation: psych signal vs probe score across configs
    print("=" * 80)
    print("1. PSYCH-PERFORMANCE CORRELATION")
    print("   Does psych signal predict probe performance across configs?")
    print("=" * 80)

    for probe in probes:
        correlations = []
        for cat in PSYCH_CATEGORIES:
            psych_key = f"{probe}_psych_{cat}"

            # Collect (psych_value, score) pairs across configs
            pairs = []
            for r in results:
                scores = r.get("probe_scores", {})
                if psych_key in scores and probe in scores:
                    psych_val = scores[psych_key]
                    score_val = scores[probe]
                    if isinstance(psych_val, (int, float)) and isinstance(score_val, (int, float)):
                        pairs.append((psych_val, score_val))

            if len(pairs) < 10:
                continue

            # Simple correlation (Pearson)
            n = len(pairs)
            sx = sum(p[0] for p in pairs)
            sy = sum(p[1] for p in pairs)
            sxx = sum(p[0] ** 2 for p in pairs)
            syy = sum(p[1] ** 2 for p in pairs)
            sxy = sum(p[0] * p[1] for p in pairs)

            denom = ((n * sxx - sx ** 2) * (n * syy - sy ** 2)) ** 0.5
            if denom == 0:
                continue
            r_val = (n * sxy - sx * sy) / denom
            correlations.append((cat, round(r_val, 4), len(pairs)))

        if correlations:
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            strong = [c for c in correlations if abs(c[1]) > 0.15]
            if strong:
                print(f"\n  {probe}:")
                for cat, r_val, n in strong:
                    direction = "↑score" if r_val > 0 else "↓score"
                    print(f"    psych_{cat}: r={r_val:+.3f} ({direction}, n={n})")
                findings[probe] = strong

    # 2. Psych profile of correct vs incorrect answers
    print(f"\n{'=' * 80}")
    print("2. CORRECT vs INCORRECT PSYCH PROFILES")
    print("   When the model gets it right vs wrong, how does psych differ?")
    print("=" * 80)

    for probe in probes:
        correct_psych = defaultdict(list)
        incorrect_psych = defaultdict(list)

        for r in results:
            scores = r.get("probe_scores", {})
            if probe not in scores:
                continue
            score = scores[probe]
            if not isinstance(score, (int, float)):
                continue

            base = baseline["probe_scores"].get(probe, 0.5) if baseline else 0.5
            is_good = score > base  # above baseline = "correct direction"

            for cat in PSYCH_CATEGORIES:
                psych_key = f"{probe}_psych_{cat}"
                val = scores.get(psych_key)
                if val is not None and isinstance(val, (int, float)):
                    if is_good:
                        correct_psych[cat].append(val)
                    else:
                        incorrect_psych[cat].append(val)

        if not correct_psych or not incorrect_psych:
            continue

        diffs = []
        for cat in PSYCH_CATEGORIES:
            c_vals = correct_psych.get(cat, [])
            i_vals = incorrect_psych.get(cat, [])
            if c_vals and i_vals:
                c_mean = sum(c_vals) / len(c_vals)
                i_mean = sum(i_vals) / len(i_vals)
                diff = c_mean - i_mean
                if abs(diff) > 1e-5:
                    diffs.append((cat, c_mean, i_mean, diff))

        if diffs:
            diffs.sort(key=lambda x: abs(x[3]), reverse=True)
            significant = [d for d in diffs if abs(d[3]) > 1e-4]
            if significant:
                print(f"\n  {probe} ({len(correct_psych[PSYCH_CATEGORIES[0]])} good / "
                      f"{len(incorrect_psych[PSYCH_CATEGORIES[0]])} bad configs):")
                for cat, c_mean, i_mean, diff in significant[:5]:
                    direction = "HIGHER when correct" if diff > 0 else "LOWER when correct"
                    print(f"    psych_{cat}: {direction} "
                          f"(good={c_mean:.6f} bad={i_mean:.6f} diff={diff:+.6f})")

    # 3. Universal psych patterns (across all probes)
    print(f"\n{'=' * 80}")
    print("3. UNIVERSAL PSYCH PATTERNS")
    print("   Psych categories that consistently predict performance across ALL probes")
    print("=" * 80)

    universal = defaultdict(list)  # cat -> list of (probe, correlation)
    for probe, correlations in findings.items():
        for cat, r_val, n in correlations:
            universal[cat].append((probe, r_val))

    for cat in sorted(universal.keys(), key=lambda c: len(universal[c]), reverse=True):
        probes_affected = universal[cat]
        if len(probes_affected) >= 3:
            directions = [r for _, r in probes_affected]
            avg_r = sum(directions) / len(directions)
            consistent = all(r > 0 for r in directions) or all(r < 0 for r in directions)
            marker = "CONSISTENT" if consistent else "mixed"
            print(f"\n  psych_{cat}: affects {len(probes_affected)} probes "
                  f"(avg r={avg_r:+.3f}, {marker})")
            for probe, r_val in sorted(probes_affected, key=lambda x: abs(x[1]), reverse=True):
                print(f"    {probe}: r={r_val:+.3f}")

    # 4. Uncertainty detection potential
    print(f"\n{'=' * 80}")
    print("4. UNCERTAINTY DETECTION POTENTIAL")
    print("   Can psych signal flag when the model is likely wrong?")
    print("=" * 80)

    uncertainty_cats = ["hedging", "epistemic_uncertain", "confidence",
                        "epistemic_certain", "approximators"]

    for probe in probes:
        signals = []
        for cat in uncertainty_cats:
            psych_key = f"{probe}_psych_{cat}"

            # Split configs into high-score and low-score groups
            high_psych = []
            low_psych = []
            for r in results:
                scores = r.get("probe_scores", {})
                if psych_key not in scores or probe not in scores:
                    continue
                psych_val = scores[psych_key]
                score_val = scores[probe]
                if not isinstance(psych_val, (int, float)):
                    continue
                if not isinstance(score_val, (int, float)):
                    continue

                base = baseline["probe_scores"].get(probe, 0.5) if baseline else 0.5
                if score_val > base:
                    high_psych.append(psych_val)
                else:
                    low_psych.append(psych_val)

            if high_psych and low_psych:
                h_mean = sum(high_psych) / len(high_psych)
                l_mean = sum(low_psych) / len(low_psych)
                if h_mean + l_mean > 0:
                    ratio = h_mean / (l_mean + 1e-10)
                    signals.append((cat, h_mean, l_mean, ratio))

        if signals:
            useful = [s for s in signals if abs(s[3] - 1.0) > 0.1]
            if useful:
                useful.sort(key=lambda x: abs(x[3] - 1.0), reverse=True)
                print(f"\n  {probe}:")
                for cat, h_mean, l_mean, ratio in useful[:3]:
                    if ratio > 1:
                        print(f"    psych_{cat}: {ratio:.2f}x HIGHER on correct configs "
                              f"(correct={h_mean:.6f} incorrect={l_mean:.6f})")
                    else:
                        print(f"    psych_{cat}: {1/ratio:.2f}x HIGHER on INCORRECT configs "
                              f"(correct={h_mean:.6f} incorrect={l_mean:.6f})")

    return findings


def analyze_verify_psych(data: dict) -> dict:
    """Analyze psych signal from generation verification data."""
    baseline_psych = data.get("baseline_psych", {})
    results = data.get("results", [])

    print("=" * 80)
    print("GENERATION VERIFICATION — Psych Signal Analysis")
    print("=" * 80)

    for r in results:
        psych_deltas = r.get("psych_deltas", {})
        if not psych_deltas:
            continue

        label = r["label"]
        gen_deltas = r.get("deltas", {})
        gen_improved = [k for k, v in gen_deltas.items() if v > 0.05]
        gen_degraded = [k for k, v in gen_deltas.items() if v < -0.05]

        # Find biggest psych shifts
        psych_shifts = {}
        for cat in PSYCH_CATEGORIES:
            cat_deltas = [v for k, v in psych_deltas.items()
                          if f"_psych_{cat}" in k and isinstance(v, (int, float))]
            if cat_deltas:
                mean_shift = sum(cat_deltas) / len(cat_deltas)
                if abs(mean_shift) > 1e-5:
                    psych_shifts[cat] = mean_shift

        if psych_shifts:
            top_shifts = sorted(psych_shifts.items(),
                                key=lambda x: abs(x[1]), reverse=True)[:5]
            print(f"\n  {label}:")
            if gen_improved:
                print(f"    Gen improved: {', '.join(gen_improved)}")
            if gen_degraded:
                print(f"    Gen degraded: {', '.join(gen_degraded)}")
            print(f"    Psych shifts:")
            for cat, shift in top_shifts:
                direction = "↑" if shift > 0 else "↓"
                print(f"      {cat}: {direction} {abs(shift):.6f}")

    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze psycholinguistic signal for performance correlations")
    parser.add_argument("input", help="sweep_results_merged.json or generation_verify_*.json")
    parser.add_argument("--mode", choices=["sweep", "verify"], default="sweep",
                        help="Analysis mode: sweep (default) or verify")
    parser.add_argument("--output", default="results/analysis/PSYCH_ANALYSIS.md")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if args.mode == "verify" or "baseline_psych" in data:
        findings = analyze_verify_psych(data)
    else:
        if isinstance(data, list):
            findings = analyze_sweep_psych(data)
        else:
            print("ERROR: Expected list of sweep results")
            sys.exit(1)


if __name__ == "__main__":
    main()
