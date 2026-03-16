#!/usr/bin/env python3
"""
Analyze logprob sweep results to select configs for generation verification.

Reads sweep_results.json from a logprob sweep and identifies:
  A. Single-domain boosters — one probe improves significantly
  B. Multi-domain synergies — 3+ probes improve together
  C. Skip candidates — layers that hurt performance
  D. Antagonistic pairs — improving one domain degrades another

Outputs:
  - targeted_configs.json — configs to verify with generation probes
  - CIRCUIT_FINDINGS.md — human-readable summary
  - Command to run generation verification sweep

Usage:
  python scripts/analyze_logprob_sweep.py results/tier1_reasoning
  python scripts/analyze_logprob_sweep.py results/tier1_reasoning results/tier2_perception --merge
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_sweep(sweep_dir: Path) -> list[dict]:
    """Load sweep results from a directory."""
    f = sweep_dir / "sweep_results.json"
    if not f.exists():
        print(f"ERROR: {f} not found")
        sys.exit(1)
    with open(f) as fh:
        return json.load(fh)


def extract_baseline(results: list[dict]) -> dict:
    """Find baseline (0,0) scores."""
    for r in results:
        if r["i"] == 0 and r["j"] == 0:
            return r["probe_scores"]
    return {}


def get_probe_names(results: list[dict]) -> list[str]:
    """Get probe names (excluding _easy, _hard, _pcorrect, _failures)."""
    baseline = extract_baseline(results)
    return [k for k in baseline
            if not k.startswith("_")
            and not k.endswith("_easy")
            and not k.endswith("_hard")
            and not k.endswith("_pcorrect")
            and not k.endswith("_pcorrect_easy")
            and not k.endswith("_pcorrect_hard")]


def analyze(results: list[dict],
            boost_threshold: float = 0.15,
            drop_threshold: float = -0.15,
            synergy_min_probes: int = 3,
            synergy_threshold: float = 0.10,
            top_n: int = 10) -> dict:
    """Analyze sweep results and categorize configs."""

    baseline = extract_baseline(results)
    probes = get_probe_names(results)
    pcorrect_keys = [f"{p}_pcorrect" for p in probes
                     if f"{p}_pcorrect" in baseline]

    # Skip baseline in analysis
    configs = [r for r in results if not (r["i"] == 0 and r["j"] == 0)]

    # --- Per-probe rankings ---
    probe_rankings = {}  # probe -> sorted list of (i, j, delta, pcorrect_delta)
    for probe in probes:
        pc_key = f"{probe}_pcorrect"
        ranked = []
        for r in configs:
            scores = r["probe_scores"]
            deltas = r["probe_deltas"]
            delta = deltas.get(probe, 0.0)
            pc_delta = deltas.get(pc_key, 0.0) if pc_key in deltas else None
            ranked.append({
                "i": r["i"], "j": r["j"],
                "delta": delta,
                "pcorrect_delta": pc_delta,
                "score": scores.get(probe, 0.0),
                "pcorrect": scores.get(pc_key),
            })
        ranked.sort(key=lambda x: x["delta"], reverse=True)
        probe_rankings[probe] = ranked

    # --- Category A: Single-domain boosters ---
    boosters = {}  # probe -> top configs
    for probe, ranked in probe_rankings.items():
        top = [r for r in ranked[:top_n] if r["delta"] > boost_threshold]
        if top:
            boosters[probe] = top

    # --- Category B: Multi-domain synergies ---
    config_boost_count = defaultdict(lambda: {"probes": [], "total_delta": 0.0})
    for probe, ranked in probe_rankings.items():
        for r in ranked:
            if r["delta"] > synergy_threshold:
                key = (r["i"], r["j"])
                config_boost_count[key]["probes"].append(probe)
                config_boost_count[key]["total_delta"] += r["delta"]

    synergies = []
    for (i, j), info in config_boost_count.items():
        if len(info["probes"]) >= synergy_min_probes:
            synergies.append({
                "i": i, "j": j,
                "n_probes": len(info["probes"]),
                "probes": info["probes"],
                "total_delta": round(info["total_delta"], 4),
            })
    synergies.sort(key=lambda x: x["total_delta"], reverse=True)

    # --- Category C: Skip candidates ---
    skip_candidates = {}
    for probe, ranked in probe_rankings.items():
        bottom = [r for r in ranked if r["delta"] < drop_threshold]
        bottom.sort(key=lambda x: x["delta"])
        if bottom:
            skip_candidates[probe] = bottom[:top_n]

    # Multi-probe drops
    config_drop_count = defaultdict(lambda: {"probes": [], "total_delta": 0.0})
    for probe, ranked in probe_rankings.items():
        for r in ranked:
            if r["delta"] < drop_threshold:
                key = (r["i"], r["j"])
                config_drop_count[key]["probes"].append(probe)
                config_drop_count[key]["total_delta"] += r["delta"]

    multi_drops = []
    for (i, j), info in config_drop_count.items():
        if len(info["probes"]) >= 2:
            multi_drops.append({
                "i": i, "j": j,
                "n_probes": len(info["probes"]),
                "probes": info["probes"],
                "total_delta": round(info["total_delta"], 4),
            })
    multi_drops.sort(key=lambda x: x["total_delta"])

    # --- Category D: Antagonistic pairs ---
    antagonistic = []
    for probe_a in probes:
        for probe_b in probes:
            if probe_a >= probe_b:
                continue
            for r in configs:
                delta_a = r["probe_deltas"].get(probe_a, 0.0)
                delta_b = r["probe_deltas"].get(probe_b, 0.0)
                if delta_a > boost_threshold and delta_b < drop_threshold:
                    antagonistic.append({
                        "i": r["i"], "j": r["j"],
                        "improves": probe_a,
                        "improves_delta": round(delta_a, 4),
                        "degrades": probe_b,
                        "degrades_delta": round(delta_b, 4),
                    })
    antagonistic.sort(key=lambda x: x["improves_delta"] - x["degrades_delta"],
                      reverse=True)

    # --- Collect all unique target configs ---
    target_configs = set()
    for probe, tops in boosters.items():
        for r in tops[:5]:  # top 5 per probe
            target_configs.add((r["i"], r["j"]))
    for s in synergies[:20]:
        target_configs.add((s["i"], s["j"]))
    for probe, bottoms in skip_candidates.items():
        for r in bottoms[:3]:  # worst 3 per probe
            target_configs.add((r["i"], r["j"]))
    for a in antagonistic[:10]:
        target_configs.add((a["i"], a["j"]))

    # --- pcorrect analysis: strongest signals ---
    pcorrect_highlights = {}
    for probe in probes:
        pc_key = f"{probe}_pcorrect"
        if pc_key not in baseline:
            continue
        ranked_pc = []
        for r in configs:
            pc_delta = r["probe_deltas"].get(pc_key, 0.0)
            if abs(pc_delta) > 0.05:  # meaningful shift
                ranked_pc.append({
                    "i": r["i"], "j": r["j"],
                    "pcorrect_delta": round(pc_delta, 4),
                })
        ranked_pc.sort(key=lambda x: abs(x["pcorrect_delta"]), reverse=True)
        if ranked_pc:
            pcorrect_highlights[probe] = ranked_pc[:10]

    return {
        "baseline": {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in baseline.items()
                     if not k.startswith("_")},
        "n_configs": len(configs),
        "probes": probes,
        "boosters": {k: v[:5] for k, v in boosters.items()},
        "synergies": synergies[:20],
        "skip_candidates": {k: v[:5] for k, v in skip_candidates.items()},
        "multi_drops": multi_drops[:20],
        "antagonistic": antagonistic[:20],
        "pcorrect_highlights": pcorrect_highlights,
        "target_configs": sorted(target_configs),
        "n_targets": len(target_configs),
    }


def generate_report(analysis: dict, output_dir: Path) -> str:
    """Generate human-readable CIRCUIT_FINDINGS.md."""
    lines = []
    lines.append("# Circuit Findings — Logprob Sweep Analysis\n")

    # Baseline
    lines.append("## Baseline Scores\n")
    for k, v in sorted(analysis["baseline"].items()):
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}")
    lines.append(f"\nTotal configs analyzed: {analysis['n_configs']}\n")

    # Boosters
    lines.append("## Category A: Single-Domain Boosters\n")
    for probe, tops in analysis["boosters"].items():
        lines.append(f"\n### {probe}")
        for r in tops:
            pc = f", pcorrect_delta={r['pcorrect_delta']}" if r.get('pcorrect_delta') is not None else ""
            lines.append(f"  ({r['i']},{r['j']}) delta={r['delta']:+.4f}{pc}")

    # Synergies
    lines.append("\n## Category B: Multi-Domain Synergies\n")
    if analysis["synergies"]:
        for s in analysis["synergies"][:10]:
            probes_str = ", ".join(s["probes"])
            lines.append(f"  ({s['i']},{s['j']}) {s['n_probes']} probes, "
                         f"total_delta={s['total_delta']:+.4f} — {probes_str}")
    else:
        lines.append("  None found at current thresholds.\n")

    # Skip candidates
    lines.append("\n## Category C: Skip Candidates (harmful layers)\n")
    if analysis["multi_drops"]:
        for d in analysis["multi_drops"][:10]:
            probes_str = ", ".join(d["probes"])
            lines.append(f"  ({d['i']},{d['j']}) {d['n_probes']} probes degraded, "
                         f"total_delta={d['total_delta']:+.4f} — {probes_str}")
    else:
        lines.append("  None found at current thresholds.\n")

    # Antagonistic
    lines.append("\n## Category D: Antagonistic Pairs\n")
    if analysis["antagonistic"]:
        for a in analysis["antagonistic"][:10]:
            lines.append(f"  ({a['i']},{a['j']}) {a['improves']} {a['improves_delta']:+.4f} "
                         f"vs {a['degrades']} {a['degrades_delta']:+.4f}")
    else:
        lines.append("  None found at current thresholds.\n")

    # pcorrect highlights
    lines.append("\n## Probability Tracking Highlights (p_correct)\n")
    for probe, highlights in analysis["pcorrect_highlights"].items():
        lines.append(f"\n### {probe}")
        for h in highlights[:5]:
            lines.append(f"  ({h['i']},{h['j']}) pcorrect_delta={h['pcorrect_delta']:+.4f}")

    # Summary
    lines.append(f"\n## Verification Targets\n")
    lines.append(f"**{analysis['n_targets']} unique configs** selected for generation verification.\n")

    # Emerging circuit map
    lines.append("## Emerging Circuit Map\n")
    lines.append("Compile layer regions by domain from the booster data above.")
    lines.append("Look for convergent evidence: multiple probes pointing at the same layers.\n")

    return "\n".join(lines)


def generate_verification_command(analysis: dict, model_path: str) -> str:
    """Generate the run_sweep.py command for targeted generation verification."""
    configs = analysis["target_configs"]
    if not configs:
        return "# No target configs found"

    # For now, the sweep runner doesn't support arbitrary config lists.
    # Find the bounding box and suggest a constrained sweep.
    min_i = min(c[0] for c in configs)
    max_j = max(c[1] for c in configs)

    return (
        f"# Generation verification on {len(configs)} target configs\n"
        f"# Bounding box: i=[{min_i},{max(c[0] for c in configs)}], "
        f"j=[{min(c[1] for c in configs)},{max_j}]\n"
        f"#\n"
        f"# Option 1: Run full generation probes on the interesting region\n"
        f"python scripts/run_sweep.py \\\n"
        f"  --model {model_path} \\\n"
        f"  --probes eq language spatial_pong_simple spatial_pong_strategic \\\n"
        f"  --mode duplicate \\\n"
        f"  --max-layers {max_j + 1} \\\n"
        f"  --output results/generation_verification\n"
        f"#\n"
        f"# Option 2: For targeted configs, use Python directly:\n"
        f"# from scripts.analyze_logprob_sweep import load_targets\n"
        f"# configs = load_targets('results/targeted_configs.json')\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze logprob sweep results for generation verification targets")
    parser.add_argument("sweep_dirs", nargs="+",
                        help="One or more sweep result directories to analyze")
    parser.add_argument("--merge", action="store_true",
                        help="Merge results from multiple directories")
    parser.add_argument("--boost-threshold", type=float, default=0.15,
                        help="Minimum delta for booster classification (default: 0.15)")
    parser.add_argument("--drop-threshold", type=float, default=-0.15,
                        help="Maximum delta for skip classification (default: -0.15)")
    parser.add_argument("--model", default="models/Qwen3-30B-A3B-exl2",
                        help="Model path for verification command")
    args = parser.parse_args()

    # Load and optionally merge results
    all_results = []
    for d in args.sweep_dirs:
        sweep_dir = Path(d)
        results = load_sweep(sweep_dir)
        print(f"Loaded {len(results)} configs from {sweep_dir}")
        all_results.extend(results)

    if args.merge and len(args.sweep_dirs) > 1:
        # Merge by (i, j, mode): combine probe_scores from different runs
        merged = {}
        for r in all_results:
            key = (r["i"], r["j"], r.get("mode", "duplicate"))
            if key not in merged:
                merged[key] = dict(r)
            else:
                # Merge probe_scores and probe_deltas
                merged[key]["probe_scores"].update(r["probe_scores"])
                merged[key]["probe_deltas"].update(r["probe_deltas"])
        all_results = list(merged.values())
        print(f"Merged to {len(all_results)} unique configs")

    # Analyze
    analysis = analyze(
        all_results,
        boost_threshold=args.boost_threshold,
        drop_threshold=args.drop_threshold,
    )

    # Output directory = first sweep dir
    output_dir = Path(args.sweep_dirs[0])

    # Save targeted configs
    targets_file = output_dir / "targeted_configs.json"
    with open(targets_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {targets_file}")

    # Generate report
    report = generate_report(analysis, output_dir)
    report_file = output_dir / "CIRCUIT_FINDINGS.md"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Report saved to {report_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Configs analyzed: {analysis['n_configs']}")
    print(f"Probes: {', '.join(analysis['probes'])}")
    print(f"\nBoosters (Category A):")
    for probe, tops in analysis["boosters"].items():
        best = tops[0]
        print(f"  {probe}: ({best['i']},{best['j']}) delta={best['delta']:+.4f}")
    print(f"\nSynergies (Category B): {len(analysis['synergies'])} found")
    if analysis["synergies"]:
        s = analysis["synergies"][0]
        print(f"  Best: ({s['i']},{s['j']}) {s['n_probes']} probes, "
              f"total_delta={s['total_delta']:+.4f}")
    print(f"\nSkip candidates (Category C): {len(analysis['multi_drops'])} multi-probe drops")
    print(f"Antagonistic pairs (Category D): {len(analysis['antagonistic'])} found")
    print(f"\nVerification targets: {analysis['n_targets']} unique configs")

    # Verification command
    print(f"\n{'='*60}")
    print("NEXT STEP: Generation Verification")
    print(f"{'='*60}")
    print(generate_verification_command(analysis, args.model))


def load_targets(path: str) -> list[tuple[int, int]]:
    """Utility: load target configs from analysis output."""
    with open(path) as f:
        data = json.load(f)
    return [tuple(c) for c in data["target_configs"]]


if __name__ == "__main__":
    main()
