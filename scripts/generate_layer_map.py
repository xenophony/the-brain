#!/usr/bin/env python3
"""
Generate a visual layer map from sweep data.

Reads merged sweep results and produces a human-readable layer-by-layer
summary showing what each layer does (benefits when duplicated),
what it harms, and whether it's safely skippable.

Outputs: results/analysis/LAYER_MAP.md

Usage:
  python scripts/generate_layer_map.py results/analysis/sweep_results_merged.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_probes(results):
    """Get main probe names."""
    baseline = next((r for r in results if r["i"] == 0 and r["j"] == 0), {})
    scores = baseline.get("probe_scores", {})
    return [k for k in scores
            if not k.startswith("_") and "_easy" not in k and "_hard" not in k
            and "_pcorrect" not in k and "_psych_" not in k]


def analyze_single_layer_effects(results, probes):
    """For each layer, find what happens when JUST that layer is duplicated or skipped."""
    layer_effects = defaultdict(lambda: {
        "dup_helps": [],    # probes that improve when this layer is duplicated
        "dup_hurts": [],    # probes that degrade when this layer is duplicated
        "skip_helps": [],   # probes that improve when this layer is removed
        "skip_hurts": [],   # probes that degrade when this layer is removed
        "dup_psych": {},    # psych signal changes on duplication
        "skip_psych": {},   # psych signal changes on skip
        "dup_delta": 0.0,   # combined delta when duplicated
        "skip_delta": 0.0,  # combined delta when skipped
    })

    threshold = 0.05  # minimum delta to count

    for r in results:
        i, j, mode = r["i"], r["j"], r.get("mode", "duplicate")
        if i == 0 and j == 0:
            continue
        # Only look at single-layer manipulations (j - i == 1)
        if j - i != 1:
            continue

        layer = i  # the layer being duplicated/skipped
        deltas = r["probe_deltas"]

        for probe in probes:
            d = deltas.get(probe, 0.0)
            if not isinstance(d, (int, float)):
                continue

            if mode == "duplicate":
                layer_effects[layer]["dup_delta"] += d
                if d > threshold:
                    layer_effects[layer]["dup_helps"].append((probe, round(d, 4)))
                elif d < -threshold:
                    layer_effects[layer]["dup_hurts"].append((probe, round(d, 4)))
            elif mode == "skip":
                layer_effects[layer]["skip_delta"] += d
                if d > threshold:
                    layer_effects[layer]["skip_helps"].append((probe, round(d, 4)))
                elif d < -threshold:
                    layer_effects[layer]["skip_hurts"].append((probe, round(d, 4)))

        # Psych signals
        for key, val in deltas.items():
            if "_psych_" not in key or not isinstance(val, (int, float)):
                continue
            cat = key.split("_psych_")[-1]
            if abs(val) > 1e-5:
                if mode == "duplicate":
                    if cat not in layer_effects[layer]["dup_psych"]:
                        layer_effects[layer]["dup_psych"][cat] = []
                    layer_effects[layer]["dup_psych"][cat].append(val)
                elif mode == "skip":
                    if cat not in layer_effects[layer]["skip_psych"]:
                        layer_effects[layer]["skip_psych"][cat] = []
                    layer_effects[layer]["skip_psych"][cat].append(val)

    return layer_effects


def classify_layer(effects):
    """Classify a layer based on its effects."""
    dup_d = effects["dup_delta"]
    skip_d = effects["skip_delta"]
    n_dup_helps = len(effects["dup_helps"])
    n_dup_hurts = len(effects["dup_hurts"])
    n_skip_helps = len(effects["skip_helps"])
    n_skip_hurts = len(effects["skip_hurts"])

    tags = []

    # Duplication effects
    if dup_d > 0.5 and n_dup_helps >= 3:
        tags.append("AMPLIFY")
    elif dup_d > 0.2:
        tags.append("amplify-mild")
    elif dup_d < -0.5:
        tags.append("FRAGILE")
    elif dup_d < -0.2:
        tags.append("fragile-mild")

    # Skip effects
    if skip_d > 0.5 and n_skip_helps >= 3:
        tags.append("DISPENSABLE")
    elif skip_d > 0.2:
        tags.append("dispensable-mild")
    elif skip_d < -0.5 and n_skip_hurts >= 3:
        tags.append("CRITICAL")
    elif skip_d < -0.2:
        tags.append("critical-mild")

    # Special patterns
    if n_dup_helps > 0 and n_dup_hurts > 0:
        tags.append("TRADEOFF")
    if skip_d > 0.3 and dup_d > 0.3:
        tags.append("PARADOX")  # both removing and adding helps?

    return tags if tags else ["neutral"]


def generate_map(results, output_path):
    """Generate the layer map."""
    probes = get_probes(results)
    effects = analyze_single_layer_effects(results, probes)

    n_layers = max(effects.keys()) + 1 if effects else 48

    lines = []
    lines.append("# Layer Map — Qwen3-30B-A3B Circuit Functions\n")
    lines.append("Generated from sweep data. Shows single-layer effects only.\n")
    lines.append("Tags: AMPLIFY (duplicate helps), DISPENSABLE (skip helps),")
    lines.append("CRITICAL (skip hurts), FRAGILE (duplicate hurts), TRADEOFF (mixed).\n")

    # Domain-function summary
    lines.append("## Layer Functions\n")
    lines.append("What each layer does (benefits when duplicated) and what it harms.\n")

    # Domain groupings for readable labels
    domain_labels = {
        "math": "MATH",
        "causal_logprob": "CAUSAL",
        "logic_logprob": "LOGIC",
        "sentiment_logprob": "EMOTION",
        "error_logprob": "ERROR-CHECK",
        "hallucination_logprob": "HALLUC-RESIST",
        "sycophancy_logprob": "INTEGRITY",
        "consistency_logprob": "CONSISTENCY",
        "routing_logprob": "ROUTING",
        "judgement_logprob": "SELF-EVAL",
        "language_logprob": "LANGUAGE",
        "pong_simple_logprob": "SPATIAL",
        "pong_strategic_logprob": "SPATIAL-PLAN",
        "implication_logprob": "IMPLICATION",
        "negation_logprob": "NEGATION",
        "temporal_logprob": "TEMPORAL",
        "psych_conflict_logprob": "CONFLICT-DETECT",
        "psych_difficulty_logprob": "DIFFICULTY-SENSE",
        "psych_unknowable_logprob": "UNKNOWABLE-SENSE",
        "psych_urgency_logprob": "URGENCY-SENSE",
    }

    for layer in range(n_layers):
        e = effects.get(layer, {})
        if not e:
            lines.append(f"**L{layer:02d}** | no data")
            continue

        # Build function description from top probe effects
        dup_helps = sorted(e.get("dup_helps", []), key=lambda x: x[1], reverse=True)
        dup_hurts = sorted(e.get("dup_hurts", []), key=lambda x: x[1])
        skip_helps = sorted(e.get("skip_helps", []), key=lambda x: x[1], reverse=True)
        skip_hurts = sorted(e.get("skip_hurts", []), key=lambda x: x[1])

        # Top 3 things it helps (when duplicated)
        amplifies = [f"{domain_labels.get(p, p)}({d:+.2f})"
                     for p, d in dup_helps[:4]]
        # Top 3 things it hurts (when duplicated)
        harms = [f"{domain_labels.get(p, p)}({d:+.2f})"
                 for p, d in dup_hurts[:3]]
        # What improves when removed
        dispensable_for = [f"{domain_labels.get(p, p)}({d:+.2f})"
                          for p, d in skip_helps[:3]]
        # What breaks when removed
        critical_for = [f"{domain_labels.get(p, p)}({d:+.2f})"
                        for p, d in skip_hurts[:3]]

        line = f"**L{layer:02d}**"
        parts = []
        if amplifies:
            parts.append(f"Amplifies: {', '.join(amplifies)}")
        if harms:
            parts.append(f"Harms: {', '.join(harms)}")
        if dispensable_for:
            parts.append(f"Removable for: {', '.join(dispensable_for)}")
        if critical_for:
            parts.append(f"Critical for: {', '.join(critical_for)}")

        if parts:
            lines.append(f"{line} | {' | '.join(parts)}")
        else:
            lines.append(f"{line} | neutral (no strong effects)")

    lines.append("")

    # Detailed per-layer breakdown
    lines.append("## Detailed Layer Analysis\n")
    for layer in range(n_layers):
        e = effects.get(layer)
        if not e:
            lines.append(f"### Layer {layer} — no data\n")
            continue

        tags = classify_layer(e)
        tag_str = " | ".join(tags)
        lines.append(f"### Layer {layer} — {tag_str}\n")

        # Duplication effects
        if e["dup_helps"] or e["dup_hurts"]:
            lines.append(f"**Duplicate (dup_delta={e['dup_delta']:+.2f}):**")
            if e["dup_helps"]:
                helps = sorted(e["dup_helps"], key=lambda x: x[1], reverse=True)
                lines.append(f"  Helps: {', '.join(f'{p} ({d:+.2f})' for p, d in helps)}")
            if e["dup_hurts"]:
                hurts = sorted(e["dup_hurts"], key=lambda x: x[1])
                lines.append(f"  Hurts: {', '.join(f'{p} ({d:+.2f})' for p, d in hurts)}")

        # Skip effects
        if e["skip_helps"] or e["skip_hurts"]:
            lines.append(f"**Skip (skip_delta={e['skip_delta']:+.2f}):**")
            if e["skip_helps"]:
                helps = sorted(e["skip_helps"], key=lambda x: x[1], reverse=True)
                lines.append(f"  Helps: {', '.join(f'{p} ({d:+.2f})' for p, d in helps)}")
            if e["skip_hurts"]:
                hurts = sorted(e["skip_hurts"], key=lambda x: x[1])
                lines.append(f"  Hurts: {', '.join(f'{p} ({d:+.2f})' for p, d in hurts)}")

        # Psych summary
        dup_psych = e.get("dup_psych", {})
        skip_psych = e.get("skip_psych", {})
        if dup_psych or skip_psych:
            lines.append("**Psych signal:**")
            for cat in sorted(set(list(dup_psych.keys()) + list(skip_psych.keys()))):
                dup_vals = dup_psych.get(cat, [])
                skip_vals = skip_psych.get(cat, [])
                dup_mean = sum(dup_vals) / len(dup_vals) if dup_vals else 0
                skip_mean = sum(skip_vals) / len(skip_vals) if skip_vals else 0
                if abs(dup_mean) > 1e-4 or abs(skip_mean) > 1e-4:
                    lines.append(f"  {cat}: dup={dup_mean:+.6f} skip={skip_mean:+.6f}")

        lines.append("")

    # Region summary
    lines.append("## Region Summary\n")

    # Find contiguous regions with same classification
    current_tags = None
    region_start = 0
    regions = []
    for layer in range(n_layers):
        e = effects.get(layer)
        tags = tuple(classify_layer(e)) if e else ("no-data",)
        if tags != current_tags:
            if current_tags is not None:
                regions.append((region_start, layer - 1, current_tags))
            current_tags = tags
            region_start = layer
    if current_tags:
        regions.append((region_start, n_layers - 1, current_tags))

    for start, end, tags in regions:
        tag_str = ", ".join(tags)
        if start == end:
            lines.append(f"- **Layer {start}**: {tag_str}")
        else:
            lines.append(f"- **Layers {start}-{end}**: {tag_str}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual layer map from sweep data")
    parser.add_argument("input", help="sweep_results_merged.json")
    parser.add_argument("--output", default="results/analysis/LAYER_MAP.md")
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} configs")

    report = generate_map(results, args.output)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    print(f"Layer map saved to {out_path}")


if __name__ == "__main__":
    main()
