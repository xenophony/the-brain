#!/usr/bin/env python3
"""
Test steering vectors on generative probes — actual text output quality.

Runs generative probes with and without steering vectors applied,
comparing real output quality (not just logprobs).

Usage:
  python scripts/test_steering_generative.py --model models/Qwen3-30B-A3B-exl2 \
    --vectors results/analysis/steering_vectors_hallucination_logprob_20260318_143130.pt \
    --alpha 1.5 --probes tool_use math factual hallucination code

  python scripts/test_steering_generative.py --model models/Qwen3-30B-A3B-exl2 \
    --vectors results/analysis/steering_vectors_hallucination_logprob_20260318_143130.pt \
    --alpha 1.5 --probes all
"""

import argparse
import json
import sys
import time
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Probes that use generation (not logprobs)
GENERATIVE_PROBES = [
    "math", "eq", "code", "factual", "spatial",
    "language", "tool_use", "holistic", "planning",
    "instruction", "hallucination", "sycophancy",
    "consistency", "metacognition", "counterfactual",
    "abstraction", "estimation", "reasoning",
    "implication", "negation",
]

# Also include logprob probes for comparison
LOGPROB_PROBES = [
    "hallucination_logprob", "causal_logprob", "logic_logprob",
    "sentiment_logprob", "error_logprob", "judgement_logprob",
    "consistency_logprob", "negation_logprob",
]


def run_with_steering(model, probe, steering_vectors, alpha):
    """Run a probe with steering vectors applied via monkey-patch."""
    original_run_module = model._run_module

    def steered_run_module(module, x, cache, attn_params, past_len):
        result = original_run_module(module, x, cache, attn_params, past_len)
        for layer_idx, vector in steering_vectors.items():
            layer = model._layer_modules[layer_idx]
            target_module = layer[1] if model._moe_mode else layer
            if module is target_module:
                device = result.device
                sv = vector.to(device) * alpha
                if result.dim() == 3:
                    result[:, -1, :] += sv
                elif result.dim() == 2:
                    result[-1, :] += sv
                break
        return result

    model._run_module = steered_run_module
    try:
        probe.log_responses = True
        result = probe.run(model)
    finally:
        model._run_module = original_run_module

    return result


def extract_score(result):
    """Extract score from probe result."""
    if isinstance(result, dict):
        return result.get("score", 0.0)
    return float(result)


def main():
    parser = argparse.ArgumentParser(
        description="Test steering vectors on generative probes")
    default_model = str(Path(__file__).parent.parent / "models" / "Qwen3-30B-A3B-exl2")
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--vectors", required=True,
                        help="Path to saved steering vectors .pt file")
    parser.add_argument("--alpha", type=float, default=1.5,
                        help="Steering strength (default: 1.5)")
    parser.add_argument("--probes", nargs="+", default=["all"],
                        help="Probes to test (default: all)")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Max items per probe (default: all)")
    parser.add_argument("--output-dir", type=str, default="results/steering_generative/")
    args = parser.parse_args()

    # Load model
    from sweep.exllama_adapter import ExLlamaV2LayerAdapter
    model = ExLlamaV2LayerAdapter(args.model)

    # Load steering vectors
    steering_vectors = torch.load(args.vectors, map_location="cpu", weights_only=True)
    layers = sorted(steering_vectors.keys())
    print(f"Loaded steering vectors for layers: {layers}")
    print(f"Alpha: {args.alpha}")

    # Resolve probes
    from probes.registry import get_probe
    import probes  # trigger auto-discovery

    if "all" in args.probes:
        probe_names = GENERATIVE_PROBES
    elif "logprob" in args.probes:
        probe_names = LOGPROB_PROBES
    elif "both" in args.probes:
        probe_names = GENERATIVE_PROBES + LOGPROB_PROBES
    else:
        probe_names = args.probes

    # Filter to probes that exist
    available = []
    for name in probe_names:
        try:
            get_probe(name)
            available.append(name)
        except KeyError:
            print(f"  Warning: probe '{name}' not found, skipping")
    probe_names = available

    print(f"\nTesting {len(probe_names)} probes: {probe_names}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run each probe: baseline then steered
    results = {}
    t0 = time.time()

    for probe_name in probe_names:
        probe = get_probe(probe_name)
        if args.max_items is not None:
            probe.max_items = args.max_items
        probe.log_responses = True

        print(f"\n{'='*60}")
        print(f"Probe: {probe_name}")
        print(f"{'='*60}")

        # Baseline
        model.set_layer_path(list(range(model.num_layers)))
        tp = time.time()
        baseline_result = probe.run(model)
        baseline_time = time.time() - tp
        baseline_score = extract_score(baseline_result)

        # Steered
        tp = time.time()
        steered_result = run_with_steering(model, probe, steering_vectors, args.alpha)
        steered_time = time.time() - tp
        steered_score = extract_score(steered_result)

        delta = steered_score - baseline_score
        marker = "+++" if delta > 0.05 else "---" if delta < -0.05 else ""

        print(f"  Baseline: {baseline_score:.4f} ({baseline_time:.1f}s)")
        print(f"  Steered:  {steered_score:.4f} ({steered_time:.1f}s)")
        print(f"  Delta:    {delta:+.4f} {marker}")

        # Show response differences for generative probes
        if isinstance(baseline_result, dict) and isinstance(steered_result, dict):
            b_items = baseline_result.get("item_results", [])
            s_items = steered_result.get("item_results", [])
            if b_items and s_items:
                diffs = 0
                for bi, si in zip(b_items, s_items):
                    b_resp = bi.get("response", bi.get("argmax", ""))
                    s_resp = si.get("response", si.get("argmax", ""))
                    b_sc = bi.get("score", bi.get("argmax_correct", 0))
                    s_sc = si.get("score", si.get("argmax_correct", 0))
                    if b_resp != s_resp:
                        diffs += 1
                print(f"  Responses changed: {diffs}/{len(b_items)}")

        results[probe_name] = {
            "baseline_score": baseline_score,
            "steered_score": steered_score,
            "delta": round(delta, 4),
            "baseline_result": baseline_result if isinstance(baseline_result, dict) else {"score": baseline_score},
            "steered_result": steered_result if isinstance(steered_result, dict) else {"score": steered_score},
        }

    total_time = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY (alpha={args.alpha}, layers={layers})")
    print(f"{'='*60}")
    print(f"{'Probe':<25s} {'Baseline':>9s} {'Steered':>9s} {'Delta':>9s}")
    print("-" * 55)
    for name, r in sorted(results.items()):
        delta = r["delta"]
        marker = " +++" if delta > 0.05 else " ---" if delta < -0.05 else ""
        print(f"{name:<25s} {r['baseline_score']:>9.4f} {r['steered_score']:>9.4f} {delta:>+9.4f}{marker}")

    total_delta = sum(r["delta"] for r in results.values())
    print(f"\n  Total delta: {total_delta:+.4f}")
    print(f"  Time: {total_time:.1f}s")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"steering_generative_{timestamp}.json"
    save_data = {
        "timestamp": timestamp,
        "vectors_file": str(args.vectors),
        "alpha": args.alpha,
        "layers": [int(l) for l in layers],
        "results": results,
        "total_delta": round(total_delta, 4),
    }
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {out_file}")


if __name__ == "__main__":
    main()
