#!/usr/bin/env python3
"""
Build steering vectors from activation captures.

Step 1: Run questions through model, capture hidden states at target layers.
Step 2: Compute steering vector = mean(correct) - mean(incorrect).
Step 3: Test steering by injecting vector at target layer.

Usage:
  python scripts/build_steering_vectors.py --model models/Qwen3-30B-A3B-exl2
"""

import argparse
import json
import sys
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def capture_activations(model, probe_name, target_layers, max_items=None):
    """Run a probe's questions and capture hidden states at target layers.

    Returns list of {
        "item": item_dict,
        "score": float,
        "correct": bool,
        "activations": {layer_idx: tensor}
    }
    """
    from probes.registry import get_probe
    import importlib

    probe = get_probe(probe_name)
    mod = importlib.import_module(f"probes.{probe_name}.probe")

    # Load items — check probe instance for ITEMS (logprob) or module for EASY/HARD
    if hasattr(probe_instance, 'ITEMS') and probe_instance.ITEMS:
        all_items = list(probe_instance.ITEMS)
    else:
        easy_items = list(getattr(mod, "EASY_ITEMS", []))
        hard_items = list(getattr(mod, "HARD_ITEMS", []))
        all_items = easy_items + hard_items
    if max_items:
        all_items = all_items[:max_items]

    # Get the prompt function and score function based on probe type
    # Generation probes: use generate_short + scoring function
    # Logprob probes: use get_logprobs + argmax check
    from probes.registry import BaseLogprobProbe
    probe_instance = get_probe(probe_name)
    is_logprob = isinstance(probe_instance, BaseLogprobProbe)

    if is_logprob:
        choices = probe_instance.CHOICES
        make_prompt = lambda item: item["prompt"] + " Answer with one word."
        def score_fn(resp_or_logprobs, item):
            # For logprob probes, resp is actually a logprobs dict
            import math as _math
            expected = item["answer"].lower()
            probs = {}
            for c in choices:
                lp = resp_or_logprobs.get(c, float('-inf'))
                probs[c] = _math.exp(lp) if lp > -100 else 0.0
            best = max(probs, key=probs.get) if probs else ""
            return 1.0 if best == expected else 0.0
    elif probe_name == "math":
        from probes.math.probe import score_math
        make_prompt = lambda item: item["prompt"]
        score_fn = lambda resp, item: score_math(resp, item["answer"])
    elif probe_name == "eq":
        from probes.eq.probe import _eq_digit_score
        make_prompt = lambda item: item["prompt"]
        score_fn = lambda resp, item: _eq_digit_score(resp, item["expected"])
    elif probe_name == "language":
        from probes.language.probe import score_language, PROMPT_TEMPLATE
        make_prompt = lambda item: PROMPT_TEMPLATE.format(sentence=item["sentence"])
        score_fn = lambda resp, item: score_language(resp, item["label"])
    else:
        raise ValueError(f"Unsupported probe: {probe_name}")

    # Normal layer path
    model.set_layer_path(list(range(model.num_layers)))

    results = []
    for item in all_items:
        prompt = make_prompt(item)

        # Capture activations at target layers
        layer_activations = {}

        def hook_fn(exec_pos, layer_idx, hidden_state):
            if layer_idx in target_layers:
                # Store last-token hidden state
                layer_activations[layer_idx] = hidden_state[:, -1, :].detach().cpu()

        model.forward_with_hooks(prompt, hook_fn)

        # Score: generate for gen probes, logprobs for logprob probes
        if is_logprob:
            logprobs = model.get_logprobs(prompt, choices)
            score = score_fn(logprobs, item)
            response_str = str({k: round(v, 3) for k, v in logprobs.items()
                               if k in choices})
        else:
            response_str = model.generate_short(prompt, max_new_tokens=15)
            score = score_fn(response_str, item)

        results.append({
            "prompt": prompt[:100],
            "response": response_str[:100],
            "score": score,
            "correct": score > 0.5,
            "activations": layer_activations,
        })

    return results


def compute_steering_vector(captures, layer_idx):
    """Compute steering vector from captured activations.

    steering = mean(correct activations) - mean(incorrect activations)
    """
    correct = [c["activations"][layer_idx] for c in captures if c["correct"]]
    incorrect = [c["activations"][layer_idx] for c in captures if not c["correct"]]

    if not correct or not incorrect:
        print(f"  WARNING: layer {layer_idx} — {len(correct)} correct, "
              f"{len(incorrect)} incorrect. Need both.")
        return None

    correct_mean = torch.stack(correct).mean(dim=0)
    incorrect_mean = torch.stack(incorrect).mean(dim=0)
    steering = correct_mean - incorrect_mean

    print(f"  Layer {layer_idx}: {len(correct)} correct, {len(incorrect)} incorrect, "
          f"vector norm={steering.norm():.4f}")

    return steering


def test_steering(model, probe_name, steering_vectors, alpha=1.0, max_items=None):
    """Test steering by injecting vectors during forward pass.

    Monkey-patches _run_module to add steering vectors at target layers.
    """
    from probes.registry import get_probe

    probe = get_probe(probe_name)
    if max_items is None:
        probe.max_items = None
    else:
        probe.max_items = max_items

    # Store original _run_module
    original_run_module = model._run_module

    def steered_run_module(module, x, cache, attn_params, past_len):
        result = original_run_module(module, x, cache, attn_params, past_len)

        # Check if this module corresponds to a steered layer
        for layer_idx, vector in steering_vectors.items():
            layer = model._layer_modules[layer_idx]
            target_module = layer[1] if model._moe_mode else layer  # MLP for MoE
            if module is target_module:
                # Add steering vector to last token position
                device = result.device
                sv = vector.to(device) * alpha
                if result.dim() == 3:
                    result[:, -1, :] += sv
                elif result.dim() == 2:
                    result[-1, :] += sv
                break

        return result

    # Patch and run
    model._run_module = steered_run_module
    try:
        result = probe.run(model)
        score = result["score"] if isinstance(result, dict) else float(result)
    finally:
        model._run_module = original_run_module

    return score


def main():
    parser = argparse.ArgumentParser(
        description="Build and test steering vectors from activation captures")
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe", default="math",
                        help="Probe to use for captures (default: math)")
    parser.add_argument("--layers", nargs="+", type=int, default=[2],
                        help="Target layers for steering (default: 2)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Steering strength multiplier (default: 1.0)")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Max items for capture (default: all)")
    args = parser.parse_args()

    from sweep.exllama_adapter import ExLlamaV2LayerAdapter
    model = ExLlamaV2LayerAdapter(args.model)

    target_layers = args.layers
    print(f"Probe: {args.probe}")
    print(f"Target layers: {target_layers}")
    print(f"Alpha: {args.alpha}")

    # Step 1: Capture activations
    print(f"\n{'='*60}")
    print("STEP 1: Capturing activations...")
    print(f"{'='*60}")
    t0 = time.time()
    captures = capture_activations(model, args.probe, target_layers, args.max_items)
    elapsed = time.time() - t0

    n_correct = sum(1 for c in captures if c["correct"])
    n_incorrect = sum(1 for c in captures if not c["correct"])
    print(f"  Captured {len(captures)} items in {elapsed:.1f}s")
    print(f"  {n_correct} correct, {n_incorrect} incorrect")

    # Step 2: Compute steering vectors
    print(f"\n{'='*60}")
    print("STEP 2: Computing steering vectors...")
    print(f"{'='*60}")
    steering_vectors = {}
    for layer_idx in target_layers:
        sv = compute_steering_vector(captures, layer_idx)
        if sv is not None:
            steering_vectors[layer_idx] = sv

    if not steering_vectors:
        print("ERROR: No steering vectors computed. Need both correct and incorrect samples.")
        sys.exit(1)

    # Step 3: Test steering at various alpha values
    print(f"\n{'='*60}")
    print("STEP 3: Testing steering vectors...")
    print(f"{'='*60}")

    # Baseline
    model.set_layer_path(list(range(model.num_layers)))
    from probes.registry import get_probe
    probe = get_probe(args.probe)
    probe.max_items = args.max_items
    baseline_result = probe.run(model)
    baseline_score = baseline_result["score"] if isinstance(baseline_result, dict) else float(baseline_result)
    print(f"\n  Baseline: {baseline_score:.3f}")

    # Test various steering strengths
    alphas = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    results = []
    for alpha in alphas:
        score = test_steering(model, args.probe, steering_vectors, alpha, args.max_items)
        delta = score - baseline_score
        marker = " +++" if delta > 0.05 else " ---" if delta < -0.05 else ""
        print(f"  Alpha={alpha:.2f}: {score:.3f} (delta={delta:+.3f}){marker}")
        results.append({"alpha": alpha, "score": score, "delta": round(delta, 4)})

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"steering_{args.probe}_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "probe": args.probe,
        "target_layers": target_layers,
        "n_captures": len(captures),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "baseline_score": baseline_score,
        "results": results,
        "vector_norms": {str(k): v.norm().item() for k, v in steering_vectors.items()},
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    # Save steering vectors for reuse
    vectors_file = output_dir / f"steering_vectors_{args.probe}_{timestamp}.pt"
    torch.save(steering_vectors, vectors_file)

    print(f"\n  Results saved to {out_file}")
    print(f"  Vectors saved to {vectors_file}")

    # Summary
    best = max(results, key=lambda r: r["score"])
    print(f"\n{'='*60}")
    print(f"BEST: alpha={best['alpha']:.2f} score={best['score']:.3f} "
          f"(delta={best['delta']:+.3f} vs baseline {baseline_score:.3f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
