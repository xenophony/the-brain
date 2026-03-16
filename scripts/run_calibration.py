#!/usr/bin/env python3
"""
Run calibration questions against API to find sweet-spot probes for circuit mapping.

Scores each question, then reports per-domain accuracy.
Domains scoring 0.3-0.7 are ideal for logprob-based sweep probes.

Usage:
  python scripts/run_calibration.py --model qwen-30b
  python scripts/run_calibration.py --model qwen-30b --no-think
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.calibration_questions import DOMAINS
from scripts.run_baselines import MODEL_REGISTRY, FALLBACK_REGISTRY, _resolve_provider
from sweep.api_adapters import get_adapter

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "calibration"


def score_response(response: str, expected: str, answer_choices: list | None) -> float:
    """Fuzzy score: exact match, contains match, or first-word match."""
    resp = response.strip().lower()
    expected = expected.strip().lower()

    # Strip common preamble
    for prefix in ["the answer is ", "answer: ", "it's ", "it is "]:
        if resp.startswith(prefix):
            resp = resp[len(prefix):]

    # Take first word/line
    first_word = resp.split()[0] if resp.split() else ""
    first_line = resp.split("\n")[0].strip()

    # Exact match
    if resp == expected or first_word == expected:
        return 1.0

    # Contains match (for multi-word answers like "carbon dioxide")
    if expected in resp:
        return 1.0

    # For categorical answers, check if response starts with a valid choice
    if answer_choices:
        for choice in answer_choices:
            if first_word == choice or resp.startswith(choice):
                return 1.0 if choice == expected else 0.0

    # Partial: first line contains the answer
    if expected in first_line:
        return 1.0

    return 0.0


def run_calibration(model_name: str, no_think: bool = False):
    resolved = _resolve_provider(model_name)
    if resolved is None:
        print(f"Cannot resolve model {model_name}")
        return

    provider, sdk_model = resolved
    adapter = get_adapter(provider, sdk_model)

    if no_think:
        _orig = adapter.generate_short
        def _nothink(prompt, max_new_tokens=20, temperature=0.0, _o=_orig):
            return _o(prompt + " /no_think", max_new_tokens, temperature)
        adapter.generate_short = _nothink

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_nothink" if no_think else ""
    out_file = OUTPUT_DIR / f"calibration_{model_name}{suffix}.json"

    results = {}
    total_correct = 0
    total_items = 0

    for domain_name, domain in DOMAINS.items():
        items = domain["items"]
        choices = domain.get("answer_choices")
        domain_results = []
        correct = 0

        print(f"\n=== {domain_name} ({domain['description']}) ===")

        for item in items:
            prompt = item["prompt"] + " Answer with one word."
            try:
                response = adapter.generate_short(
                    prompt, max_new_tokens=10, temperature=0.0)
            except Exception as e:
                response = f"ERROR: {e}"

            score = score_response(response, item["answer"], choices)
            correct += score
            total_correct += score
            total_items += 1

            mark = "✓" if score > 0 else "✗"
            print(f"  {mark} [{item['difficulty']}] {item['prompt'][:50]:50s} "
                  f"expected={item['answer']:12s} got={response[:30]}")

            domain_results.append({
                "prompt": item["prompt"],
                "expected": item["answer"],
                "response": response[:200],
                "score": score,
                "difficulty": item["difficulty"],
            })

        accuracy = correct / len(items) if items else 0
        sweet = "<<< SWEET SPOT" if 0.3 <= accuracy <= 0.7 else ""
        results[domain_name] = {
            "accuracy": round(accuracy, 3),
            "correct": int(correct),
            "total": len(items),
            "items": domain_results,
        }
        print(f"  → {domain_name}: {accuracy:.1%} ({int(correct)}/{len(items)}) {sweet}")

    # Summary
    print(f"\n{'='*60}")
    print(f"CALIBRATION SUMMARY — {model_name}{' (no_think)' if no_think else ''}")
    print(f"{'='*60}")
    print(f"{'Domain':<25s} {'Accuracy':>8s}  {'Signal':>8s}")
    print(f"{'-'*25} {'-'*8}  {'-'*8}")

    for domain_name, r in sorted(results.items(), key=lambda x: x[1]["accuracy"]):
        acc = r["accuracy"]
        if 0.3 <= acc <= 0.7:
            signal = "GOOD"
        elif acc > 0.9:
            signal = "ceiling"
        elif acc < 0.15:
            signal = "floor"
        else:
            signal = "ok"
        print(f"{domain_name:<25s} {acc:>7.1%}  {signal:>8s}")

    overall = total_correct / total_items if total_items else 0
    print(f"\nOverall: {overall:.1%} ({int(total_correct)}/{total_items})")

    # Save
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run calibration questions against API models")
    parser.add_argument("--model", default="qwen-30b",
                        help="Model name from registry")
    parser.add_argument("--no-think", action="store_true",
                        help="Append /no_think to prompts")
    args = parser.parse_args()

    run_calibration(args.model, no_think=args.no_think)


if __name__ == "__main__":
    main()
