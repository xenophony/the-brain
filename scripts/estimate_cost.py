#!/usr/bin/env python3
"""
Standalone cost estimator for baselines and sweeps.

Usage:
  python scripts/estimate_cost.py --task baselines --models all
  python scripts/estimate_cost.py --task sweep --model Qwen3-30B-A3B --n-layers 48
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
#  Pricing (hardcoded with date for transparency)
# ---------------------------------------------------------------------------

PRICING = {  # (input_per_1M, output_per_1M) USD, updated 2026-03
    "llama-8b": (0.06, 0.06),      # OpenRouter Llama 3.1 8B
    "llama-70b": (0.40, 0.40),     # OpenRouter Llama 3.3 70B
    "qwen-30b": (0.30, 0.30),      # OpenRouter Qwen 32B
    "claude-sonnet": (3.00, 15.00), # OpenRouter Claude Sonnet 4
    "gpt-5": (2.00, 8.00),         # OpenRouter GPT-5
}

# Map model registry names to pricing keys (direct mapping now)
_PRICING_KEY = {
    "llama-8b": "llama-8b",
    "llama-70b": "llama-70b",
    "qwen-30b": "qwen-30b",
    "claude-sonnet": "claude-sonnet",
    "gpt-5": "gpt-5",
}

# Probe token estimates
N_PROBES = 19
EST_ITEMS_PER_PROBE = 15
EST_INPUT_TOKENS_PER_ITEM = 200
EST_OUTPUT_TOKENS_PER_ITEM = 15

ALL_MODELS = ["llama-8b", "llama-70b", "qwen-30b", "claude-sonnet", "gpt-5"]


# ---------------------------------------------------------------------------
#  Baseline cost
# ---------------------------------------------------------------------------

def estimate_baseline_cost(model_names: list[str], n_probes: int = N_PROBES) -> dict:
    """Estimate cost for running baselines against given models."""
    breakdown = {}
    total = 0.0

    for model in model_names:
        pricing_key = _PRICING_KEY.get(model, model)
        if pricing_key not in PRICING:
            breakdown[model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "note": f"No pricing data for '{pricing_key}'",
            }
            continue

        input_price, output_price = PRICING[pricing_key]
        total_input = n_probes * EST_ITEMS_PER_PROBE * EST_INPUT_TOKENS_PER_ITEM
        total_output = n_probes * EST_ITEMS_PER_PROBE * EST_OUTPUT_TOKENS_PER_ITEM
        cost = (total_input / 1_000_000 * input_price) + (total_output / 1_000_000 * output_price)

        breakdown[model] = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost_usd": round(cost, 4),
        }
        total += cost

    return {"task": "baselines", "breakdown": breakdown, "total_usd": round(total, 4)}


# ---------------------------------------------------------------------------
#  Sweep cost (GPU time, not API)
# ---------------------------------------------------------------------------

def estimate_sweep_cost(
    n_layers: int,
    max_block: int | None = None,
    secs_per_config: float = 10.0,
    gpu_cost_per_hour: float = 0.40,
) -> dict:
    """Estimate GPU time and cost for an (i,j) sweep."""
    max_block = max_block or n_layers
    n_configs = 0
    for i in range(n_layers):
        for j in range(i + 1, min(i + max_block + 1, n_layers + 1)):
            n_configs += 1
    # Add baseline
    n_configs += 1

    total_seconds = n_configs * secs_per_config
    total_hours = total_seconds / 3600
    cost = total_hours * gpu_cost_per_hour

    return {
        "task": "sweep",
        "n_layers": n_layers,
        "max_block": max_block,
        "n_configs": n_configs,
        "secs_per_config": secs_per_config,
        "total_hours": round(total_hours, 2),
        "gpu_cost_per_hour": gpu_cost_per_hour,
        "total_usd": round(cost, 2),
    }


# ---------------------------------------------------------------------------
#  Pretty print
# ---------------------------------------------------------------------------

def print_baseline_estimate(est: dict) -> None:
    print("\n=== Baseline Cost Estimate ===")
    print(f"{'Model':<20s} {'Input tok':>10s} {'Output tok':>10s} {'Cost':>10s}")
    print("-" * 55)
    for model, info in est["breakdown"].items():
        note = info.get("note", "")
        if note:
            print(f"  {model:<18s} {'?':>10s} {'?':>10s} {'?':>10s}  ({note})")
        else:
            print(f"  {model:<18s} {info['input_tokens']:>10,} {info['output_tokens']:>10,} "
                  f"${info['cost_usd']:>8.4f}")
    print("-" * 55)
    print(f"  {'TOTAL':<18s} {'':>10s} {'':>10s} ${est['total_usd']:>8.4f}")
    if est["total_usd"] > 50:
        print("\n  WARNING: Estimated cost exceeds $50!")


def print_sweep_estimate(est: dict) -> None:
    print("\n=== Sweep Cost Estimate ===")
    print(f"  Layers:          {est['n_layers']}")
    print(f"  Max block:       {est['max_block']}")
    print(f"  Configs:         {est['n_configs']}")
    print(f"  Secs/config:     {est['secs_per_config']}")
    print(f"  Total time:      {est['total_hours']:.1f} hours")
    print(f"  GPU rate:        ${est['gpu_cost_per_hour']:.2f}/hr")
    print(f"  Estimated cost:  ${est['total_usd']:.2f}")
    if est["total_usd"] > 50:
        print("\n  WARNING: Estimated cost exceeds $50!")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cost estimator for baselines and sweeps")
    parser.add_argument("--task", required=True, choices=["baselines", "sweep"],
                        help="What to estimate")
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="Models for baseline estimate (default: all)")
    parser.add_argument("--model", default=None,
                        help="Model name for sweep estimate (used for display only)")
    parser.add_argument("--n-layers", type=int, default=48,
                        help="Number of layers (for sweep estimate)")
    parser.add_argument("--max-block", type=int, default=None,
                        help="Max block size (for sweep estimate)")
    parser.add_argument("--secs-per-config", type=float, default=10.0,
                        help="Seconds per config (for sweep estimate)")
    parser.add_argument("--gpu-rate", type=float, default=0.40,
                        help="GPU cost per hour in USD (for sweep estimate)")

    args = parser.parse_args()

    if args.task == "baselines":
        models = ALL_MODELS if "all" in args.models else args.models
        est = estimate_baseline_cost(models)
        print_baseline_estimate(est)

    elif args.task == "sweep":
        est = estimate_sweep_cost(
            n_layers=args.n_layers,
            max_block=args.max_block,
            secs_per_config=args.secs_per_config,
            gpu_cost_per_hour=args.gpu_rate,
        )
        if args.model:
            print(f"  Model: {args.model}")
        print_sweep_estimate(est)


if __name__ == "__main__":
    main()
