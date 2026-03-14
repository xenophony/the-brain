#!/usr/bin/env python3
"""
Run all probes against API models to collect baseline scores.

No (i,j) sweep — just single baseline score per model per probe.

Usage:
  python scripts/run_baselines.py --models all
  python scripts/run_baselines.py --models claude-sonnet gemini-2.5-pro
  python scripts/run_baselines.py --probes math spatial eq
  python scripts/run_baselines.py --dry-run
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from probes.registry import get_probe, list_probes
from sweep.api_adapters import get_adapter, available_providers, _ENV_KEYS


# ---------------------------------------------------------------------------
#  Model registry: friendly name -> (provider, sdk_model_id)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "llama-8b": ("openrouter", "meta-llama/llama-3.1-8b-instruct"),
    "llama-70b": ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    "qwen-30b": ("openrouter", "qwen/qwen-2.5-32b-instruct"),
    "claude-sonnet": ("openrouter", "anthropic/claude-sonnet-4"),
    "gemini-3-pro": ("openrouter", "google/gemini-2.5-pro-preview-05-06"),
}

FALLBACK_REGISTRY = {
    "llama-8b": ("groq", "llama-3.1-8b-instant"),
    "llama-70b": ("groq", "llama-3.3-70b-versatile"),
    "claude-sonnet": ("claude", "claude-sonnet-4-20250514"),
    "gemini-3-pro": ("gemini", "gemini-2.5-pro-preview-05-06"),
}

ALL_PROBES = [
    "math", "eq", "code", "factual", "spatial", "language", "tool_use",
    "holistic", "planning", "instruction", "hallucination", "sycophancy",
    "consistency", "temporal", "metacognition", "counterfactual",
    "abstraction", "noise_robustness",
]

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "baselines"
OUTPUT_FILE = OUTPUT_DIR / "baseline_scores.json"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _atomic_save(data: dict, path: Path) -> None:
    """Write JSON atomically via tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".baseline_"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # os.replace is atomic on POSIX; on Windows it may raise if target
        # exists on some older Python versions, but 3.10+ handles it.
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_existing(path: Path) -> dict:
    """Load existing baseline scores if present."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _resolve_provider(model_name: str) -> tuple[str, str] | None:
    """Resolve model to (provider, sdk_model_id), trying OpenRouter first then fallback."""
    if model_name not in MODEL_REGISTRY:
        return None
    provider, sdk_model = MODEL_REGISTRY[model_name]
    env_key = _ENV_KEYS.get(provider, "")
    if os.environ.get(env_key):
        return provider, sdk_model
    # Try fallback if OpenRouter key not set
    if model_name in FALLBACK_REGISTRY:
        fb_provider, fb_model = FALLBACK_REGISTRY[model_name]
        fb_env_key = _ENV_KEYS.get(fb_provider, "")
        if os.environ.get(fb_env_key):
            return fb_provider, fb_model
    return None


def _check_model_available(model_name: str) -> tuple[bool, str]:
    """Check if the API key for a model is set. Returns (ok, reason)."""
    if model_name not in MODEL_REGISTRY:
        return False, f"Unknown model '{model_name}'"
    resolved = _resolve_provider(model_name)
    if resolved is not None:
        return True, ""
    provider, _ = MODEL_REGISTRY[model_name]
    env_key = _ENV_KEYS.get(provider, "")
    fb_keys = []
    if model_name in FALLBACK_REGISTRY:
        fb_provider, _ = FALLBACK_REGISTRY[model_name]
        fb_env_key = _ENV_KEYS.get(fb_provider, "")
        fb_keys.append(fb_env_key)
    needed = [env_key] + fb_keys
    return False, f"None of {needed} set"


# ---------------------------------------------------------------------------
#  Cost estimation (dry-run)
# ---------------------------------------------------------------------------

# Approximate tokens per probe invocation (input + output)
_EST_INPUT_TOKENS_PER_PROBE_ITEM = 200   # average prompt length in tokens
_EST_OUTPUT_TOKENS_PER_PROBE_ITEM = 15   # probes require short outputs
_EST_ITEMS_PER_PROBE = 15                # average items per probe

PRICING = {  # (input_per_1M, output_per_1M) USD, updated 2026-03
    "llama-8b": (0.06, 0.06),      # OpenRouter Llama 3.1 8B
    "llama-70b": (0.40, 0.40),     # OpenRouter Llama 3.3 70B
    "qwen-30b": (0.30, 0.30),      # OpenRouter Qwen 32B
    "claude-sonnet": (3.00, 15.00), # OpenRouter Claude Sonnet 4
    "gemini-3-pro": (2.00, 12.00),  # OpenRouter Gemini 3 Pro Preview
}


def _estimate_cost(model_names: list[str], probe_names: list[str]) -> dict:
    """Estimate cost for running baselines."""
    breakdown = {}
    total = 0.0
    for model_name in model_names:
        input_price, output_price = PRICING.get(model_name, (1.0, 5.0))
        n_probes = len(probe_names)
        total_input = n_probes * _EST_ITEMS_PER_PROBE * _EST_INPUT_TOKENS_PER_PROBE_ITEM
        total_output = n_probes * _EST_ITEMS_PER_PROBE * _EST_OUTPUT_TOKENS_PER_PROBE_ITEM
        cost = (total_input / 1_000_000 * input_price) + (total_output / 1_000_000 * output_price)
        breakdown[model_name] = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost_usd": round(cost, 4),
        }
        total += cost
    return {"breakdown": breakdown, "total_usd": round(total, 4)}


# ---------------------------------------------------------------------------
#  Main runner
# ---------------------------------------------------------------------------

def run_baselines(
    model_names: list[str],
    probe_names: list[str],
    dry_run: bool = False,
    resume: bool = True,
) -> dict:
    """Run probes against API models, return and save baseline scores."""

    # --- Dry run --------------------------------------------------------
    if dry_run:
        estimate = _estimate_cost(model_names, probe_names)
        print("\n=== Cost Estimate (dry run) ===")
        for name, info in estimate["breakdown"].items():
            avail, reason = _check_model_available(name)
            status = "OK" if avail else f"SKIP ({reason})"
            print(f"  {name:20s}  ~{info['input_tokens']:>7,} in / "
                  f"~{info['output_tokens']:>6,} out  ${info['cost_usd']:.4f}  [{status}]")
        print(f"\n  Total estimated cost: ${estimate['total_usd']:.4f}")
        if estimate["total_usd"] > 50:
            print("  WARNING: Estimated cost exceeds $50!")
        return {}

    # --- Real run -------------------------------------------------------
    results = _load_existing(OUTPUT_FILE) if resume else {}
    total_probes = len(probe_names)
    total_models = len(model_names)

    # Filter to available models
    runnable = []
    for name in model_names:
        ok, reason = _check_model_available(name)
        if ok:
            runnable.append(name)
        else:
            print(f"SKIP: {name} ({reason})")

    if not runnable:
        print("No models available to run. Set API keys and retry.")
        return results

    for m_idx, model_name in enumerate(runnable, 1):
        resolved = _resolve_provider(model_name)
        if resolved is None:
            print(f"SKIP: {model_name} (no API key)")
            continue
        provider, sdk_model = resolved

        if model_name not in results:
            results[model_name] = {}

        try:
            adapter = get_adapter(provider, sdk_model)
        except (ImportError, ValueError) as exc:
            print(f"ERROR creating adapter for {model_name}: {exc}")
            continue

        for p_idx, probe_name in enumerate(probe_names, 1):
            # Skip if already completed (resume mode)
            if resume and probe_name in results[model_name]:
                existing = results[model_name][probe_name]
                if existing.get("error") is None:
                    print(f"[{p_idx}/{total_probes} probes] [{m_idx}/{len(runnable)} models] "
                          f"{probe_name} on {model_name}... CACHED score={existing['score']:.3f}")
                    continue

            print(f"[{p_idx}/{total_probes} probes] [{m_idx}/{len(runnable)} models] "
                  f"{probe_name} on {model_name}...", end=" ", flush=True)

            t0 = time.perf_counter()
            try:
                probe = get_probe(probe_name)
                score = probe.run(adapter)
                elapsed = time.perf_counter() - t0
                # Count items (probes have varying item counts)
                n_items = getattr(probe, "n_items", None)
                if n_items is None:
                    # Heuristic: check common attributes
                    for attr in ("QUESTIONS", "SCENARIOS", "items", "ITEMS",
                                 "SENTENCES", "CHALLENGES"):
                        val = getattr(probe, attr, None)
                        if val is not None and hasattr(val, "__len__"):
                            n_items = len(val)
                            break
                    else:
                        n_items = _EST_ITEMS_PER_PROBE

                results[model_name][probe_name] = {
                    "score": round(score, 4),
                    "n_items": n_items,
                    "latency_seconds": round(elapsed, 2),
                    "error": None,
                }
                print(f"score={score:.3f} ({elapsed:.1f}s)")

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                results[model_name][probe_name] = {
                    "score": 0.0,
                    "n_items": 0,
                    "latency_seconds": round(elapsed, 2),
                    "error": str(exc),
                }
                print(f"ERROR: {exc}")

            # Atomic checkpoint after every probe/model pair
            _atomic_save(results, OUTPUT_FILE)

    print(f"\nResults saved to {OUTPUT_FILE}")
    return results


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run probes against API models for baseline calibration"
    )
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        help=f"Models to test. 'all' = all registered. Available: {list(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--probes", nargs="+", default=["all"],
        help=f"Probes to run. 'all' = all 18. Available: {ALL_PROBES}",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Estimate cost only, don't run",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing results and re-run everything",
    )
    args = parser.parse_args()

    models = list(MODEL_REGISTRY.keys()) if "all" in args.models else args.models
    probes = ALL_PROBES if "all" in args.probes else args.probes

    # Validate probe names
    available = list_probes()
    bad = [p for p in probes if p not in available]
    if bad:
        print(f"Warning: unknown probes {bad}, skipping them")
        probes = [p for p in probes if p in available]

    run_baselines(
        model_names=models,
        probe_names=probes,
        dry_run=args.dry_run,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
