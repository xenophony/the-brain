#!/usr/bin/env python3
"""
Run all probes against API models to collect baseline scores.

No (i,j) sweep — just single baseline score per model per probe.
Probes run in parallel per model for ~5-8x speedup.

Usage:
  python scripts/run_baselines.py --models all
  python scripts/run_baselines.py --models claude-sonnet llama-8b
  python scripts/run_baselines.py --probes math spatial eq
  python scripts/run_baselines.py --dry-run
"""

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()  # load .env before any os.environ.get() calls
except ImportError:
    pass  # python-dotenv optional but recommended

from probes.registry import get_probe, list_probes
from sweep.api_adapters import get_adapter, available_providers, _ENV_KEYS


# ---------------------------------------------------------------------------
#  Model registry: friendly name -> (provider, sdk_model_id)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "llama-8b": ("openrouter", "meta-llama/llama-3.1-8b-instruct"),
    "llama-70b": ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    "qwen-30b": ("openrouter", "qwen/qwen3-32b"),
    "claude-sonnet": ("openrouter", "anthropic/claude-sonnet-4"),
    "gpt-5": ("openrouter", "openai/gpt-5"),
}

FALLBACK_REGISTRY = {
    "llama-8b": ("groq", "llama-3.1-8b-instant"),
    "llama-70b": ("groq", "llama-3.3-70b-versatile"),
    "claude-sonnet": ("claude", "claude-sonnet-4-20250514"),
}

ALL_PROBES = [
    "math", "eq", "code", "factual", "spatial",
    "spatial_pong_simple", "spatial_pong_strategic",
    "language", "tool_use",
    "holistic", "planning", "instruction", "hallucination", "sycophancy",
    "consistency", "temporal", "metacognition", "counterfactual",
    "abstraction", "noise_robustness",
    "implication", "negation", "estimation",
    "reasoning", "routing",
]

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "baselines"
OUTPUT_FILE = OUTPUT_DIR / "baseline_scores.json"
RESPONSES_FILE = OUTPUT_DIR / "baseline_responses.json"

# --no-think mode: separate output files so thinking/non-thinking don't clobber
NOTHINK_OUTPUT_FILE = OUTPUT_DIR / "baseline_scores_nothink.json"
NOTHINK_RESPONSES_FILE = OUTPUT_DIR / "baseline_responses_nothink.json"

# Parallelism config
MAX_PARALLEL_PROBES = 6  # max concurrent probes per model
PROBE_TIMEOUT_SECONDS = 300  # 5 minutes max per probe (thinking models need more time)
# Note: daemon threads are abandoned on timeout but underlying sockets may
# keep the process alive. Use os._exit() in extreme cases if needed.


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


def _count_probe_items(probe_name: str) -> int:
    """Estimate number of items in a probe."""
    import importlib
    try:
        mod = importlib.import_module(f"probes.{probe_name}.probe")
        for attr in ("EASY_ITEMS", "HARD_ITEMS", "QUESTIONS", "SCENARIOS",
                      "BASE_QUESTIONS", "CHALLENGES"):
            val = getattr(mod, attr, None)
            if val is not None and hasattr(val, "__len__"):
                return len(val)
        # Check for both EASY+HARD
        easy = getattr(mod, "EASY_ITEMS", [])
        hard = getattr(mod, "HARD_ITEMS", [])
        if easy or hard:
            return len(easy) + len(hard)
    except Exception:
        pass
    return 15  # fallback estimate


# ---------------------------------------------------------------------------
#  Cost estimation (dry-run)
# ---------------------------------------------------------------------------

_EST_INPUT_TOKENS_PER_PROBE_ITEM = 200
_EST_OUTPUT_TOKENS_PER_PROBE_ITEM = 15

PRICING = {  # (input_per_1M, output_per_1M) USD, updated 2026-03
    "llama-8b": (0.06, 0.06),
    "llama-70b": (0.40, 0.40),
    "qwen-30b": (0.30, 0.30),
    "claude-sonnet": (3.00, 15.00),
    "gpt-5": (2.00, 8.00),       # OpenRouter GPT-5
}


def _estimate_cost(model_names: list[str], probe_names: list[str]) -> dict:
    """Estimate cost for running baselines."""
    breakdown = {}
    total = 0.0
    total_items = 0
    for model_name in model_names:
        input_price, output_price = PRICING.get(model_name, (1.0, 5.0))
        n_items = sum(_count_probe_items(p) for p in probe_names)
        total_input = n_items * _EST_INPUT_TOKENS_PER_PROBE_ITEM
        total_output = n_items * _EST_OUTPUT_TOKENS_PER_PROBE_ITEM
        cost = (total_input / 1_000_000 * input_price) + (total_output / 1_000_000 * output_price)
        breakdown[model_name] = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "n_items": n_items,
            "cost_usd": round(cost, 4),
        }
        total += cost
        total_items += n_items
    return {"breakdown": breakdown, "total_usd": round(total, 4), "total_items": total_items}


def _estimate_time(n_models: int, n_probes: int, total_items: int) -> dict:
    """Estimate sequential vs parallel runtime."""
    avg_latency_per_item = 0.8  # seconds per API call
    sequential_seconds = total_items * avg_latency_per_item
    # Parallel: probes run concurrently, ~MAX_PARALLEL_PROBES at a time
    items_per_probe = total_items / max(n_probes * n_models, 1)
    parallel_rounds = (n_probes + MAX_PARALLEL_PROBES - 1) // MAX_PARALLEL_PROBES
    parallel_seconds_per_model = parallel_rounds * items_per_probe * avg_latency_per_item
    parallel_seconds = parallel_seconds_per_model * n_models
    speedup = sequential_seconds / max(parallel_seconds, 1)
    return {
        "sequential_minutes": round(sequential_seconds / 60, 1),
        "parallel_minutes": round(parallel_seconds / 60, 1),
        "speedup": round(speedup, 1),
    }


# ---------------------------------------------------------------------------
#  Single probe runner (for thread pool)
# ---------------------------------------------------------------------------

def _run_single_probe(adapter, probe_name: str, progress_counter: dict,
                      progress_lock: Lock, model_name: str) -> dict:
    """Run one probe against one adapter. Thread-safe, with per-probe timeout."""
    import threading

    t0 = time.perf_counter()
    n_items = _count_probe_items(probe_name)

    # Run probe in a daemon thread with timeout
    result_holder = [None]
    error_holder = [None]

    def _target():
        try:
            probe = get_probe(probe_name)
            probe.log_responses = True
            result_holder[0] = probe.run(adapter)
        except Exception as exc:
            error_holder[0] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=PROBE_TIMEOUT_SECONDS)

    elapsed = time.perf_counter() - t0

    if thread.is_alive():
        # Probe timed out
        error_msg = (f"Probe timed out after {PROBE_TIMEOUT_SECONDS}s "
                     f"({n_items} items, model={model_name})")
        with progress_lock:
            progress_counter["done"] += 1
            done = progress_counter["done"]
            total = progress_counter["total"]
            print(f"\r  [{model_name}] {done}/{total} probes | "
                  f"{probe_name}: TIMEOUT after {elapsed:.0f}s", end="", flush=True)
        return {
            "probe": probe_name,
            "score": 0.0,
            "n_items": n_items,
            "latency_seconds": round(elapsed, 2),
            "error": error_msg,
        }

    if error_holder[0] is not None:
        exc = error_holder[0]
        error_msg = f"{type(exc).__name__}: {exc}"
        with progress_lock:
            progress_counter["done"] += 1
            done = progress_counter["done"]
            total = progress_counter["total"]
            print(f"\r  [{model_name}] {done}/{total} probes | "
                  f"{probe_name}: ERROR {type(exc).__name__} ({elapsed:.1f}s)", end="", flush=True)
        return {
            "probe": probe_name,
            "score": 0.0,
            "n_items": n_items,
            "latency_seconds": round(elapsed, 2),
            "error": error_msg,
        }

    # Success
    result = result_holder[0]
    item_results = None
    if isinstance(result, dict):
        score = result.get("score", 0.0)
        item_results = result.get("item_results", None)
    else:
        score = float(result)

    with progress_lock:
        progress_counter["done"] += 1
        done = progress_counter["done"]
        total = progress_counter["total"]
        print(f"\r  [{model_name}] {done}/{total} probes | "
              f"{probe_name}: {score:.3f} ({elapsed:.1f}s)", end="", flush=True)

    # Print sample responses (one good, one zero if exists)
    if item_results:
        good = [r for r in item_results if r.get("score", 0) > 0.5]
        bad = [r for r in item_results if r.get("score", 0) == 0.0]
        samples = []
        if good:
            samples.append(("GOOD", good[0]))
        if bad:
            samples.append(("ZERO", bad[0]))
        if samples:
            print(f"\n    --- {probe_name} samples ---")
            for label, s in samples[:2]:
                # Handle different probe response key names
                resp = (s.get("response") or s.get("raw_response")
                        or s.get("reasoning_raw") or s.get("direct_raw") or "")
                resp = resp[:120]
                extracted = s.get("extracted", s.get("reasoning_extracted", ""))
                score_val = s.get("score", s.get("item_score", "?"))
                print(f"    [{label} score={score_val}] resp={resp!r}")
                if extracted:
                    print(f"    extracted: {extracted!r}")

    return {
        "probe": probe_name,
        "score": round(score, 4),
        "n_items": n_items,
        "latency_seconds": round(elapsed, 2),
        "error": None,
        "item_results": item_results,
    }


# ---------------------------------------------------------------------------
#  Main runner
# ---------------------------------------------------------------------------

def run_baselines(
    model_names: list[str],
    probe_names: list[str],
    dry_run: bool = False,
    resume: bool = True,
    no_think: bool = False,
) -> dict:
    """Run probes against API models with parallel execution."""

    if no_think:
        print("Mode: /no_think (suppressing thinking, separate output files)")

    # --- Dry run --------------------------------------------------------
    if dry_run:
        estimate = _estimate_cost(model_names, probe_names)
        time_est = _estimate_time(len(model_names), len(probe_names),
                                   estimate["total_items"])
        print("\n=== Cost Estimate (dry run) ===")
        for name, info in estimate["breakdown"].items():
            avail, reason = _check_model_available(name)
            status = "OK" if avail else f"SKIP ({reason})"
            print(f"  {name:20s}  ~{info['input_tokens']:>7,} in / "
                  f"~{info['output_tokens']:>6,} out  ${info['cost_usd']:.4f}  [{status}]")
        print(f"\n  Total estimated cost: ${estimate['total_usd']:.4f}")
        print(f"\n  Sequential estimate: {time_est['sequential_minutes']} minutes")
        print(f"  Parallel estimate:   {time_est['parallel_minutes']} minutes")
        print(f"  Speedup:             {time_est['speedup']}x")
        if estimate["total_usd"] > 50:
            print("  WARNING: Estimated cost exceeds $50!")
        return {}

    # --- Real run -------------------------------------------------------
    out_file = NOTHINK_OUTPUT_FILE if no_think else OUTPUT_FILE
    resp_file = NOTHINK_RESPONSES_FILE if no_think else RESPONSES_FILE
    out_file.parent.mkdir(parents=True, exist_ok=True)
    results = _load_existing(out_file)
    responses = _load_existing(resp_file)
    if not resume:
        # Clear only the specified models, preserve others
        for name in model_names:
            results.pop(name, None)
            responses.pop(name, None)
    checkpoint_lock = Lock()

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
            continue
        provider, sdk_model = resolved

        if model_name not in results:
            results[model_name] = {}

        try:
            adapter = get_adapter(provider, sdk_model)
        except (ImportError, ValueError) as exc:
            print(f"ERROR creating adapter for {model_name}: {exc}")
            continue

        # Wrap adapter to append /no_think to all prompts
        if no_think:
            _orig_generate = adapter.generate_short
            def _nothink_generate(prompt, max_new_tokens=20, temperature=0.0,
                                  _orig=_orig_generate):
                return _orig(prompt + " /no_think", max_new_tokens, temperature)
            adapter.generate_short = _nothink_generate

        # Filter probes: skip already-completed in resume mode
        probes_to_run = []
        for probe_name in probe_names:
            if resume and probe_name in results[model_name]:
                existing = results[model_name][probe_name]
                if existing.get("error") is None:
                    continue
            probes_to_run.append(probe_name)

        cached = len(probe_names) - len(probes_to_run)
        if cached > 0:
            print(f"[{m_idx}/{len(runnable)}] {model_name}: {cached} cached, "
                  f"{len(probes_to_run)} to run")
        if not probes_to_run:
            print(f"[{m_idx}/{len(runnable)}] {model_name}: all cached, skipping")
            continue

        # Run probes in parallel
        progress_lock = Lock()
        progress_counter = {"done": 0, "total": len(probes_to_run)}

        model_t0 = time.perf_counter()
        print(f"[{m_idx}/{len(runnable)}] {model_name}: running {len(probes_to_run)} probes "
              f"(max {MAX_PARALLEL_PROBES} parallel)...")

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PROBES) as executor:
            futures = {
                executor.submit(
                    _run_single_probe, adapter, pname,
                    progress_counter, progress_lock, model_name
                ): pname
                for pname in probes_to_run
            }

            for future in as_completed(futures):
                probe_result = future.result()
                pname = probe_result["probe"]

                # Thread-safe checkpoint
                with checkpoint_lock:
                    results[model_name][pname] = {
                        "score": probe_result["score"],
                        "n_items": probe_result["n_items"],
                        "latency_seconds": probe_result["latency_seconds"],
                        "error": probe_result["error"],
                    }
                    _atomic_save(results, out_file)

                    # Save item-level responses if available
                    if probe_result.get("item_results"):
                        if model_name not in responses:
                            responses[model_name] = {}
                        responses[model_name][pname] = probe_result["item_results"]
                        _atomic_save(responses, resp_file)

        model_elapsed = time.perf_counter() - model_t0
        n_completed = len(probes_to_run)
        total_items = sum(_count_probe_items(p) for p in probes_to_run)
        print(f"\n  [{model_name}] Complete — {n_completed} probes, ~{total_items} questions, "
              f"{model_elapsed:.1f}s")

    print(f"\nResults saved to {out_file}")
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
        help="Estimate cost and time only, don't run",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing results and re-run everything",
    )
    parser.add_argument(
        "--no-think", action="store_true",
        help="Append /no_think to prompts (matches local sweep mode). "
             "Results saved to separate files (*_nothink.json)",
    )
    parser.add_argument(
        "--subset", type=int, default=None, metavar="N",
        help="Run only N total items per probe (N//2 easy + N//2 hard). "
             "Useful for quick smoke tests.",
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

    # --- Subset slicing: temporarily reduce probe item counts ----------
    _originals = {}
    if args.subset is not None and args.subset > 0:
        import importlib
        half = max(args.subset // 2, 1)
        for probe_name in probes:
            try:
                mod = importlib.import_module(f"probes.{probe_name}.probe")
            except ImportError:
                continue
            saved = {}
            for attr in ("EASY_ITEMS", "HARD_ITEMS", "EASY_PAIRS", "HARD_PAIRS"):
                val = getattr(mod, attr, None)
                if val is not None and hasattr(val, "__len__"):
                    saved[attr] = list(val)
                    setattr(mod, attr, val[:half])
            if saved:
                _originals[probe_name] = (mod, saved)
        print(f"Subset mode: limiting each probe to ~{args.subset} items "
              f"({half} easy + {half} hard)")

    try:
        results = run_baselines(
            model_names=models,
            probe_names=probes,
            dry_run=args.dry_run,
            resume=not args.no_resume,
            no_think=args.no_think,
        )
    finally:
        # Restore original item lists
        for probe_name, (mod, saved) in _originals.items():
            for attr, original_val in saved.items():
                setattr(mod, attr, original_val)

    # Auto-generate calibration report after real runs
    if results and not args.dry_run:
        # Quick summary to stdout
        print("\n=== Quick Calibration Summary ===")
        all_probe_names = set()
        for m in results:
            all_probe_names.update(results[m].keys())
        for probe_name in sorted(all_probe_names):
            scores = [
                results[m][probe_name]["score"]
                for m in results
                if probe_name in results[m] and results[m][probe_name].get("error") is None
            ]
            if scores:
                rng = max(scores) - min(scores)
                flag = " *** CEILING" if max(scores) > 0.95 else ""
                flag += " *** LOW RANGE" if rng < 0.15 and len(scores) > 1 else ""
                print(f"  {probe_name:22s} min={min(scores):.3f}  max={max(scores):.3f}  range={rng:.3f}{flag}")

        # Full calibration report
        try:
            from analysis.calibration import generate_calibration_report
            report_path = OUTPUT_DIR / "CALIBRATION_REPORT.md"
            generate_calibration_report(results, str(report_path))
            print(f"\nCalibration report saved to {report_path}")
        except Exception as exc:
            print(f"\nWarning: could not generate calibration report: {exc}")


if __name__ == "__main__":
    main()
