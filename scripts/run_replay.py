#!/usr/bin/env python
"""
CLI runner for harvest-replay layerwise analysis.

Loads harvested generative responses and runs layerwise analysis comparing
raw (question-only) vs replay (question + model's own reasoning) conditions.

Usage:
    python scripts/run_replay.py --harvest-dir results/harvested/ --probes causal_logprob,logic_logprob
    python scripts/run_replay.py --mock --probes causal_logprob
    python scripts/run_replay.py --mock --probes all

Options:
    --model PATH        Model path (required unless --mock)
    --harvest-dir DIR   Directory with harvested response JSONs
    --probes NAMES      Comma-separated probe names, or "all"
    --output-dir DIR    Output directory (default: results/layerwise/replay/)
    --mock              Use MockAdapter (also generates mock harvest data)
    --max-items N       Max items per probe (0 = all)
    --mode MODE         MockAdapter mode (default: random)
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_mock_harvest(probe_name: str, mode: str = "perfect") -> dict:
    """Create synthetic harvest data for mock testing.

    Generates plausible harvested responses without needing a real model.
    """
    try:
        from probes.registry import get_probe
        import probes  # trigger auto-discovery
        probe = get_probe(probe_name)
    except KeyError:
        return None

    if not hasattr(probe, 'ITEMS') or not hasattr(probe, 'CHOICES'):
        return None

    items = probe._limit(probe.ITEMS)
    choices = probe.CHOICES

    harvested_items = []
    for item in items:
        expected = item["answer"].lower()
        # Create fake thinking
        thinking = f"Let me reason about this. The question asks: {item['prompt'][:60]}. Based on my understanding, the answer should be {expected}."

        harvested_items.append({
            "prompt": item["prompt"],
            "expected": expected,
            "choices": choices,
            "full_response": f"<think>{thinking}</think>\n{expected}",
            "extracted_answer": expected,
            "correct": True,  # mock: all correct
            "thinking": thinking,
            "difficulty": item.get("difficulty", "hard"),
        })

    return {
        "probe_name": probe_name,
        "model": "mock",
        "n_items": len(harvested_items),
        "n_correct": len(harvested_items),
        "accuracy": 1.0,
        "items": harvested_items,
    }


def discover_harvest_probes(harvest_dir: Path) -> list[str]:
    """Find all probe names with harvested response files."""
    probes = []
    for f in sorted(harvest_dir.glob("*_responses.json")):
        name = f.stem.replace("_responses", "")
        probes.append(name)
    return probes


def main():
    parser = argparse.ArgumentParser(
        description="Run harvest-replay layerwise analysis")
    default_model = str(project_root / "models" / "Qwen3-30B-A3B-exl2")
    parser.add_argument("--model", type=str, default=default_model,
                        help="Model path")
    parser.add_argument("--harvest-dir", type=str,
                        default="results/harvested/",
                        help="Directory with harvested responses")
    parser.add_argument("--probes", type=str, default=None,
                        help="Comma-separated probe names, or 'all'")
    parser.add_argument("--output-dir", type=str,
                        default="results/layerwise/replay/")
    parser.add_argument("--mock", action="store_true",
                        help="Use MockAdapter (generates mock harvest data)")
    parser.add_argument("--max-items", type=int, default=0,
                        help="Max items per probe (0 = all)")
    parser.add_argument("--mode", type=str, default="random",
                        help="MockAdapter mode (default: random)")

    args = parser.parse_args()

    # Load model
    if args.mock:
        from sweep.mock_adapter import MockAdapter
        model = MockAdapter(mode=args.mode, seed=42)
        print(f"Using MockAdapter (mode={args.mode}, {model.num_layers} layers)")
    else:
        from sweep.exllama_adapter import ExLlamaV2LayerAdapter
        model = ExLlamaV2LayerAdapter(args.model)
        print(f"Loaded model: {model.num_layers} layers")

    harvest_dir = Path(args.harvest_dir)

    # Resolve probe names
    if args.mock:
        # For mock mode, use specified probes or defaults
        if args.probes is None or args.probes == "all":
            # Default set for mock testing
            probe_names = ["causal_logprob", "logic_logprob"]
        else:
            probe_names = [p.strip() for p in args.probes.split(",")]
    else:
        if args.probes is None or args.probes == "all":
            probe_names = discover_harvest_probes(harvest_dir)
        else:
            probe_names = [p.strip() for p in args.probes.split(",")]

    if not probe_names:
        print("No probes to analyze. Run harvest_responses.py first, or use --mock.")
        return

    print(f"\nAnalyzing {len(probe_names)} probes: {probe_names}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import replay probe
    from probes.layerwise.replay import ReplayLayerwiseProbe

    # Run replay analysis for each probe
    t0 = time.time()
    all_results = {}

    for probe_name in probe_names:
        print(f"\n--- {probe_name} ---")

        # Load or create harvest data
        if args.mock:
            harvest_data = create_mock_harvest(probe_name)
            if harvest_data is None:
                print(f"  Warning: could not create mock data for {probe_name}")
                continue
            harvest_file = None
        else:
            harvest_file = str(harvest_dir / f"{probe_name}_responses.json")
            if not Path(harvest_file).exists():
                print(f"  Warning: harvest file not found: {harvest_file}")
                continue
            harvest_data = None

        # Create and run replay probe
        replay_probe = ReplayLayerwiseProbe(
            harvest_file=harvest_file,
            probe_name=probe_name,
            harvest_data=harvest_data,
        )
        if args.max_items > 0:
            replay_probe.max_items = args.max_items

        tp = time.time()
        try:
            result = replay_probe.run(model)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - tp

        # Print summary
        raw_conv = result.get("raw_convergence_layer", "N/A")
        replay_conv = result.get("replay_convergence_layer", "N/A")
        conv_delta = result.get("mean_convergence_delta", "N/A")
        n_items = result.get("n_correct_items", 0)

        print(f"  Items: {n_items} correct / {result.get('n_total_items', 0)} total")
        print(f"  Raw convergence layer: {raw_conv}")
        print(f"  Replay convergence layer: {replay_conv}")
        if conv_delta is not None and conv_delta != "N/A":
            direction = "earlier" if conv_delta < 0 else "later"
            print(f"  Convergence delta: {conv_delta:+.2f} ({direction} with reasoning)")
        print(f"  Time: {elapsed:.1f}s")

        all_results[probe_name] = result

        # Save individual result
        result_file = output_dir / f"{probe_name}_replay.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

    total_time = time.time() - t0

    # Print cross-probe summary
    print(f"\n{'='*60}")
    print(f"Completed {len(all_results)}/{len(probe_names)} probes in {total_time:.1f}s")

    if all_results:
        print(f"\n{'Probe':<25} {'Raw Conv':>10} {'Replay Conv':>12} {'Delta':>8}")
        print("-" * 60)
        for name, result in all_results.items():
            raw_c = result.get("raw_convergence_layer", "N/A")
            rep_c = result.get("replay_convergence_layer", "N/A")
            delta = result.get("mean_convergence_delta")
            delta_str = f"{delta:+.2f}" if delta is not None else "N/A"
            print(f"  {name:<23} {str(raw_c):>10} {str(rep_c):>12} {delta_str:>8}")

    # Save combined results
    summary_file = output_dir / "replay_summary.json"
    summary = {
        "n_probes": len(all_results),
        "probes": {},
    }
    for name, result in all_results.items():
        summary["probes"][name] = {
            "raw_convergence_layer": result.get("raw_convergence_layer"),
            "replay_convergence_layer": result.get("replay_convergence_layer"),
            "mean_convergence_delta": result.get("mean_convergence_delta"),
            "n_correct_items": result.get("n_correct_items"),
            "raw_computation_region": result.get("raw_computation_region"),
            "replay_computation_region": result.get("replay_computation_region"),
        }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_file}")
    print(f"All results saved to {output_dir}/")


if __name__ == "__main__":
    main()
