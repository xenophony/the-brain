#!/usr/bin/env python3
"""
Main sweep runner script.

Usage:
  # Local test on small model
  python scripts/run_sweep.py --model Qwen/Qwen2.5-7B-Instruct --probes math spatial --max-block 10

  # Sanity check vs blog reference  
  python scripts/run_sweep.py --model Qwen/Qwen3-30B-A3B --probes math eq

  # Full multi-domain sweep
  python scripts/run_sweep.py --model Qwen/Qwen3-30B-A3B --probes all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sweep.runner import SweepRunner, SweepConfig, estimate_sweep_time
from probes.registry import list_probes


ALL_PROBES = ["math", "eq", "code", "factual", "spatial", "language", "tool_use", "holistic",
              "planning", "instruction", "hallucination", "sycophancy", "consistency",
              "temporal", "metacognition", "counterfactual", "abstraction", "noise_robustness"]


def main():
    parser = argparse.ArgumentParser(description="LLM circuit sweep runner")
    
    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--probes", nargs="+", default=["math", "eq"],
                        help=f"Probes to run. Use 'all' for all probes. Available: {ALL_PROBES}")
    parser.add_argument("--output", default="results/latest", help="Output directory")
    parser.add_argument("--max-layers", type=int, default=None, 
                        help="Limit sweep to first N layers (for testing)")
    parser.add_argument("--max-block", type=int, default=None,
                        help="Maximum block size to test (speeds up sweep)")
    parser.add_argument("--min-block", type=int, default=1,
                        help="Minimum block size to test")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Just print time/cost estimate, don't run")
    parser.add_argument("--analyze-after", action="store_true", default=True,
                        help="Run heatmap analysis after sweep completes")
    parser.add_argument("--mock", action="store_true",
                        help="Use MockAdapter instead of ExLlamaV2 (for testing without GPU)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Timeout in seconds per config (default: 30)")
    parser.add_argument("--mode", choices=["duplicate", "skip", "both"], default="duplicate",
                        help="Sweep mode: duplicate layers, skip layers, or both (default: duplicate)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint (skip completed configs)")
    parser.add_argument("--baseline-repeats", type=int, default=3,
                        help="Number of baseline runs for variance estimation (default: 3)")

    args = parser.parse_args()
    
    # Resolve probe list
    probes = ALL_PROBES if "all" in args.probes else args.probes
    available = list_probes()
    missing = [p for p in probes if p not in available]
    if missing:
        print(f"Warning: probes not found: {missing}")
        probes = [p for p in probes if p in available]
    
    print(f"Model: {args.model}")
    print(f"Probes: {probes}")
    print(f"Output: {args.output}")
    
    # Quick estimate (needs layer count — approximate for now)
    # Will get exact count after model loads
    if args.estimate_only:
        # Approximate layer counts for common models
        layer_guesses = {
            "7b": 28, "7B": 28,
            "14b": 40, "14B": 40, 
            "30b": 48, "30B": 48,
            "32b": 64, "32B": 64,
            "72b": 80, "72B": 80,
        }
        n_layers = 48  # default guess
        for hint, count in layer_guesses.items():
            if hint in args.model:
                n_layers = count
                break
        
        if args.max_layers:
            n_layers = min(n_layers, args.max_layers)
            
        est = estimate_sweep_time(n_layers)
        print(f"\nEstimate for ~{n_layers} layers:")
        print(f"  Configs to test: {est['n_configs']}")
        print(f"  Estimated time: {est['estimated_hours']} hours")
        print(f"  Estimated cost (4090 @ $0.40/hr): ${est['estimated_cost_4090_usd']}")
        print(f"  Estimated cost (A100 @ $0.80/hr): ${est['estimated_cost_a100_usd']}")
        return
    
    # Resolve adapter class
    adapter_class = None
    if args.mock:
        from sweep.mock_adapter import MockAdapter
        adapter_class = MockAdapter

    print(f"Mode: {args.mode}")

    # Run sweep
    config = SweepConfig(
        model_path=args.model,
        output_dir=args.output,
        probe_names=probes,
        max_layers=args.max_layers,
        min_block_size=args.min_block,
        max_block_size=args.max_block,
        timeout_seconds=args.timeout,
        mode=args.mode,
        baseline_repeats=args.baseline_repeats,
        resume=args.resume,
    )

    runner = SweepRunner(config, adapter_class=adapter_class)
    results = runner.run()

    # Report best configs
    print("\n=== Best configurations ===")
    print(f"Overall best: {runner.best_config()}")
    for probe in probes:
        best = runner.best_config(probe=probe)
        print(f"Best for {probe}: ({best.i},{best.j}) delta={best.probe_deltas.get(probe, 0):.4f}")

    # Analysis
    if args.analyze_after:
        from analysis.heatmap import generate_all_plots, generate_overlay_analysis, safety_analysis
        results_path = Path(args.output) / "sweep_results.json"
        analysis_dir = Path(args.output) / "analysis"
        generate_all_plots(str(results_path), str(analysis_dir))

        # Overlay analysis when mode=both
        if args.mode == "both":
            dup_path = Path(args.output) / "sweep_results_duplicate.json"
            skip_path = Path(args.output) / "sweep_results_skip.json"
            if dup_path.exists() and skip_path.exists():
                generate_overlay_analysis(
                    str(dup_path), str(skip_path), str(analysis_dir)
                )

        # Safety analysis if safety probes are present
        safety_probes_present = {"hallucination", "sycophancy", "consistency", "instruction"}
        if len(safety_probes_present & set(probes)) >= 2:
            safety_analysis(str(results_path), str(analysis_dir))


if __name__ == "__main__":
    main()
