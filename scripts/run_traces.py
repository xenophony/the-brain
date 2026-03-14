#!/usr/bin/env python3
"""Run residual stream traces for mechanistic analysis."""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Residual stream tracer")
    parser.add_argument("--probe", type=str, help="Probe name to trace")
    parser.add_argument("--mode", choices=["domain", "safety"], default="domain")
    parser.add_argument("--all-domains", action="store_true", help="Trace all probes")
    parser.add_argument("--mock", action="store_true", help="Use MockAdapter")
    parser.add_argument("--model", type=str, help="Model path for real tracing")
    parser.add_argument("--n-questions", type=int, default=8)
    parser.add_argument("--output", default="results/traces")

    args = parser.parse_args()

    # Set up model
    if args.mock:
        from sweep.mock_adapter import MockAdapter
        model = MockAdapter(mode="perfect", seed=42)
    else:
        if not args.model:
            print("Error: --model required when not using --mock")
            return
        from sweep.exllama_adapter import ExLlamaV2LayerAdapter
        model = ExLlamaV2LayerAdapter(args.model)

    from analysis.residual_tracer import ResidualTracer
    tracer = ResidualTracer(model)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.all_domains:
        from probes.registry import list_probes
        probes_to_trace = list_probes()
        all_traces = {}

        for probe_name in probes_to_trace:
            print(f"Tracing {probe_name}...")
            try:
                domain_trace = tracer.trace_domain(probe_name, n_questions=args.n_questions)
                all_traces[probe_name] = domain_trace

                if domain_trace.mean_probabilities:
                    print(f"  Peak layer: {domain_trace.mean_peak_layer:.1f}")
                    print(f"  Computation region: {domain_trace.answer_computation_region}")
                else:
                    print(f"  No trace data (probe may not have HARD_ITEMS/QUESTIONS/SCENARIOS)")
            except Exception as e:
                print(f"  Error: {e}")

        # Save trace data
        trace_data = {}
        for name, dt in all_traces.items():
            trace_data[name] = {
                "n_questions": dt.n_questions,
                "mean_peak_layer": dt.mean_peak_layer,
                "answer_computation_region": list(dt.answer_computation_region),
                "suppression_regions": [list(r) for r in dt.suppression_regions],
                "mean_probabilities": dt.mean_probabilities,
            }

        with open(out / "all_domain_traces.json", "w") as f:
            json.dump(trace_data, f, indent=2)
        print(f"\nSaved traces to {out / 'all_domain_traces.json'}")

        # Generate report
        from analysis.trace_heatmap import generate_mechanistic_report
        generate_mechanistic_report(
            all_traces, "results/latest/sweep_results.json", str(out)
        )

    elif args.probe:
        if args.mode == "domain":
            domain_trace = tracer.trace_domain(args.probe, n_questions=args.n_questions)
            print(f"Probe: {args.probe}")
            print(f"Questions: {domain_trace.n_questions}")
            print(f"Peak layer: {domain_trace.mean_peak_layer:.1f}")
            print(f"Computation region: {domain_trace.answer_computation_region}")
            if domain_trace.suppression_regions:
                print(f"Suppression regions: {domain_trace.suppression_regions}")

            # Save
            with open(out / f"{args.probe}_domain_trace.json", "w") as f:
                json.dump({
                    "probe": args.probe,
                    "n_questions": domain_trace.n_questions,
                    "mean_peak_layer": domain_trace.mean_peak_layer,
                    "computation_region": list(domain_trace.answer_computation_region),
                    "mean_probabilities": domain_trace.mean_probabilities,
                }, f, indent=2)

        elif args.mode == "safety":
            print(f"Safety tracing for {args.probe} (requires real model data)")
            print("Use --mock for synthetic traces or provide --model for real model")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
