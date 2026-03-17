#!/usr/bin/env python3
"""
Process sweep results: merge, analyze, archive.

Workflow:
  1. Drop sweep_results*.json files into results/incoming/
  2. Run: python scripts/process_results.py
  3. Merged data + analysis in results/analysis/
  4. Processed files moved to results/archive/ with timestamps

Handles:
  - Multiple files from split GPU runs (different probes, same configs)
  - Multiple files from different config ranges (same probes, different configs)
  - Overlapping configs (merges probe data, keeps latest)
  - Re-running safely (won't reprocess archived files)

Usage:
  python scripts/process_results.py                    # process incoming
  python scripts/process_results.py --dry-run          # show what would happen
  python scripts/process_results.py --no-archive       # don't move files after
  python scripts/process_results.py --reprocess        # also scan archive
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"
INCOMING_DIR = RESULTS_DIR / "incoming"
ARCHIVE_DIR = RESULTS_DIR / "archive"
ANALYSIS_DIR = RESULTS_DIR / "analysis"


def find_sweep_files(directory: Path) -> list[Path]:
    """Find all sweep result JSON files in a directory."""
    files = []
    for f in sorted(directory.glob("*.json")):
        if "sweep_results" in f.name or "sweep" in f.name:
            files.append(f)
    # Also check subdirectories (in case someone drops a whole output dir)
    for f in sorted(directory.rglob("sweep_results*.json")):
        if f not in files:
            files.append(f)
    return files


def load_and_validate(path: Path) -> list[dict]:
    """Load a sweep results file and validate structure."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected list, got {type(data).__name__}")
    if not data:
        raise ValueError(f"{path}: empty results list")
    # Check first entry has expected fields
    required = {"i", "j", "probe_scores", "probe_deltas"}
    missing = required - set(data[0].keys())
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    return data


def merge_results(all_files: list[Path]) -> tuple[list[dict], dict]:
    """Merge results from multiple files.

    Returns (merged_results, metadata).
    Handles overlapping (i,j,mode) by merging probe_scores dicts.
    """
    # Key: (i, j, mode) -> merged result dict
    merged = {}
    metadata = {
        "source_files": [],
        "total_configs": 0,
        "total_probes": set(),
        "merge_timestamp": datetime.now().isoformat(),
    }

    for path in all_files:
        try:
            data = load_and_validate(path)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  SKIP {path.name}: {e}")
            continue

        metadata["source_files"].append({
            "name": path.name,
            "path": str(path),
            "n_configs": len(data),
            "probes": list(data[0]["probe_scores"].keys()) if data else [],
        })

        for r in data:
            key = (r["i"], r["j"], r.get("mode", "duplicate"))
            if key not in merged:
                merged[key] = dict(r)
            else:
                # Merge probe_scores and probe_deltas from different runs
                merged[key]["probe_scores"].update(r["probe_scores"])
                merged[key]["probe_deltas"].update(r["probe_deltas"])
                # Keep longest runtime (conservative)
                if r.get("runtime_seconds", 0) > merged[key].get("runtime_seconds", 0):
                    merged[key]["runtime_seconds"] = r["runtime_seconds"]

        probes = set()
        for r in data:
            probes.update(k for k in r["probe_scores"]
                          if not k.startswith("_")
                          and not k.endswith("_easy")
                          and not k.endswith("_hard")
                          and not k.endswith("_pcorrect")
                          and not k.endswith("_pcorrect_easy")
                          and not k.endswith("_pcorrect_hard"))
        metadata["total_probes"].update(probes)

    # Convert to sorted list
    results = sorted(merged.values(), key=lambda r: (r["i"], r["j"]))
    metadata["total_configs"] = len(results)
    metadata["total_probes"] = sorted(metadata["total_probes"])

    return results, metadata


def run_analysis(results: list[dict], output_dir: Path) -> None:
    """Run the full analysis pipeline on merged results."""
    # Save merged results
    merged_file = output_dir / "sweep_results_merged.json"
    with open(merged_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Merged results: {merged_file} ({len(results)} configs)")

    # Run logprob analysis
    try:
        from scripts.analyze_logprob_sweep import analyze, generate_report
        analysis = analyze(results)

        targets_file = output_dir / "targeted_configs.json"
        with open(targets_file, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"  Targeted configs: {targets_file} ({analysis['n_targets']} targets)")

        report = generate_report(analysis, output_dir)
        report_file = output_dir / "CIRCUIT_FINDINGS.md"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"  Circuit report: {report_file}")
    except Exception as e:
        print(f"  Logprob analysis failed: {e}")

    # Run heatmap analysis
    try:
        from analysis.heatmap import generate_all_plots
        generate_all_plots(str(merged_file), str(output_dir))
        print(f"  Heatmaps generated in {output_dir}")
    except Exception as e:
        print(f"  Heatmap generation failed: {e}")

    # Run circuit boundary detection
    try:
        from analysis.heatmap import detect_circuit_boundaries
        boundaries = detect_circuit_boundaries(str(merged_file))
        bounds_file = output_dir / "circuit_boundaries.json"
        with open(bounds_file, "w") as f:
            json.dump(boundaries, f, indent=2)
        print(f"  Circuit boundaries: {bounds_file}")
    except Exception as e:
        print(f"  Circuit boundary detection failed: {e}")


def archive_files(files: list[Path]) -> None:
    """Move processed files to archive with timestamps."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    for f in files:
        dest = ARCHIVE_DIR / f"{timestamp}_{f.name}"
        shutil.move(str(f), str(dest))
        print(f"  Archived: {f.name} -> {dest.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Process sweep results: merge, analyze, archive")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without doing it")
    parser.add_argument("--no-archive", action="store_true",
                        help="Don't move files to archive after processing")
    parser.add_argument("--reprocess", action="store_true",
                        help="Also include files from archive/")
    args = parser.parse_args()

    # Ensure directories exist
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Find files
    incoming_files = find_sweep_files(INCOMING_DIR)
    archive_files_list = find_sweep_files(ARCHIVE_DIR) if args.reprocess else []

    all_files = incoming_files + archive_files_list

    if not all_files:
        print("No sweep result files found in results/incoming/")
        print("Drop sweep_results*.json files there and rerun.")
        return

    print(f"Found {len(all_files)} sweep result files:")
    for f in all_files:
        try:
            data = load_and_validate(f)
            probes = set()
            for r in data:
                probes.update(k for k in r["probe_scores"]
                              if not k.startswith("_") and "_easy" not in k
                              and "_hard" not in k and "_pcorrect" not in k)
            print(f"  {f.name}: {len(data)} configs, probes: {sorted(probes)}")
        except Exception as e:
            print(f"  {f.name}: ERROR - {e}")

    if args.dry_run:
        print("\n(dry run — no changes made)")
        return

    # Merge
    print(f"\nMerging...")
    results, metadata = merge_results(all_files)
    print(f"  {len(results)} unique configs, {len(metadata['total_probes'])} probes: "
          f"{metadata['total_probes']}")

    # Save metadata
    meta_file = ANALYSIS_DIR / "merge_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Analyze
    print(f"\nRunning analysis...")
    run_analysis(results, ANALYSIS_DIR)

    # Archive
    if not args.no_archive and incoming_files:
        print(f"\nArchiving {len(incoming_files)} incoming files...")
        archive_files(incoming_files)

    print(f"\nDone. Results in results/analysis/")


if __name__ == "__main__":
    main()
