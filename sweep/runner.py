"""
Core (i,j) layer sweep runner.

Supports two sweep modes:

  duplicate (default): config (i,j) executes 0..j-1, then i..N-1.
    Layers i..j-1 run twice. (0,0) = original model.

  skip: config (i,j) executes 0..i-1, then j..N-1.
    Layers i..j-1 are removed entirely. (0,0) = original model.

Reference: https://dnhkng.github.io/posts/rys/
"""

import json
import time
import itertools
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


@dataclass
class SweepConfig:
    model_path: str
    output_dir: str
    probe_names: list[str]
    max_layers: Optional[int] = None   # None = sweep all
    min_layer: int = 0                 # start sweep from this layer index
    min_block_size: int = 1            # minimum j-i to test
    max_block_size: Optional[int] = None
    baseline_first: bool = True        # always run (0,0) baseline first
    save_interval: int = 10            # checkpoint every N configs
    timeout_seconds: float = 30.0      # max seconds per config (0 = no timeout)
    mode: str = "duplicate"            # "duplicate", "skip", or "both"
    baseline_repeats: int = 3          # run baseline N times for variance estimation
    resume: bool = False               # resume from existing checkpoint
    full_items: bool = False             # if True, probe.max_items = None (run all items)
    prune: bool = True                  # adaptive catastrophic pruning
    prune_threshold: float = -1.0       # delta threshold for catastrophic detection
    prune_consecutive: int = 2          # consecutive catastrophic configs before pruning
    config_list: Optional[list[tuple[int, int]]] = None  # explicit configs to run (overrides all_configs)


@dataclass
class ConfigResult:
    i: int
    j: int
    n_duplicated: int
    probe_scores: dict[str, float]     # probe_name -> score
    probe_deltas: dict[str, float]     # probe_name -> delta vs baseline
    runtime_seconds: float
    mode: str = "duplicate"            # "duplicate" or "skip"


class _TimeoutError(Exception):
    """Raised when a config run exceeds its timeout."""
    pass


class SweepRunner:
    def __init__(self, config: SweepConfig, adapter_class=None):
        """
        Initialize sweep runner.

        Args:
            config: SweepConfig with sweep parameters.
            adapter_class: Optional model adapter class. If None, uses
                ExLlamaV2LayerAdapter. Pass MockAdapter for testing.
        """
        self.config = config
        self.adapter_class = adapter_class
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.n_layers = None
        self.baseline_scores = {}
        self.baseline_std = {}              # per-probe baseline standard deviation
        self.results: list[ConfigResult] = []
        self._completed_configs: set[tuple[int, int, str]] = set()  # (i, j, mode)

    def load_model(self):
        """Load model via the configured adapter class."""
        if self.adapter_class is not None:
            self.model = self.adapter_class(self.config.model_path)
        else:
            from sweep.exllama_adapter import ExLlamaV2LayerAdapter
            self.model = ExLlamaV2LayerAdapter(self.config.model_path)
        self.n_layers = self.model.num_layers
        print(f"Loaded model with {self.n_layers} layers")
        
    def build_layer_path(self, i: int, j: int) -> list[int]:
        """
        Build execution path for config (i,j).
        
        Layers 0..j-1 run first, then i..N-1.
        Layers i..j-1 are duplicated.
        
        (0,0) returns the original unmodified path [0,1,...,N-1].
        """
        N = self.n_layers
        if i == 0 and j == 0:
            return list(range(N))
        
        assert 0 <= i < j <= N, f"Invalid config: i={i}, j={j}, N={N}"
        
        first_pass = list(range(j))       # 0..j-1
        second_pass = list(range(i, N))   # i..N-1
        return first_pass + second_pass
    
    def build_skip_path(self, i: int, j: int) -> list[int]:
        """
        Build execution path for skip config (i,j).

        Layers 0..i-1 run first, then j..N-1.
        Layers i..j-1 are skipped entirely.

        (0,0) returns the original unmodified path [0,1,...,N-1].
        """
        N = self.n_layers
        if i == 0 and j == 0:
            return list(range(N))

        assert 0 <= i < j <= N, f"Invalid skip config: i={i}, j={j}, N={N}"

        before = list(range(i))         # 0..i-1
        after = list(range(j, N))       # j..N-1
        return before + after

    def run_probes(self, layer_path: list[int]) -> dict[str, float]:
        """Run all configured probes on the given layer path.

        Logprob probes are cross-batched: all prompts from all logprob
        probes are collected, run in one mega-batch forward pass, then
        results are distributed back for per-probe scoring. This
        amortizes the 48-layer forward pass across all logprob items.

        Generation probes run sequentially after the logprob batch.
        """
        from probes.registry import get_probe, BaseLogprobProbe

        scores = {}
        self.model.set_layer_path(layer_path)

        timeout = self.config.timeout_seconds
        is_gpu = hasattr(self.model, '_model') and self.model._model is not None

        # Separate logprob probes (batchable) from generation probes
        logprob_probes = []  # (name, probe, prompts, targets, items)
        gen_probe_names = []

        for probe_name in self.config.probe_names:
            probe = get_probe(probe_name)
            if self.config.full_items:
                probe.max_items = None
            probe.log_responses = True

            if isinstance(probe, BaseLogprobProbe) and hasattr(probe, 'get_prompts_and_targets'):
                prompts, targets, items = probe.get_prompts_and_targets()
                logprob_probes.append((probe_name, probe, prompts, targets, items))
            else:
                gen_probe_names.append(probe_name)

        # --- Cross-probe logprob batch ---
        if logprob_probes and hasattr(self.model, 'get_logprobs_batch'):
            # Collect all prompts and track which belong to which probe.
            # Group by target_tokens (probes with same choices can share a batch).
            target_groups = {}  # frozenset(targets) -> [(name, probe, prompts, items)]
            for name, probe, prompts, targets, items in logprob_probes:
                key = tuple(sorted(targets))
                if key not in target_groups:
                    target_groups[key] = {"targets": targets, "entries": []}
                target_groups[key]["entries"].append((name, probe, prompts, items))

            for key, group in target_groups.items():
                targets = group["targets"]
                all_prompts = []
                slices = []  # (name, probe, items, start, end)
                for name, probe, prompts, items in group["entries"]:
                    start = len(all_prompts)
                    all_prompts.extend(prompts)
                    end = len(all_prompts)
                    slices.append((name, probe, items, start, end))

                # One mega-batch forward pass for this target group
                all_logprobs = self.model.get_logprobs_batch(all_prompts, targets)

                # Distribute results back to each probe for scoring
                for name, probe, items, start, end in slices:
                    probe_logprobs = all_logprobs[start:end]
                    result = probe.score_logprobs(items, probe_logprobs)
                    self._store_result(scores, name, result)

        elif logprob_probes:
            # Fallback: no batch support, run individually
            for name, probe, prompts, targets, items in logprob_probes:
                result = probe.run(self.model)
                self._store_result(scores, name, result)

        # --- Generation probes (sequential) ---
        for probe_name in gen_probe_names:
            probe = get_probe(probe_name)
            if self.config.full_items:
                probe.max_items = None
            probe.log_responses = True
            if is_gpu:
                try:
                    result = probe.run(self.model)
                except Exception as e:
                    print(f"  Probe {probe_name} error: {e}")
                    result = 0.0
            elif timeout and timeout > 0:
                result = self._run_probe_with_timeout(probe, timeout)
            else:
                try:
                    result = probe.run(self.model)
                except Exception as e:
                    print(f"  Probe {probe_name} error: {e}")
                    result = 0.0
            self._store_result(scores, probe_name, result)

            if hasattr(self.model, '_think_close_total'):
                self.model._think_close_total = 0

        return scores

    def _store_result(self, scores: dict, probe_name: str, result) -> None:
        """Extract scores from a probe result into the scores dict."""
        if isinstance(result, dict):
            scores[probe_name] = result["score"]
            if "easy_score" in result:
                scores[f"{probe_name}_easy"] = result["easy_score"]
            if "hard_score" in result:
                scores[f"{probe_name}_hard"] = result["hard_score"]
            if "p_correct" in result:
                scores[f"{probe_name}_pcorrect"] = result["p_correct"]
            if "p_correct_easy" in result:
                scores[f"{probe_name}_pcorrect_easy"] = result["p_correct_easy"]
            if "p_correct_hard" in result:
                scores[f"{probe_name}_pcorrect_hard"] = result["p_correct_hard"]
            if "item_results" in result:
                failures = [r for r in result["item_results"]
                            if r.get("score", 1.0) == 0.0]
                if failures:
                    scores[f"_failures_{probe_name}"] = failures
        else:
            scores[probe_name] = result

    def _run_probe_with_timeout(self, probe, timeout: float) -> "float | dict":
        """Run a single probe with a timeout. Returns 0.0 on timeout or error."""
        result = [0.0]
        error = [None]

        def target():
            try:
                result[0] = probe.run(self.model)
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            print(f"  Probe {probe.name} timed out after {timeout}s — returning 0.0")
            return 0.0
        if error[0] is not None:
            print(f"  Probe {probe.name} error: {error[0]} — returning 0.0")
            return 0.0
        return result[0]
    
    def compute_deltas(self, scores: dict[str, float]) -> dict[str, float]:
        return {
            name: scores[name] - self.baseline_scores.get(name, 0.0)
            for name in scores
            if isinstance(scores[name], (int, float))
        }
    
    def all_configs(self) -> list[tuple[int, int]]:
        """Generate all valid (i,j) pairs to sweep."""
        if self.config.config_list is not None:
            return self.config.config_list

        N = self.config.max_layers or self.n_layers
        min_i = self.config.min_layer
        configs = []

        for i in range(min_i, N):
            for j in range(i + self.config.min_block_size, N + 1):
                block_size = j - i
                if self.config.max_block_size and block_size > self.config.max_block_size:
                    continue
                configs.append((i, j))

        return configs
    
    def run(self):
        """Run the full sweep. Supports duplicate, skip, or both modes."""
        self.load_model()

        # Resume from checkpoint if requested
        if self.config.resume:
            self._load_checkpoint()

        mode = self.config.mode
        if mode == "both":
            # Run duplicate sweep first, then skip sweep
            print("=== Mode: both — running duplicate sweep then skip sweep ===\n")
            self._run_single_sweep("duplicate")
            dup_results = list(self.results)

            # Reset for skip sweep
            self.results = []
            self._run_single_sweep("skip")
            skip_results = list(self.results)

            # Save separately
            self._save_results(dup_results, "sweep_results_duplicate.json")
            self._save_results(skip_results, "sweep_results_skip.json")

            # Combined for legacy compatibility
            self.results = dup_results + [r for r in skip_results if not (r.i == 0 and r.j == 0)]
            self._checkpoint()
            print(f"\nBoth sweeps complete. Results saved to {self.output_dir}")
        else:
            self._run_single_sweep(mode)

        return self.results

    def _run_single_sweep(self, mode: str):
        """Run one sweep pass (duplicate or skip)."""
        path_builder = self.build_layer_path if mode == "duplicate" else self.build_skip_path
        label = "duplicate" if mode == "duplicate" else "skip"
        block_label = "duplicated" if mode == "duplicate" else "skipped"

        # Check if baseline already completed (resume case)
        if (0, 0, mode) not in self._completed_configs:
            # Baseline with variance estimation
            n_repeats = max(1, self.config.baseline_repeats)
            print(f"Running {label} baseline (0,0) x{n_repeats}...")
            baseline_path = self.build_layer_path(0, 0)

            all_scores = []
            for rep in range(n_repeats):
                scores = self.run_probes(baseline_path)
                all_scores.append(scores)

            # Mean baseline — include difficulty sub-scores if present
            self.baseline_scores = {}
            self.baseline_std = {}
            all_keys = set()
            for s in all_scores:
                all_keys.update(s.keys())
            for key in all_keys:
                if key.startswith("_"):  # skip _failures_ and other metadata
                    continue
                values = [s.get(key, 0.0) for s in all_scores]
                # Ensure all values are numeric (skip if any are dicts/lists)
                if not all(isinstance(v, (int, float)) for v in values):
                    continue
                self.baseline_scores[key] = float(np.mean(values))
                self.baseline_std[key] = float(np.std(values)) if n_repeats > 1 else 0.0

            print(f"Baseline scores: {self.baseline_scores}")
            if n_repeats > 1:
                print(f"Baseline std:    {self.baseline_std}")

            baseline_result = ConfigResult(
                i=0, j=0,
                n_duplicated=0,
                probe_scores=self.baseline_scores,
                probe_deltas={k: 0.0 for k in self.baseline_scores},
                runtime_seconds=0.0,
                mode=mode,
            )
            self.results.append(baseline_result)
            self._completed_configs.add((0, 0, mode))
            self._checkpoint()
        else:
            print(f"Baseline already completed (resumed), skipping")

        # Sweep
        configs = self.all_configs()
        remaining = [(i, j) for i, j in configs if (i, j, mode) not in self._completed_configs]
        total = len(configs)
        skipped = total - len(remaining)
        if skipped > 0:
            print(f"Resuming: {skipped}/{total} configs already done, {len(remaining)} remaining")
        print(f"Sweeping {len(remaining)} {label} configs ({self.n_layers} layers, "
              f"probes: {self.config.probe_names})")

        # Adaptive catastrophic pruning state: track consecutive catastrophic
        # deltas per i value. Once triggered, run one more config (for research)
        # then skip remaining j values for that i.
        # Key: i -> {"consecutive": int, "pruned": bool, "capture_one": bool}
        prune_state: dict[int, dict] = {}
        pruned_count = 0

        for idx, (i, j) in enumerate(remaining):
            # Check if this i has been pruned
            ps = prune_state.get(i)
            if ps and ps.get("pruned"):
                # Already captured the one extra — skip entirely
                pruned_count += 1
                self._completed_configs.add((i, j, mode))
                continue

            t0 = time.time()
            layer_path = path_builder(i, j)
            scores = self.run_probes(layer_path)
            deltas = self.compute_deltas(scores)
            elapsed = time.time() - t0

            n_affected = j - i
            result = ConfigResult(
                i=i, j=j,
                n_duplicated=n_affected,
                probe_scores=scores,
                probe_deltas=deltas,
                runtime_seconds=elapsed,
                mode=mode,
            )
            self.results.append(result)
            self._completed_configs.add((i, j, mode))

            # Progress
            combined_delta = sum(
                v for k, v in deltas.items()
                if not k.startswith("_") and not k.endswith("_easy") and not k.endswith("_hard")
            )
            sign = "+" if mode == "duplicate" else "-"
            print(f"[{skipped+idx+1-pruned_count}/{total}] ({i},{j}) {sign}{n_affected} layers {block_label} | "
                  f"delta={combined_delta:+.4f} | {elapsed:.1f}s")

            # Adaptive catastrophic pruning
            if self.config.prune:
                if i not in prune_state:
                    prune_state[i] = {"consecutive": 0, "pruned": False, "capture_one": False}
                ps = prune_state[i]

                if combined_delta < self.config.prune_threshold:
                    ps["consecutive"] += 1
                else:
                    ps["consecutive"] = 0

                if ps["consecutive"] >= self.config.prune_consecutive:
                    if ps.get("capture_one"):
                        # We already captured the extra one — now prune
                        ps["pruned"] = True
                        print(f"  PRUNED: i={i} catastrophic from j={j} onward "
                              f"({ps['consecutive']} consecutive delta < {self.config.prune_threshold})")
                    else:
                        # Mark to capture one more, then prune
                        ps["capture_one"] = True

            if (idx + 1) % self.config.save_interval == 0:
                self._checkpoint()

        if pruned_count > 0:
            print(f"\nAdaptive pruning skipped {pruned_count} configs "
                  f"(~{100*pruned_count//(len(remaining) or 1)}% savings)")

        self._checkpoint()
        print(f"\n{label.capitalize()} sweep complete. Results saved to {self.output_dir}")
    
    def _checkpoint(self):
        """Save current results to disk atomically (write tmp then rename)."""
        out = self.output_dir / "sweep_results.json"
        tmp = self.output_dir / "sweep_results.json.tmp"
        with open(tmp, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        tmp.replace(out)

    def _save_results(self, results: list[ConfigResult], filename: str):
        """Save a specific result set to disk atomically."""
        out = self.output_dir / filename
        tmp = self.output_dir / (filename + ".tmp")
        with open(tmp, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        tmp.replace(out)

    def _load_checkpoint(self) -> bool:
        """
        Load existing checkpoint and populate completed configs set.
        Returns True if checkpoint was loaded.
        """
        out = self.output_dir / "sweep_results.json"
        if not out.exists():
            return False
        try:
            with open(out) as f:
                data = json.load(f)
            for r in data:
                mode = r.get("mode", "duplicate")
                result = ConfigResult(
                    i=r["i"], j=r["j"],
                    n_duplicated=r["n_duplicated"],
                    probe_scores=r["probe_scores"],
                    probe_deltas=r["probe_deltas"],
                    runtime_seconds=r["runtime_seconds"],
                    mode=mode,
                )
                self.results.append(result)
                self._completed_configs.add((r["i"], r["j"], mode))

                # Restore baseline scores from (0,0) entry
                if r["i"] == 0 and r["j"] == 0:
                    self.baseline_scores = r["probe_scores"]

            print(f"Resumed from checkpoint: {len(self.results)} configs already completed")
            return True
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: corrupt checkpoint, starting fresh: {e}")
            self.results = []
            self._completed_configs = set()
            return False

    def best_config(self, probe: Optional[str] = None) -> ConfigResult:
        """Return config with highest combined (or single probe) delta."""
        def score(r):
            if probe:
                return r.probe_deltas.get(probe, 0.0)
            return sum(r.probe_deltas.values())
        
        return max(self.results, key=score)


def estimate_sweep_time(n_layers: int, seconds_per_config: float = 120) -> dict:
    """Estimate total sweep time and cost."""
    n_configs = (n_layers * (n_layers + 1)) // 2 + 1
    total_seconds = n_configs * seconds_per_config
    total_hours = total_seconds / 3600
    
    return {
        "n_configs": n_configs,
        "estimated_hours": round(total_hours, 1),
        "estimated_cost_4090_usd": round(total_hours * 0.40, 2),  # ~$0.40/hr Vast.ai
        "estimated_cost_a100_usd": round(total_hours * 0.80, 2),
    }


def build_optimized_path(
    n_layers: int,
    skip_regions: list[tuple[int, int]],
    duplicate_regions: list[tuple[int, int]],
) -> list[int]:
    """
    Build an optimized layer execution path combining skip and duplicate regions.

    Args:
        n_layers: Total number of layers in the model.
        skip_regions: List of (i, j) tuples — layers i..j-1 are removed.
        duplicate_regions: List of (i, j) tuples — layers i..j-1 are duplicated.

    Returns:
        Ordered list of layer indices representing the execution path.

    Skip regions take priority over duplicate regions if they overlap.
    Regions are processed from the base path [0..N-1]:
      1. Remove all skipped layers.
      2. Insert duplicated layers (only those not skipped).
    """
    # Build sets
    skipped = set()
    for i, j in skip_regions:
        for k in range(i, j):
            skipped.add(k)

    duplicated = set()
    for i, j in duplicate_regions:
        for k in range(i, j):
            if k not in skipped:
                duplicated.add(k)

    # Build path: for each layer 0..N-1, include it if not skipped,
    # and if duplicated, include it twice (second pass comes after first full pass)
    first_pass = [k for k in range(n_layers) if k not in skipped]

    # For duplicated layers, insert them again after the first pass reaches j
    # Group duplicate regions for ordered insertion
    dup_insertions = []
    for i, j in duplicate_regions:
        dup_layers = [k for k in range(i, j) if k not in skipped]
        if dup_layers:
            dup_insertions.append((i, dup_layers))

    if not dup_insertions:
        return first_pass

    # Sort by insertion point
    dup_insertions.sort(key=lambda x: x[0])

    # Build the path with duplications inserted at the right positions
    # Strategy: run layers 0..j-1 (minus skipped), then dup layers i..j-1,
    # then continue j..N-1 (minus skipped)
    # For multiple dup regions, process from right to left to keep indices stable
    path = list(first_pass)
    for start_i, dup_layers in reversed(dup_insertions):
        # Find where to insert the duplicate: right after the last dup layer
        # appears in the path
        last_dup = dup_layers[-1]
        insert_pos = None
        for idx, layer in enumerate(path):
            if layer == last_dup:
                insert_pos = idx + 1
                break
        if insert_pos is not None:
            for offset, layer in enumerate(dup_layers):
                path.insert(insert_pos + offset, layer)

    return path


if __name__ == "__main__":
    # Quick estimate for Qwen3-30B-A3B (48 layers)
    est = estimate_sweep_time(n_layers=48, seconds_per_config=90)
    print(f"Qwen3-30B-A3B sweep estimate: {est}")
    
    # And for a 7B test model (~28 layers)
    est_small = estimate_sweep_time(n_layers=28, seconds_per_config=30)
    print(f"7B test model sweep estimate: {est_small}")
