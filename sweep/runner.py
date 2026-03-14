"""
Core (i,j) layer sweep runner.

For a model with N layers, config (i,j) executes:
  0..j-1, then i..N-1
  
This duplicates layers i..j-1 in the execution path.
The original model is config (0,0).

Reference: https://dnhkng.github.io/posts/rys/
"""

import json
import time
import itertools
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
    min_block_size: int = 1            # minimum j-i to test
    max_block_size: Optional[int] = None
    baseline_first: bool = True        # always run (0,0) baseline first
    save_interval: int = 10            # checkpoint every N configs


@dataclass  
class ConfigResult:
    i: int
    j: int
    n_duplicated: int
    probe_scores: dict[str, float]     # probe_name -> score
    probe_deltas: dict[str, float]     # probe_name -> delta vs baseline
    runtime_seconds: float


class SweepRunner:
    def __init__(self, config: SweepConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.n_layers = None
        self.baseline_scores = {}
        self.results: list[ConfigResult] = []
        
    def load_model(self):
        """Load model via ExLlamaV2 with layer path injection support."""
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
    
    def run_probes(self, layer_path: list[int]) -> dict[str, float]:
        """Run all configured probes on the given layer path."""
        from probes.registry import get_probe
        
        scores = {}
        self.model.set_layer_path(layer_path)
        
        for probe_name in self.config.probe_names:
            probe = get_probe(probe_name)
            scores[probe_name] = probe.run(self.model)
            
        return scores
    
    def compute_deltas(self, scores: dict[str, float]) -> dict[str, float]:
        return {
            name: scores[name] - self.baseline_scores.get(name, 0.0)
            for name in scores
        }
    
    def all_configs(self) -> list[tuple[int, int]]:
        """Generate all valid (i,j) pairs to sweep."""
        N = self.config.max_layers or self.n_layers
        configs = []
        
        for i in range(N):
            for j in range(i + self.config.min_block_size, N + 1):
                block_size = j - i
                if self.config.max_block_size and block_size > self.config.max_block_size:
                    continue
                configs.append((i, j))
                
        return configs
    
    def run(self):
        """Run the full sweep."""
        self.load_model()
        
        # Baseline (0,0)
        print("Running baseline (0,0)...")
        baseline_path = self.build_layer_path(0, 0)
        self.baseline_scores = self.run_probes(baseline_path)
        print(f"Baseline scores: {self.baseline_scores}")
        
        baseline_result = ConfigResult(
            i=0, j=0,
            n_duplicated=0,
            probe_scores=self.baseline_scores,
            probe_deltas={k: 0.0 for k in self.baseline_scores},
            runtime_seconds=0.0
        )
        self.results.append(baseline_result)
        self._checkpoint()
        
        # Sweep
        configs = self.all_configs()
        total = len(configs)
        print(f"Sweeping {total} configs ({self.n_layers} layers, "
              f"probes: {self.config.probe_names})")
        
        for idx, (i, j) in enumerate(configs):
            t0 = time.time()
            layer_path = self.build_layer_path(i, j)
            scores = self.run_probes(layer_path)
            deltas = self.compute_deltas(scores)
            elapsed = time.time() - t0
            
            result = ConfigResult(
                i=i, j=j,
                n_duplicated=j - i,
                probe_scores=scores,
                probe_deltas=deltas,
                runtime_seconds=elapsed
            )
            self.results.append(result)
            
            # Progress
            combined_delta = sum(deltas.values())
            print(f"[{idx+1}/{total}] ({i},{j}) +{j-i} layers | "
                  f"delta={combined_delta:+.4f} | {elapsed:.1f}s")
            
            if (idx + 1) % self.config.save_interval == 0:
                self._checkpoint()
        
        self._checkpoint()
        print(f"\nSweep complete. Results saved to {self.output_dir}")
        return self.results
    
    def _checkpoint(self):
        """Save current results to disk."""
        out = self.output_dir / "sweep_results.json"
        with open(out, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
            
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


if __name__ == "__main__":
    # Quick estimate for Qwen3-30B-A3B (48 layers)
    est = estimate_sweep_time(n_layers=48, seconds_per_config=90)
    print(f"Qwen3-30B-A3B sweep estimate: {est}")
    
    # And for a 7B test model (~28 layers)
    est_small = estimate_sweep_time(n_layers=28, seconds_per_config=30)
    print(f"7B test model sweep estimate: {est_small}")
