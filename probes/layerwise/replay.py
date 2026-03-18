"""
Replay layerwise probe — runs layerwise analysis using the model's own reasoning.

Instead of bare "question + Answer with one word.", constructs a replay prompt
that includes the model's own <think>...</think> reasoning from a prior harvest.
Then measures per-layer logprobs at the point where the model would produce its
answer.

This separates:
  - "Where does the answer form from the question alone?" (raw layerwise)
  - "Where does the answer form given the model's own reasoning?" (replay)

The difference reveals how much each layer relies on the reasoning chain.

Usage:
    probe = ReplayLayerwiseProbe(
        harvest_file="results/harvested/causal_logprob_responses.json",
        probe_name="causal_logprob",
    )
    result = probe.run(model)
"""

import json
import math
from pathlib import Path

from probes.layerwise_registry import BaseLayerwiseProbe


class ReplayLayerwiseProbe(BaseLayerwiseProbe):
    """Runs layerwise analysis using the model's own reasoning as context.

    For each correctly-answered harvested item, constructs two prompts:
      1. RAW: bare question + " Answer with one word." (standard layerwise)
      2. REPLAY: full chat template with the model's own thinking as prefix

    Compares per-layer convergence between the two conditions.
    """
    capture_psych = False  # Replay analysis focuses on convergence, not psych

    def __init__(self, harvest_file: str = None, probe_name: str = None,
                 harvest_data: dict = None):
        """Initialize replay probe.

        Args:
            harvest_file: Path to harvested responses JSON file.
            probe_name: Name of the original probe (to look up CHOICES).
            harvest_data: Pre-loaded harvest dict (alternative to harvest_file).
        """
        self._harvest_file = harvest_file
        self._probe_name = probe_name or "unknown"
        self._harvest_data = harvest_data
        self._choices = None
        self.name = f"{self._probe_name}_replay"
        self.description = f"Replay layerwise analysis for {self._probe_name}"
        self.max_items = None  # Use all correct items by default

    def _load_harvest(self) -> dict:
        """Load harvest data from file or pre-loaded dict."""
        if self._harvest_data is not None:
            return self._harvest_data
        if self._harvest_file is None:
            raise ValueError("Either harvest_file or harvest_data must be provided")
        path = Path(self._harvest_file)
        if not path.exists():
            raise FileNotFoundError(f"Harvest file not found: {path}")
        with open(path) as f:
            return json.load(f)

    def _get_choices(self, harvest: dict) -> list[str]:
        """Get CHOICES for this probe, from harvest data or probe registry."""
        if self._choices is not None:
            return self._choices

        # Try getting from harvest items
        if harvest.get("items") and harvest["items"][0].get("choices"):
            self._choices = harvest["items"][0]["choices"]
            return self._choices

        # Fall back to probe registry
        try:
            from probes.registry import get_probe
            probe = get_probe(self._probe_name)
            self._choices = probe.CHOICES
            return self._choices
        except (KeyError, ImportError):
            pass

        # Last resort: infer from expected answers
        answers = set()
        for item in harvest.get("items", []):
            answers.add(item.get("expected", ""))
        self._choices = sorted(answers)
        return self._choices

    def run(self, model) -> dict:
        """Run both raw and replay layerwise analysis.

        Returns a dict with:
        - score: final-layer argmax accuracy (raw condition)
        - p_correct: final-layer mean p(correct) (raw condition)
        - raw_results: full layerwise data for raw condition
        - replay_results: full layerwise data for replay condition
        - comparison: per-item convergence comparison
        """
        harvest = self._load_harvest()
        choices = self._get_choices(harvest)

        # Filter to correct items only
        correct_items = [
            item for item in harvest.get("items", [])
            if item.get("correct", False) and item.get("thinking", "").strip()
        ]

        if self.max_items is not None and len(correct_items) > self.max_items:
            correct_items = correct_items[:self.max_items]

        if not correct_items:
            return {
                "score": 0.0,
                "p_correct": 0.0,
                "n_correct_items": 0,
                "n_total_items": len(harvest.get("items", [])),
                "raw_results": [],
                "replay_results": [],
                "comparison": {},
                "probe_name": self.name,
            }

        psych_map = self._get_psych_token_map()
        raw_results = []
        replay_results = []
        comparisons = []

        for item in correct_items:
            prompt_text = item["prompt"]
            expected = item["expected"].lower()
            thinking = item.get("thinking", "")

            # --- RAW condition: bare question ---
            raw_prompt = prompt_text + " Answer with one word."
            raw_data = model.get_layerwise_logprobs(
                raw_prompt, choices, psych_token_map=psych_map)

            # --- REPLAY condition: question + model's own thinking ---
            # Build replay prompt in chat format so the adapter won't re-wrap
            replay_prompt = (
                f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"<think>{thinking}</think>\n"
            )
            replay_data = model.get_layerwise_logprobs(
                replay_prompt, choices, psych_token_map=psych_map)

            # Process raw layers
            raw_layers = self._process_layers(raw_data, expected)
            replay_layers = self._process_layers(replay_data, expected)

            raw_results.append({
                "prompt": prompt_text[:100],
                "answer": expected,
                "difficulty": item.get("difficulty", "hard"),
                "layers": raw_layers,
            })

            replay_results.append({
                "prompt": prompt_text[:100],
                "answer": expected,
                "difficulty": item.get("difficulty", "hard"),
                "thinking_length": len(thinking),
                "layers": replay_layers,
            })

            # Per-item convergence comparison
            raw_conv = self._find_convergence_layer(raw_layers)
            replay_conv = self._find_convergence_layer(replay_layers)
            raw_final_p = raw_layers[-1]["p_correct"] if raw_layers else 0.0
            replay_final_p = replay_layers[-1]["p_correct"] if replay_layers else 0.0

            comparisons.append({
                "prompt": prompt_text[:80],
                "answer": expected,
                "raw_convergence_layer": raw_conv,
                "replay_convergence_layer": replay_conv,
                "convergence_delta": (replay_conv - raw_conv) if (
                    raw_conv is not None and replay_conv is not None) else None,
                "raw_final_p": round(raw_final_p, 4),
                "replay_final_p": round(replay_final_p, 4),
                "thinking_length": len(thinking),
            })

        # Aggregate analysis
        raw_analysis = self._analyze(raw_results)
        replay_analysis = self._analyze(replay_results)

        # Comparison summary
        conv_deltas = [c["convergence_delta"] for c in comparisons
                       if c["convergence_delta"] is not None]
        mean_conv_delta = (sum(conv_deltas) / len(conv_deltas)
                           if conv_deltas else None)

        return {
            "score": raw_analysis.get("score", 0.0),
            "p_correct": raw_analysis.get("p_correct", 0.0),
            "probe_name": self.name,
            "n_correct_items": len(correct_items),
            "n_total_items": len(harvest.get("items", [])),

            "raw_convergence_layer": raw_analysis.get(
                "mean_convergence_layer"),
            "replay_convergence_layer": replay_analysis.get(
                "mean_convergence_layer"),
            "mean_convergence_delta": (
                round(mean_conv_delta, 2) if mean_conv_delta is not None
                else None),

            "raw_computation_region": raw_analysis.get("computation_region"),
            "replay_computation_region": replay_analysis.get(
                "computation_region"),

            "raw_mean_p_correct_by_layer": raw_analysis.get(
                "mean_p_correct_by_layer", []),
            "replay_mean_p_correct_by_layer": replay_analysis.get(
                "mean_p_correct_by_layer", []),

            "comparison_items": comparisons,

            "raw_analysis": raw_analysis,
            "replay_analysis": replay_analysis,
        }

    def _process_layers(self, layer_data: dict, answer: str) -> list[dict]:
        """Convert raw layerwise logprob data to per-layer score dicts."""
        layers = []
        for entry in layer_data.get("layer_logprobs", []):
            logprobs = entry["target_logprobs"]
            probs = {}
            for c, lp in logprobs.items():
                if lp > -100:
                    probs[c] = math.exp(lp)
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}

            p_correct = probs.get(answer, 0.0)
            entropy = 0.0
            for p in probs.values():
                if p > 1e-10:
                    entropy -= p * math.log(p)

            layers.append({
                "layer_idx": entry["layer_idx"],
                "exec_pos": entry["exec_pos"],
                "p_correct": p_correct,
                "argmax": max(probs, key=probs.get) if probs else "",
                "entropy": entropy,
                "psych": entry.get("psych_scores", {}),
            })
        return layers

    def _find_convergence_layer(self, layers: list[dict]) -> int | None:
        """Find first layer where p_correct > 0.5 and stays stable."""
        for k in range(len(layers)):
            if layers[k]["p_correct"] > 0.5:
                stable = True
                for j in range(k, min(k + 3, len(layers))):
                    if layers[j]["p_correct"] < 0.4:
                        stable = False
                        break
                if stable:
                    return k
        return None
