"""Layerwise function word convergence probe.

Tracks per-layer convergence speed for articles, conjunctions,
prepositions, auxiliaries, modals, and quantifiers.

Scores each item within its category's choices (not all 50+ words),
matching the logprob probe's scoring. This prevents cross-category
dilution of p(correct).
"""
import math
from collections import defaultdict
from probes.convergence_logprob.probe import ITEMS, CHOICES, CATEGORY_CHOICES
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class ConvergenceLayerwiseProbe(BaseLayerwiseProbe):
    name = "convergence_layerwise"
    description = "Per-layer function word convergence — syntactic circuit timing"
    ITEMS = ITEMS
    CHOICES = CHOICES
    max_items = None  # always run all — need all categories

    def run(self, model) -> dict:
        """Override run to score within category at each layer."""
        items = self._limit(self.ITEMS)
        psych_map = self._get_psych_token_map()
        results = []

        for item in items:
            prompt = item["prompt"]
            category = item.get("category", "unknown")
            cat_choices = CATEGORY_CHOICES.get(category, self.CHOICES)
            answer = item["answer"].lower()

            layer_data = model.get_layerwise_logprobs(
                prompt, self.CHOICES, psych_token_map=psych_map)

            layer_scores = []
            for layer_entry in layer_data["layer_logprobs"]:
                logprobs = layer_entry["target_logprobs"]

                # Normalize within category choices only
                probs = {}
                for c in cat_choices:
                    lp = logprobs.get(c, float('-inf'))
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

                layer_scores.append({
                    "layer_idx": layer_entry["layer_idx"],
                    "exec_pos": layer_entry["exec_pos"],
                    "p_correct": p_correct,
                    "argmax": max(probs, key=probs.get) if probs else "",
                    "entropy": entropy,
                    "psych": layer_entry.get("psych_scores", {}),
                })

            results.append({
                "prompt": item["prompt"][:100],
                "answer": answer,
                "difficulty": item.get("difficulty", "hard"),
                "category": category,
                "layers": layer_scores,
            })

        return self._analyze(results)

    def _analyze(self, results):
        """Extend standard analysis with per-category convergence layers."""
        base = super()._analyze(results)

        if not results or not results[0].get("layers"):
            return base

        n_layers = len(results[0]["layers"])

        # Group items by function word category
        cat_items = defaultdict(list)
        for item_result in results:
            cat = item_result.get("category", "unknown")
            cat_items[cat].append(item_result)

        # Per-category convergence analysis
        cat_convergence = {}
        for cat, items in sorted(cat_items.items()):
            cat_mean_p = [0.0] * n_layers
            conv_layers = []

            for item_result in items:
                layers = item_result["layers"]
                if len(layers) != n_layers:
                    continue
                for k, layer in enumerate(layers):
                    cat_mean_p[k] += layer["p_correct"]

                # Find convergence for this item
                for k in range(len(layers)):
                    if layers[k]["p_correct"] > 0.5:
                        stable = all(
                            layers[j]["p_correct"] > 0.4
                            for j in range(k, min(k + 3, len(layers)))
                        )
                        if stable:
                            conv_layers.append(k)
                            break

            n = len(items) or 1
            cat_mean_p = [v / n for v in cat_mean_p]
            mean_conv = sum(conv_layers) / len(conv_layers) if conv_layers else float(n_layers)

            cat_convergence[cat] = {
                "n_items": len(items),
                "mean_convergence_layer": round(mean_conv, 1),
                "n_converged": len(conv_layers),
                "mean_p_correct_by_layer": [round(v, 6) for v in cat_mean_p],
                "final_p_correct": round(cat_mean_p[-1], 4) if cat_mean_p else 0.0,
            }

        base["category_convergence"] = cat_convergence

        # Summary: which categories converge earliest?
        ranked = sorted(
            cat_convergence.items(),
            key=lambda x: x[1]["mean_convergence_layer"]
        )
        base["convergence_ranking"] = [
            {"category": cat, "layer": data["mean_convergence_layer"]}
            for cat, data in ranked
        ]

        return base
