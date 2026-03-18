"""Layerwise function word convergence probe.

Tracks per-layer convergence speed for articles, conjunctions,
prepositions, auxiliaries, modals, and quantifiers. Directly
maps to potential inference savings — early convergence means
later layers aren't needed for syntactic processing.
"""
import math
from collections import defaultdict
from probes.convergence_logprob.probe import ConvergenceLogprobProbe, ITEMS, CHOICES
from probes.layerwise_registry import BaseLayerwiseProbe, register_layerwise_probe


@register_layerwise_probe
class ConvergenceLayerwiseProbe(BaseLayerwiseProbe):
    name = "convergence_layerwise"
    description = "Per-layer function word convergence — syntactic circuit timing"
    ITEMS = ITEMS
    CHOICES = CHOICES

    def _analyze(self, results):
        """Extend standard analysis with per-category convergence layers."""
        base = super()._analyze(results)

        if not results or not results[0].get("layers"):
            return base

        n_layers = len(results[0]["layers"])

        # Group items by function word category
        cat_items = defaultdict(list)
        for item_result in results:
            # Find original item to get category
            for orig in ITEMS:
                if orig["prompt"][:40] == item_result["prompt"][:40]:
                    cat_items[orig.get("category", "unknown")].append(item_result)
                    break

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
