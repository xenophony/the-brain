"""
Layerwise probe registry — BaseLayerwiseProbe and registration.

Probes that capture per-layer answer probabilities AND psycholinguistic
signals at EVERY layer, enabling "fMRI-style" functional imaging.

Reuses ITEMS and CHOICES from existing logprob probes. Outputs rich
per-layer data instead of single scores.
"""

import math
from probes.registry import BaseProbe

_LAYERWISE_REGISTRY: dict[str, "BaseLayerwiseProbe"] = {}


class BaseLayerwiseProbe(BaseProbe):
    """Probe that captures per-layer answer convergence and psych signals.

    Reuses ITEMS and CHOICES from existing logprob probes.
    Outputs rich per-layer data instead of single scores.
    """
    ITEMS: list[dict] = []
    CHOICES: list[str] = []
    capture_psych: bool = True
    _psych_token_map: dict | None = None

    def _get_psych_token_map(self) -> dict | None:
        """Build psych token map for layerwise scoring.

        Returns a mapping from psych category -> list of word strings,
        used to look up probabilities in layerwise logprob dicts.
        """
        if not self.capture_psych:
            return None
        if self._psych_token_map is not None:
            return self._psych_token_map
        try:
            from probes.psycholinguistics import PSYCH_VOCAB
            # For layerwise analysis, we pass the raw vocab map.
            # The adapter's get_layerwise_logprobs handles tokenization.
            self._psych_token_map = PSYCH_VOCAB
            return self._psych_token_map
        except ImportError:
            return None

    def run(self, model) -> dict:
        """Run per-layer analysis on all items.

        Returns a dict with:
        - score: final-layer argmax accuracy (for compat)
        - p_correct: final-layer mean p(correct)
        - layerwise_results: full per-layer data for analysis
        """
        items = self._limit(self.ITEMS)
        psych_map = self._get_psych_token_map()
        results = []

        for item in items:
            prompt = item["prompt"] + " Answer with one word."
            layer_data = model.get_layerwise_logprobs(
                prompt, self.CHOICES, psych_token_map=psych_map)

            answer = item["answer"].lower()
            layer_scores = []
            for layer_entry in layer_data["layer_logprobs"]:
                logprobs = layer_entry["target_logprobs"]
                # Compute p(correct) at this layer
                probs = {}
                for c, lp in logprobs.items():
                    if lp > -100:
                        probs[c] = math.exp(lp)
                total = sum(probs.values())
                if total > 0:
                    probs = {k: v / total for k, v in probs.items()}
                p_correct = probs.get(answer, 0.0)

                # Entropy of choice distribution
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
                "layers": layer_scores,
            })

        return self._analyze(results)

    def _analyze(self, results: list[dict]) -> dict:
        """Compute convergence metrics, psych correlations, etc.

        Returns analysis dict with per-layer aggregates and per-item details.
        """
        if not results:
            return {"score": 0.0, "p_correct": 0.0, "layerwise_results": []}

        n_items = len(results)
        n_layers = len(results[0]["layers"]) if results[0]["layers"] else 0

        if n_layers == 0:
            return {"score": 0.0, "p_correct": 0.0, "layerwise_results": results}

        # Per-layer aggregates
        mean_p_correct = [0.0] * n_layers
        sum_sq_p_correct = [0.0] * n_layers
        mean_entropy = [0.0] * n_layers

        # Convergence tracking
        convergence_layers = []

        # Correct vs incorrect split
        correct_items = []
        incorrect_items = []

        for item_result in results:
            layers = item_result["layers"]
            if len(layers) != n_layers:
                continue

            for k, layer in enumerate(layers):
                mean_p_correct[k] += layer["p_correct"]
                sum_sq_p_correct[k] += layer["p_correct"] ** 2
                mean_entropy[k] += layer["entropy"]

            # Final layer argmax correctness
            final_layer = layers[-1]
            is_correct = final_layer["argmax"] == item_result["answer"]
            if is_correct:
                correct_items.append(item_result)
            else:
                incorrect_items.append(item_result)

            # Convergence: first layer where p_correct > 0.5 and stays > 0.4
            # for at least 3 consecutive layers
            conv_layer = None
            for k in range(len(layers)):
                if layers[k]["p_correct"] > 0.5:
                    # Check stability
                    stable = True
                    for j in range(k, min(k + 3, len(layers))):
                        if layers[j]["p_correct"] < 0.4:
                            stable = False
                            break
                    if stable:
                        conv_layer = k
                        break
            if conv_layer is not None:
                convergence_layers.append(conv_layer)

        # Normalize means
        for k in range(n_layers):
            mean_p_correct[k] /= n_items
            mean_entropy[k] /= n_items

        # Std dev
        std_p_correct = []
        for k in range(n_layers):
            variance = sum_sq_p_correct[k] / n_items - mean_p_correct[k] ** 2
            std_p_correct.append(max(0.0, variance) ** 0.5)

        # Computation region: steepest rise in mean p_correct
        computation_region = self._find_computation_region(mean_p_correct)

        # Convergence stats
        mean_conv = (sum(convergence_layers) / len(convergence_layers)
                     if convergence_layers else float(n_layers))
        conv_std = 0.0
        if len(convergence_layers) > 1:
            m = mean_conv
            conv_std = (sum((c - m) ** 2 for c in convergence_layers)
                        / len(convergence_layers)) ** 0.5

        # Psych by layer
        psych_by_layer = self._aggregate_psych(results, n_layers)

        # Correct vs incorrect psych comparison
        correct_vs_incorrect = self._compare_correct_incorrect(
            correct_items, incorrect_items, n_layers)

        # Surprise detection: layers where p_correct changes by > 0.15
        surprise_layers = []
        for k in range(1, n_layers):
            delta = abs(mean_p_correct[k] - mean_p_correct[k - 1])
            if delta > 0.15:
                surprise_layers.append({
                    "layer": k,
                    "delta": round(mean_p_correct[k] - mean_p_correct[k - 1], 4),
                    "direction": "rise" if mean_p_correct[k] > mean_p_correct[k - 1] else "drop",
                })

        # Final layer score (compat with existing system)
        final_argmax_correct = sum(
            1.0 for r in results
            if r["layers"][-1]["argmax"] == r["answer"]
        ) / n_items
        final_p_correct = mean_p_correct[-1]

        return {
            "score": final_argmax_correct,
            "p_correct": final_p_correct,

            "probe_name": getattr(self, "name", "unknown"),
            "n_items": n_items,
            "n_layers": n_layers,

            "mean_p_correct_by_layer": [round(v, 6) for v in mean_p_correct],
            "std_p_correct_by_layer": [round(v, 6) for v in std_p_correct],
            "mean_entropy_by_layer": [round(v, 6) for v in mean_entropy],

            "mean_convergence_layer": round(mean_conv, 2),
            "convergence_layer_std": round(conv_std, 2),
            "n_converged": len(convergence_layers),
            "computation_region": computation_region,

            "psych_by_layer": psych_by_layer,
            "correct_vs_incorrect": correct_vs_incorrect,
            "surprise_layers": surprise_layers,

            "items": results,
        }

    def _find_computation_region(self, mean_p: list[float]) -> tuple:
        """Find the region of steepest rise in mean p_correct."""
        if len(mean_p) < 2:
            return (0, 0)

        diffs = [mean_p[i] - mean_p[i - 1] for i in range(1, len(mean_p))]
        best_start = 0
        best_end = 0
        best_sum = 0.0
        cur_start = 0
        cur_sum = 0.0

        for i, d in enumerate(diffs):
            if d > 0:
                if cur_sum <= 0:
                    cur_start = i
                    cur_sum = d
                else:
                    cur_sum += d
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_start = cur_start
                    best_end = i + 1
            else:
                cur_sum = 0.0

        return (best_start, best_end) if best_sum > 0 else (0, 0)

    def _aggregate_psych(self, results: list[dict], n_layers: int) -> dict:
        """Aggregate psych scores by layer across all items."""
        if not results or not results[0]["layers"]:
            return {}

        # Discover psych categories from first item's first layer
        first_psych = results[0]["layers"][0].get("psych", {})
        if not first_psych:
            return {}

        categories = list(first_psych.keys())
        psych_sums = {cat: [0.0] * n_layers for cat in categories}
        count = 0

        for item_result in results:
            layers = item_result["layers"]
            if len(layers) != n_layers:
                continue
            count += 1
            for k, layer in enumerate(layers):
                psych = layer.get("psych", {})
                for cat in categories:
                    psych_sums[cat][k] += psych.get(cat, 0.0)

        if count == 0:
            return {}

        return {
            cat: [round(v / count, 6) for v in values]
            for cat, values in psych_sums.items()
        }

    def _compare_correct_incorrect(self, correct_items: list, incorrect_items: list,
                                   n_layers: int) -> dict:
        """Compare psych profiles between correctly and incorrectly answered items."""
        if not correct_items or not incorrect_items:
            return {}

        # Get psych categories
        first_psych = correct_items[0]["layers"][0].get("psych", {})
        if not first_psych:
            return {}

        categories = list(first_psych.keys())
        result = {}

        for cat in categories:
            correct_by_layer = [0.0] * n_layers
            incorrect_by_layer = [0.0] * n_layers

            for item in correct_items:
                for k, layer in enumerate(item["layers"]):
                    if k < n_layers:
                        correct_by_layer[k] += layer.get("psych", {}).get(cat, 0.0)

            for item in incorrect_items:
                for k, layer in enumerate(item["layers"]):
                    if k < n_layers:
                        incorrect_by_layer[k] += layer.get("psych", {}).get(cat, 0.0)

            nc = len(correct_items) or 1
            ni = len(incorrect_items) or 1
            correct_mean = [round(v / nc, 6) for v in correct_by_layer]
            incorrect_mean = [round(v / ni, 6) for v in incorrect_by_layer]

            # Find maximum divergence layer
            max_div = 0.0
            div_layer = 0
            for k in range(n_layers):
                d = abs(correct_mean[k] - incorrect_mean[k])
                if d > max_div:
                    max_div = d
                    div_layer = k

            result[cat] = {
                "correct_mean_by_layer": correct_mean,
                "incorrect_mean_by_layer": incorrect_mean,
                "divergence_layer": div_layer,
                "max_divergence": round(max_div, 6),
            }

        return result


def register_layerwise_probe(cls):
    """Class decorator — registers a layerwise probe by its .name attribute."""
    instance = cls()
    _LAYERWISE_REGISTRY[instance.name] = instance
    return cls


def get_layerwise_probe(name: str) -> BaseLayerwiseProbe:
    """Look up a registered layerwise probe by name."""
    if name not in _LAYERWISE_REGISTRY:
        # Try auto-importing
        import importlib
        try:
            importlib.import_module(f"probes.layerwise.{name.replace('_layerwise', '')}")
        except ImportError:
            pass
    if name not in _LAYERWISE_REGISTRY:
        raise KeyError(f"Layerwise probe '{name}' not registered. "
                       f"Available: {list(_LAYERWISE_REGISTRY.keys())}")
    return _LAYERWISE_REGISTRY[name]


def list_layerwise_probes() -> list[str]:
    """Return names of all registered layerwise probes."""
    # Trigger auto-discovery
    import importlib
    try:
        importlib.import_module("probes.layerwise")
    except ImportError:
        pass
    return list(_LAYERWISE_REGISTRY.keys())
