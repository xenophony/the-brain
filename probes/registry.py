"""
Probe registry — BaseProbe, registration, and lookup.

All probes inherit from BaseProbe and use @register_probe to self-register.
"""

from abc import ABC, abstractmethod

_REGISTRY: dict[str, "BaseProbe"] = {}


class BaseProbe(ABC):
    name: str = ""
    description: str = ""
    log_responses: bool = False  # set True during baseline runs
    max_items: int | None = 8   # default subset size; None = use all items

    @abstractmethod
    def run(self, model) -> "float | dict":
        """Run the probe against a model adapter, return score in [0.0, 1.0] or result dict."""
        ...

    def _run_items(self, model, items: list, prompt_fn, score_fn,
                   max_new_tokens: int = 20, temperature: float = 0.0,
                   difficulty: str = "") -> tuple[list[float], list[dict]]:
        """Run items with automatic batching when available.

        Args:
            model: adapter with generate_short (and optionally generate_short_batch)
            items: list of item dicts
            prompt_fn: callable(item) -> prompt string
            score_fn: callable(response, item) -> float score
            max_new_tokens: max tokens per response
            temperature: sampling temperature
            difficulty: label for item_results

        Returns:
            (scores, item_results) — item_results is [] if not log_responses
        """
        prompts = [prompt_fn(item) for item in items]

        # Use batched generation if available
        if hasattr(model, 'generate_short_batch') and len(prompts) > 1:
            responses = model.generate_short_batch(
                prompts, max_new_tokens, temperature)
        else:
            responses = [model.generate_short(p, max_new_tokens, temperature)
                         for p in prompts]

        scores = []
        item_results = []
        for item, response in zip(items, responses):
            score = score_fn(response, item)
            scores.append(score)
            if self.log_responses:
                item_results.append({
                    "difficulty": difficulty,
                    "response": response[:200] if response else "",
                    "score": score,
                })
        return scores, item_results

    def _limit(self, items: list) -> list:
        """Slice items to max_items, taking equally from the list.
        Returns all items if max_items is None."""
        if self.max_items is None or len(items) <= self.max_items:
            return items
        return items[:self.max_items]

    def _make_result(self, easy_scores: list[float], hard_scores: list[float],
                     item_results: list[dict] | None = None) -> dict:
        """Build standardized result dict from easy and hard score lists."""
        all_scores = easy_scores + hard_scores
        result = {
            "score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "easy_score": sum(easy_scores) / len(easy_scores) if easy_scores else 0.0,
            "hard_score": sum(hard_scores) / len(hard_scores) if hard_scores else 0.0,
            "n_easy": len(easy_scores),
            "n_hard": len(hard_scores),
        }
        if item_results is not None:
            result["item_results"] = item_results
        return result

    def expected_digit_score(self, response: str, expected: int) -> float:
        """Score a digit response with partial credit for near-misses."""
        response = response.strip()
        # Extract first digit found
        for ch in response:
            if ch.isdigit():
                got = int(ch)
                if got == expected:
                    return 1.0
                diff = abs(got - expected)
                if diff == 1:
                    return 0.5
                if diff == 2:
                    return 0.25
                return 0.0
        return 0.0


class BaseLogprobProbe(BaseProbe):
    """Base class for logprob-based probes. Zero decode steps — 1 forward pass per question.

    Subclasses define ITEMS (list of dicts with "prompt", "answer", "difficulty")
    and CHOICES (list of valid answer strings for logprob measurement).

    Returns two scoring signals per question:
      - argmax_correct: 1.0 if highest-probability choice matches answer, else 0.0
      - p_correct: raw probability of the correct answer (continuous 0.0-1.0)

    The result dict contains:
      - score: argmax accuracy (for compatibility with existing heatmaps)
      - p_correct: mean probability of correct answer (more sensitive signal)
      - p_correct_easy / p_correct_hard: difficulty breakdown
      - Per-item logprob details when log_responses=True
    """
    ITEMS: list[dict] = []
    CHOICES: list[str] = []

    def run(self, model) -> dict:
        import math

        items = self._limit(self.ITEMS)
        choices = self.CHOICES

        easy_argmax = []
        hard_argmax = []
        easy_pcorrect = []
        hard_pcorrect = []
        item_results = []

        # Batch all logprob calls in one forward pass
        prompts = [item["prompt"] + " Answer with one word." for item in items]
        if hasattr(model, 'get_logprobs_batch') and len(prompts) > 1:
            all_logprobs = model.get_logprobs_batch(prompts, choices)
        else:
            all_logprobs = [model.get_logprobs(p, choices) for p in prompts]

        for item, logprobs in zip(items, all_logprobs):
            expected = item["answer"].lower()
            difficulty = item.get("difficulty", "hard")

            # Convert log probs to probabilities
            probs = {}
            for choice in choices:
                lp = logprobs.get(choice, float('-inf'))
                probs[choice] = math.exp(lp) if lp > -100 else 0.0

            # Normalize to sum to 1 across choices
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}

            # Mode 1: argmax accuracy
            best_choice = max(probs, key=probs.get) if probs else ""
            argmax_correct = 1.0 if best_choice == expected else 0.0

            # Mode 2: probability of correct answer
            p_correct = probs.get(expected, 0.0)

            if difficulty == "easy":
                easy_argmax.append(argmax_correct)
                easy_pcorrect.append(p_correct)
            else:
                hard_argmax.append(argmax_correct)
                hard_pcorrect.append(p_correct)

            if self.log_responses:
                item_results.append({
                    "difficulty": difficulty,
                    "prompt": item["prompt"][:100],
                    "expected": expected,
                    "argmax": best_choice,
                    "argmax_correct": argmax_correct,
                    "p_correct": round(p_correct, 4),
                    "probs": {k: round(v, 4) for k, v in probs.items()},
                })

        all_argmax = easy_argmax + hard_argmax
        all_pcorrect = easy_pcorrect + hard_pcorrect

        result = {
            # Mode 1: argmax accuracy (compatible with existing heatmap pipeline)
            "score": sum(all_argmax) / len(all_argmax) if all_argmax else 0.0,
            "easy_score": sum(easy_argmax) / len(easy_argmax) if easy_argmax else 0.0,
            "hard_score": sum(hard_argmax) / len(hard_argmax) if hard_argmax else 0.0,
            # Mode 2: probability tracking (more sensitive for circuit mapping)
            "p_correct": sum(all_pcorrect) / len(all_pcorrect) if all_pcorrect else 0.0,
            "p_correct_easy": sum(easy_pcorrect) / len(easy_pcorrect) if easy_pcorrect else 0.0,
            "p_correct_hard": sum(hard_pcorrect) / len(hard_pcorrect) if hard_pcorrect else 0.0,
            "n_easy": len(easy_argmax),
            "n_hard": len(hard_argmax),
        }
        if item_results:
            result["item_results"] = item_results
        return result


def register_probe(cls):
    """Class decorator — registers a probe by its .name attribute."""
    instance = cls()
    _REGISTRY[instance.name] = instance
    return cls


def get_probe(name: str) -> BaseProbe:
    """Look up a registered probe by name."""
    if name not in _REGISTRY:
        # Try auto-importing
        import importlib
        try:
            importlib.import_module(f"probes.{name}.probe")
        except ImportError:
            pass
    if name not in _REGISTRY:
        raise KeyError(f"Probe '{name}' not registered. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_probes() -> list[str]:
    """Return names of all registered probes."""
    # Trigger auto-discovery
    import probes  # noqa: F811 — triggers __init__ auto-import
    return list(_REGISTRY.keys())
