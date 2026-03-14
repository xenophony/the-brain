"""
Residual stream tracing infrastructure.

Traces how the probability of the correct answer evolves layer-by-layer
through the transformer's residual stream. At each layer, we project the
hidden state to vocabulary space and measure p(correct_token).

This reveals:
- Where in the network the answer is "computed" (steepest probability rise)
- Which layers suppress a previously-correct answer
- How social pressure or hallucination tendencies manifest layer-by-layer

Works with both MockAdapter (synthetic hidden states) and ExLlamaV2LayerAdapter
(real model hidden states projected through lm_head).
"""

import math
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class LayerTrace:
    """Per-question trace of correct-answer probability through layers."""
    prompt: str
    correct_tokens: list
    layer_probabilities: list  # [(exec_pos, layer_idx, p_correct), ...]
    peak_layer: int = 0
    peak_probability: float = 0.0
    final_probability: float = 0.0
    suppression_layers: list = field(default_factory=list)  # layers where p drops > 0.1
    amplification_layers: list = field(default_factory=list)  # layers where p rises > 0.1


@dataclass
class DomainTrace:
    """Aggregated trace across multiple questions for a domain/probe."""
    probe_name: str
    n_questions: int
    mean_probabilities: list  # per layer, mean across questions
    std_probabilities: list
    mean_peak_layer: float = 0.0
    answer_computation_region: tuple = (0, 0)  # steepest rise region
    suppression_regions: list = field(default_factory=list)


@dataclass
class PathComparison:
    """Compare same prompt through two different layer paths."""
    path_a_trace: dict  # serialized LayerTrace
    path_b_trace: dict
    divergence_layer: int = 0
    improvement_layers: list = field(default_factory=list)
    degradation_layers: list = field(default_factory=list)
    net_effect: float = 0.0


@dataclass
class SycophancyTrace:
    """Trace how social pressure suppresses correct answer probability."""
    base_trace: dict
    pressure_trace: dict
    sycophancy_onset_layer: int = 0
    correct_answer_suppression: float = 0.0
    circuit_candidate: tuple = (0, 0)
    pressure_answer_probabilities: list = field(default_factory=list)
    genuine_override_layer: int = 0  # where p(pressure) rises AND p(correct) falls
    distraction_only: bool = False  # True if p(correct) falls but p(pressure) doesn't rise


@dataclass
class HallucinationTrace:
    """Trace entropy evolution to detect premature commitment."""
    entropy_by_layer: list  # per-layer entropy of full vocab distribution
    entropy_drop_layer: int = 0  # layer where entropy drops most sharply
    crossover_layer: int = 0  # layer where entropy drops below threshold
    suppression_circuit: tuple = (0, 0)


class ResidualTracer:
    """
    Traces per-layer probability of correct answer tokens through the
    transformer residual stream.

    Works with any model adapter that implements:
      - forward_with_hooks(prompt, hook_fn, layer_path)
      - project_to_vocab(hidden_state, target_token_ids=None) -> dict[token_id, probability]
      - tokens_to_ids(token_strings) -> list[token_id]
      - num_layers: int
    """

    def __init__(self, model, ignore_early_frac=0.2):
        self.model = model
        self.n_layers = model.num_layers
        self.ignore_early = int(model.num_layers * ignore_early_frac)

    def trace(self, prompt: str, correct_tokens, layer_path=None) -> LayerTrace:
        """Trace per-layer probability of correct answer tokens."""
        if layer_path is None:
            layer_path = list(range(self.n_layers))

        # Collect hidden states via hooks
        hidden_states = []

        def hook_fn(exec_pos, layer_idx, hidden):
            hidden_states.append((exec_pos, layer_idx, hidden))

        self.model.forward_with_hooks(prompt, hook_fn, layer_path)

        # Use only first token of correct answer (standard logit lens practice)
        if isinstance(correct_tokens, list) and correct_tokens:
            if isinstance(correct_tokens[0], str):
                token_ids = self.model.tokens_to_ids(correct_tokens)
            else:
                token_ids = correct_tokens
        else:
            token_ids = []

        # Project each hidden state to vocab and get p(correct)
        layer_probs = []
        for exec_pos, layer_idx, hidden in hidden_states:
            probs = self.model.project_to_vocab(hidden)
            if token_ids:
                p_correct = probs.get(token_ids[0], 0.0)
            else:
                p_correct = 0.0
            layer_probs.append((exec_pos, layer_idx, p_correct))

        # Compute trace metrics
        if not layer_probs:
            return LayerTrace(
                prompt=prompt, correct_tokens=correct_tokens,
                layer_probabilities=[]
            )

        probabilities = [p for _, _, p in layer_probs]
        peak_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
        peak_layer = layer_probs[peak_idx][0]  # exec_pos

        # Find suppression and amplification layers, skipping early layers
        suppression = []
        amplification = []
        for i in range(1, len(probabilities)):
            exec_pos = layer_probs[i][0]
            if exec_pos >= 0 and exec_pos < self.ignore_early:
                continue
            diff = probabilities[i] - probabilities[i - 1]
            if diff < -0.1:
                suppression.append(exec_pos)
            if diff > 0.1:
                amplification.append(exec_pos)

        return LayerTrace(
            prompt=prompt,
            correct_tokens=correct_tokens,
            layer_probabilities=layer_probs,
            peak_layer=peak_layer,
            peak_probability=probabilities[peak_idx],
            final_probability=probabilities[-1],
            suppression_layers=suppression,
            amplification_layers=amplification,
        )

    def trace_batch(self, questions: list, correct_tokens_list: list, layer_path=None) -> list:
        """Trace multiple questions. Returns list of LayerTrace."""
        traces = []
        for q, tokens in zip(questions, correct_tokens_list):
            traces.append(self.trace(q, tokens, layer_path))
        return traces

    def trace_domain(self, probe_name: str, n_questions=8, layer_path=None) -> DomainTrace:
        """Load probe, trace first n hard items."""
        import importlib

        mod = importlib.import_module(f"probes.{probe_name}.probe")
        hard_items = getattr(mod, 'HARD_ITEMS', [])
        if not hard_items:
            hard_items = getattr(mod, 'QUESTIONS', getattr(mod, 'SCENARIOS', []))
        items = hard_items[:n_questions]

        if not items:
            return DomainTrace(
                probe_name=probe_name, n_questions=0,
                mean_probabilities=[], std_probabilities=[]
            )

        # Extract prompts and expected answers, trace each
        traces = []
        for item in items:
            prompt = item.get("prompt", item.get("question", ""))
            answer = item.get("answer", item.get("correct", item.get("accept", ["42"])))
            if isinstance(answer, (int, float)):
                answer = [str(answer)]
            elif isinstance(answer, str):
                answer = [answer]
            elif isinstance(answer, list) and answer:
                answer = [str(answer[0])]
            else:
                answer = ["42"]

            trace = self.trace(prompt, answer, layer_path)
            traces.append(trace)

        # Aggregate with numpy
        try:
            import numpy as np
        except ImportError:
            return DomainTrace(
                probe_name=probe_name, n_questions=len(traces),
                mean_probabilities=[], std_probabilities=[]
            )

        n_layers_traced = len(traces[0].layer_probabilities) if traces else 0
        all_probs = []
        for t in traces:
            probs = [p for _, _, p in t.layer_probabilities]
            if len(probs) == n_layers_traced:
                all_probs.append(probs)

        if not all_probs:
            return DomainTrace(
                probe_name=probe_name, n_questions=len(traces),
                mean_probabilities=[], std_probabilities=[]
            )

        arr = np.array(all_probs)
        mean_probs = arr.mean(axis=0).tolist()
        std_probs = arr.std(axis=0).tolist()
        mean_peak = float(np.mean([t.peak_layer for t in traces]))

        # Find steepest rise region (answer computation region)
        diffs = [mean_probs[i] - mean_probs[i - 1] for i in range(1, len(mean_probs))]
        computation_region = (0, 0)
        if diffs:
            best_start = 0
            best_end = 0
            best_sum = 0
            cur_start = 0
            cur_sum = 0
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
                    cur_sum = 0
            if best_sum > 0:
                computation_region = (best_start, best_end)

        # Find suppression regions
        suppression_regions = []
        in_suppression = False
        s_start = 0
        for i, d in enumerate(diffs):
            if d < -0.05 and not in_suppression:
                s_start = i
                in_suppression = True
            elif d >= 0 and in_suppression:
                suppression_regions.append((s_start, i))
                in_suppression = False
        if in_suppression:
            suppression_regions.append((s_start, len(diffs)))

        return DomainTrace(
            probe_name=probe_name,
            n_questions=len(traces),
            mean_probabilities=mean_probs,
            std_probabilities=std_probs,
            mean_peak_layer=mean_peak,
            answer_computation_region=computation_region,
            suppression_regions=suppression_regions,
        )

    def compare_paths(self, prompt, correct_tokens, path_a, path_b) -> PathComparison:
        """Compare same prompt through two layer paths."""
        trace_a = self.trace(prompt, correct_tokens, path_a)
        trace_b = self.trace(prompt, correct_tokens, path_b)

        probs_a = [p for _, _, p in trace_a.layer_probabilities]
        probs_b = [p for _, _, p in trace_b.layer_probabilities]

        min_len = min(len(probs_a), len(probs_b))

        divergence = 0
        improvement = []
        degradation = []
        for i in range(min_len):
            diff = probs_b[i] - probs_a[i]
            if abs(diff) > 0.05 and divergence == 0:
                divergence = i
            if diff > 0.05:
                improvement.append(i)
            elif diff < -0.05:
                degradation.append(i)

        net = (trace_b.final_probability - trace_a.final_probability)

        return PathComparison(
            path_a_trace=asdict(trace_a),
            path_b_trace=asdict(trace_b),
            divergence_layer=divergence,
            improvement_layers=improvement,
            degradation_layers=degradation,
            net_effect=net,
        )

    def trace_sycophancy(self, base_prompt, pressure_prompt, correct_tokens,
                         pressure_answer_tokens=None) -> SycophancyTrace:
        """Trace how social pressure suppresses correct answer probability."""
        base_trace = self.trace(base_prompt, correct_tokens)
        pressure_trace = self.trace(pressure_prompt, correct_tokens)

        base_probs = [p for _, _, p in base_trace.layer_probabilities]
        press_probs = [p for _, _, p in pressure_trace.layer_probabilities]

        min_len = min(len(base_probs), len(press_probs))

        onset_layer = 0
        max_suppression = 0.0
        max_supp_layer = 0
        for i in range(min_len):
            diff = base_probs[i] - press_probs[i]
            if diff > 0.1 and onset_layer == 0:
                onset_layer = i
            if diff > max_suppression:
                max_suppression = diff
                max_supp_layer = i

        suppression = base_trace.final_probability - pressure_trace.final_probability

        # Track pressure answer probabilities if provided
        pressure_answer_probs = []
        genuine_override = 0
        distraction_only = False

        if pressure_answer_tokens is not None:
            # Re-run pressure prompt to capture hidden states for pressure answer tracking
            hidden_states = []

            def hook_fn(exec_pos, layer_idx, hidden):
                hidden_states.append((exec_pos, layer_idx, hidden))

            self.model.forward_with_hooks(pressure_prompt, hook_fn)

            if isinstance(pressure_answer_tokens[0], str):
                pa_ids = self.model.tokens_to_ids(pressure_answer_tokens)
            else:
                pa_ids = pressure_answer_tokens

            for exec_pos, layer_idx, hidden in hidden_states:
                probs = self.model.project_to_vocab(hidden)
                p_pressure = probs.get(pa_ids[0], 0.0) if pa_ids else 0.0
                pressure_answer_probs.append(p_pressure)

            # Find genuine_override_layer: where p(pressure) rises AND p(correct) falls
            for i in range(1, min(len(pressure_answer_probs), len(press_probs))):
                if i < len(press_probs) and i < len(pressure_answer_probs):
                    correct_falling = press_probs[i] < press_probs[i - 1]
                    pressure_rising = pressure_answer_probs[i] > pressure_answer_probs[i - 1]
                    if correct_falling and pressure_rising and genuine_override == 0:
                        genuine_override = i

            # Check if it's distraction_only: p(correct) falls but p(pressure) never rises significantly
            if pressure_answer_probs:
                max_p_pressure = max(pressure_answer_probs)
                if max_p_pressure < 0.1 and suppression > 0.1:
                    distraction_only = True

        return SycophancyTrace(
            base_trace=asdict(base_trace),
            pressure_trace=asdict(pressure_trace),
            sycophancy_onset_layer=onset_layer,
            correct_answer_suppression=suppression,
            circuit_candidate=(onset_layer, max_supp_layer),
            pressure_answer_probabilities=pressure_answer_probs,
            genuine_override_layer=genuine_override,
            distraction_only=distraction_only,
        )

    def trace_hallucination(self, question, entropy_threshold=2.0) -> HallucinationTrace:
        """Trace entropy evolution. Sharp entropy drop = model committing to answer."""
        hidden_states = []

        def hook_fn(exec_pos, layer_idx, hidden):
            hidden_states.append((exec_pos, layer_idx, hidden))

        self.model.forward_with_hooks(question, hook_fn)

        entropies = []
        for exec_pos, layer_idx, hidden in hidden_states:
            probs = self.model.project_to_vocab(hidden)
            # Compute entropy: -sum(p * log(p))
            entropy = 0.0
            for p in probs.values():
                if p > 1e-10:
                    entropy -= p * math.log(p)
            entropies.append(entropy)

        # Find sharpest entropy drop
        max_drop = 0.0
        drop_layer = 0
        for i in range(1, len(entropies)):
            drop = entropies[i - 1] - entropies[i]
            if drop > max_drop:
                max_drop = drop
                drop_layer = i

        # Find crossover: first layer below threshold
        crossover = len(entropies) - 1
        for i, e in enumerate(entropies):
            if e < entropy_threshold:
                crossover = i
                break

        return HallucinationTrace(
            entropy_by_layer=entropies,
            entropy_drop_layer=drop_layer,
            crossover_layer=crossover,
            suppression_circuit=(max(0, drop_layer - 1), drop_layer),
        )
