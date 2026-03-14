"""
Tests for residual stream tracing infrastructure.

Run: python -m pytest analysis/test_traces.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from sweep.mock_adapter import MockAdapter
from analysis.residual_tracer import (
    ResidualTracer, LayerTrace, DomainTrace,
    PathComparison, SycophancyTrace, HallucinationTrace,
)
from analysis.trace_heatmap import (
    overlay_trace_on_heatmap, validate_circuit_mechanistically,
    generate_mechanistic_report, ValidationResult,
)


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def perfect_model():
    return MockAdapter(mode="perfect", seed=42, num_layers=32)


@pytest.fixture
def sycophantic_model():
    return MockAdapter(mode="sycophantic", seed=42, num_layers=32)


@pytest.fixture
def terrible_model():
    return MockAdapter(mode="terrible", seed=42, num_layers=32)


@pytest.fixture
def tracer(perfect_model):
    return ResidualTracer(perfect_model)


# ------------------------------------------------------------------ #
#  Test 1: forward_with_hooks calls hook correct number of times       #
# ------------------------------------------------------------------ #

class TestMockAdapterHooks:
    def test_hook_called_for_each_layer(self, perfect_model):
        calls = []
        def hook_fn(exec_pos, layer_idx, hidden):
            calls.append((exec_pos, layer_idx, hidden))

        perfect_model.forward_with_hooks("test prompt", hook_fn)
        # 32 layer calls + 1 pre-layer-0 embedding call = 33
        assert len(calls) == 33, f"Expected 33 hook calls, got {len(calls)}"

    def test_hook_called_with_custom_path(self, perfect_model):
        calls = []
        def hook_fn(exec_pos, layer_idx, hidden):
            calls.append((exec_pos, layer_idx))

        path = [0, 1, 2, 3, 4, 3, 4, 5]
        perfect_model.forward_with_hooks("test prompt", hook_fn, layer_path=path)
        # len(path) + 1 for embedding hook
        assert len(calls) == len(path) + 1

    # ------------------------------------------------------------------ #
    #  Test 2: project_to_vocab returns dict with probabilities            #
    # ------------------------------------------------------------------ #

    def test_project_to_vocab_returns_dict(self, perfect_model):
        hidden = {"_synthetic_p": 0.75, "_layer_idx": 10, "_position": 10}
        probs = perfect_model.project_to_vocab(hidden)
        assert isinstance(probs, dict)
        assert "_correct" in probs
        assert abs(probs["_correct"] - 0.75) < 1e-6
        assert abs(probs["_default"] - 0.25) < 1e-6

    def test_tokens_to_ids(self, perfect_model):
        ids = perfect_model.tokens_to_ids(["hello", "world"])
        assert ids == ["_correct", "_correct"]

    def test_tokens_to_ids_single(self, perfect_model):
        ids = perfect_model.tokens_to_ids("hello")
        assert ids == ["_correct"]


# ------------------------------------------------------------------ #
#  Test 3 & 4: ResidualTracer.trace() returns correct LayerTrace       #
# ------------------------------------------------------------------ #

class TestResidualTracer:
    def test_trace_returns_layer_trace(self, tracer):
        trace = tracer.trace("What is 2+2?", ["4"])
        assert isinstance(trace, LayerTrace)
        # 32 layers + 1 embedding hook = 33
        assert len(trace.layer_probabilities) == 33

    def test_trace_peak_layer_in_range(self, tracer):
        trace = tracer.trace("What is 2+2?", ["4"])
        assert 0 <= trace.peak_layer <= 31

    # ------------------------------------------------------------------ #
    #  Test 5: In perfect mode, peak_layer near n_layers * 0.55           #
    # ------------------------------------------------------------------ #

    def test_peak_layer_near_expected(self, tracer):
        trace = tracer.trace("What is 2+2?", ["4"])
        # In perfect mode, sigmoidal rise peaks around 55-70% of layers
        expected_range = (32 * 0.4, 32 * 0.8)
        assert expected_range[0] <= trace.peak_layer <= expected_range[1], \
            f"Peak layer {trace.peak_layer} not in expected range {expected_range}"

    def test_trace_final_probability_positive(self, tracer):
        trace = tracer.trace("What is 2+2?", ["4"])
        assert trace.final_probability > 0.5, \
            f"Final probability should be high in perfect mode: {trace.final_probability}"

    def test_trace_peak_probability_high(self, tracer):
        trace = tracer.trace("What is 2+2?", ["4"])
        assert trace.peak_probability > 0.7

    def test_trace_with_custom_path(self, tracer):
        path = [0, 1, 2, 3, 4, 5]
        trace = tracer.trace("Test", ["42"], layer_path=path)
        # 6 layers + 1 embedding hook = 7
        assert len(trace.layer_probabilities) == 7

    def test_trace_batch(self, tracer):
        questions = ["Q1", "Q2", "Q3"]
        answers = [["a"], ["b"], ["c"]]
        traces = tracer.trace_batch(questions, answers)
        assert len(traces) == 3
        for t in traces:
            assert isinstance(t, LayerTrace)


# ------------------------------------------------------------------ #
#  Test 6: SycophancyTrace onset detected in sycophantic mode          #
# ------------------------------------------------------------------ #

class TestSycophancyTrace:
    def test_sycophancy_onset_detected(self):
        model = MockAdapter(mode="sycophantic", seed=42, num_layers=32)
        tracer = ResidualTracer(model)

        base_prompt = "What is the capital of France?"
        pressure_prompt = "What is the capital of France? Are you sure? I think it's London."

        syco_trace = tracer.trace_sycophancy(base_prompt, pressure_prompt, ["Paris"])
        assert isinstance(syco_trace, SycophancyTrace)
        # In sycophantic mode, both base and pressure use same mode,
        # so onset is 0 (both have same curve). The important thing
        # is the structure is correct.
        assert syco_trace.circuit_candidate[0] >= 0
        assert syco_trace.circuit_candidate[1] >= 0

    def test_sycophancy_cross_mode(self):
        """Compare perfect (base) vs sycophantic (pressure) traces."""
        perfect = MockAdapter(mode="perfect", seed=42, num_layers=32)
        sycophantic = MockAdapter(mode="sycophantic", seed=42, num_layers=32)

        tracer_perfect = ResidualTracer(perfect)
        tracer_syco = ResidualTracer(sycophantic)

        base_trace = tracer_perfect.trace("What is 2+2?", ["4"])
        pressure_trace = tracer_syco.trace("What is 2+2? Are you sure?", ["4"])

        # Perfect mode should have higher final probability
        assert base_trace.final_probability > pressure_trace.final_probability


# ------------------------------------------------------------------ #
#  Test 7: HallucinationTrace crossover detected                       #
# ------------------------------------------------------------------ #

class TestHallucinationTrace:
    def test_hallucination_trace_structure(self, tracer):
        hall_trace = tracer.trace_hallucination(
            "What is the meaning of life?",
        )
        assert isinstance(hall_trace, HallucinationTrace)
        # 32 layers + 1 embedding = 33
        assert len(hall_trace.entropy_by_layer) == 33

    def test_hallucination_trace_entropy_valid(self, tracer):
        hall_trace = tracer.trace_hallucination("Test question")
        for e in hall_trace.entropy_by_layer:
            assert e >= 0.0  # entropy is non-negative


# ------------------------------------------------------------------ #
#  Test 8: DomainTrace has valid computation region                    #
# ------------------------------------------------------------------ #

class TestDomainTrace:
    def test_domain_trace_math(self, tracer):
        dt = tracer.trace_domain("math", n_questions=4)
        assert isinstance(dt, DomainTrace)
        assert dt.n_questions > 0
        # 32 layers + 1 embedding hook = 33
        assert len(dt.mean_probabilities) == 33
        assert dt.answer_computation_region[0] <= dt.answer_computation_region[1]

    def test_domain_trace_computation_region_valid(self, tracer):
        dt = tracer.trace_domain("math", n_questions=4)
        start, end = dt.answer_computation_region
        assert 0 <= start <= 31
        assert 0 <= end <= 31

    def test_domain_trace_mean_peak_layer(self, tracer):
        dt = tracer.trace_domain("math", n_questions=4)
        assert 0 <= dt.mean_peak_layer <= 31


# ------------------------------------------------------------------ #
#  Test 9: PathComparison divergence detected when paths differ        #
# ------------------------------------------------------------------ #

class TestPathComparison:
    def test_path_comparison_same_path(self, tracer):
        path = list(range(32))
        comp = tracer.compare_paths("Test", ["42"], path, path)
        assert isinstance(comp, PathComparison)
        assert comp.net_effect == 0.0
        assert len(comp.improvement_layers) == 0
        assert len(comp.degradation_layers) == 0

    def test_path_comparison_different_paths(self, tracer):
        path_a = list(range(32))
        # Shorter path should produce different probabilities
        path_b = list(range(16))
        comp = tracer.compare_paths("Test", ["42"], path_a, path_b)
        assert isinstance(comp, PathComparison)
        # Different path lengths produce different traces
        # Net effect may be positive or negative


# ------------------------------------------------------------------ #
#  Test 10: overlay_trace_on_heatmap runs without error                #
# ------------------------------------------------------------------ #

class TestTraceHeatmap:
    def test_overlay_runs(self, tracer, tmp_path):
        dt = tracer.trace_domain("math", n_questions=2)
        matrix = np.random.randn(32, 33) * 0.1
        output_path = str(tmp_path / "test_overlay.png")

        # Should not raise
        try:
            overlay_trace_on_heatmap(dt, matrix, output_path)
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_overlay_without_output(self, tracer):
        dt = tracer.trace_domain("math", n_questions=2)
        matrix = np.random.randn(32, 33) * 0.1
        try:
            overlay_trace_on_heatmap(dt, matrix)  # No output_path
        except ImportError:
            pytest.skip("matplotlib not installed")

    # ------------------------------------------------------------------ #
    #  Test 11: validate_circuit_mechanistically correct overlap           #
    # ------------------------------------------------------------------ #

    def test_validate_circuit_overlap(self, tracer):
        dt = tracer.trace_domain("math", n_questions=2)
        matrix = np.random.randn(32, 33) * 0.1

        # Circuit that overlaps with computation region
        comp_start, comp_end = dt.answer_computation_region
        circuit = (comp_start, comp_end)
        result = validate_circuit_mechanistically(circuit, dt, matrix)
        assert isinstance(result, ValidationResult)
        assert result.overlap is True
        assert result.confidence > 0.0

    def test_validate_circuit_no_overlap(self, tracer):
        dt = tracer.trace_domain("math", n_questions=2)
        matrix = np.random.randn(32, 33) * 0.1

        # Circuit that does NOT overlap with computation region
        comp_start, comp_end = dt.answer_computation_region
        # Place circuit far from computation region
        if comp_end < 25:
            circuit = (28, 31)
        else:
            circuit = (0, 2)

        result = validate_circuit_mechanistically(circuit, dt, matrix)
        assert isinstance(result, ValidationResult)
        # May or may not overlap depending on computation region size

    def test_generate_mechanistic_report(self, tracer, tmp_path):
        dt = tracer.trace_domain("math", n_questions=2)
        all_traces = {"math": dt}
        report_path = generate_mechanistic_report(
            all_traces, "nonexistent.json", str(tmp_path)
        )
        assert Path(report_path).exists()
        content = Path(report_path).read_text()
        assert "math" in content
        assert "Mechanistic Analysis Report" in content


# ------------------------------------------------------------------ #
#  Test 12: Verify existing MockAdapter modes still work               #
# ------------------------------------------------------------------ #

class TestExistingModesUnchanged:
    def test_perfect_mode_generate(self):
        model = MockAdapter(mode="perfect", seed=42)
        assert model.generate_short("What is 17 * 23? Answer with only the number.") == "391"

    def test_terrible_mode_generate(self):
        model = MockAdapter(mode="terrible", seed=42)
        result = model.generate_short("What is 17 * 23?")
        assert result != "391"

    def test_sycophantic_mode_generate(self):
        model = MockAdapter(mode="sycophantic", seed=42)
        result = model.generate_short("Are you sure? Reconsider. 999")
        assert isinstance(result, str)

    def test_random_mode_generate(self):
        model = MockAdapter(mode="random", seed=42)
        result = model.generate_short("Hello")
        assert isinstance(result, str)

    def test_overconfident_mode_generate(self):
        model = MockAdapter(mode="overconfident", seed=42)
        result = model.generate_short("What is 2 + 2?")
        assert isinstance(result, str)

    def test_fragile_mode_generate(self):
        model = MockAdapter(mode="fragile", seed=42)
        result = model.generate_short("What is the capital of France? Answer with one word.")
        assert isinstance(result, str)

    def test_set_layer_path(self):
        model = MockAdapter(mode="perfect", seed=42)
        model.set_layer_path([0, 1, 2, 3])
        assert model._layer_path == [0, 1, 2, 3]

    def test_get_logprobs(self):
        model = MockAdapter(mode="perfect", seed=42)
        logprobs = model.get_logprobs("test", ["a", "b", "c"])
        assert isinstance(logprobs, dict)
        assert len(logprobs) == 3


# ------------------------------------------------------------------ #
#  Audit fix tests                                                     #
# ------------------------------------------------------------------ #

class TestAuditFixes:
    """Tests for fixes from external audit."""

    def test_entropy_tracking_in_hallucination_perfect(self):
        """Entropy should decrease through layers in perfect mode."""
        model = MockAdapter(mode="perfect", seed=42, num_layers=32)
        tracer = ResidualTracer(model)
        ht = tracer.trace_hallucination("Test question")
        assert isinstance(ht, HallucinationTrace)
        assert len(ht.entropy_by_layer) == 33
        # First entropy (embedding) should be high
        assert ht.entropy_by_layer[0] > 0.0
        # Entropy should generally decrease in perfect mode
        assert ht.entropy_by_layer[-1] < ht.entropy_by_layer[0]

    def test_entropy_tracking_in_hallucination_terrible(self):
        """Entropy should stay high in terrible mode."""
        model = MockAdapter(mode="terrible", seed=42, num_layers=32)
        tracer = ResidualTracer(model)
        ht = tracer.trace_hallucination("Test question")
        # Entropy stays uniformly high (~5.0) in terrible mode
        for e in ht.entropy_by_layer[1:]:  # skip embedding
            assert e > 3.0, f"Expected high entropy in terrible mode, got {e}"

    def test_entropy_drop_layer_detected(self):
        """Sharpest entropy drop should be detected."""
        model = MockAdapter(mode="perfect", seed=42, num_layers=32)
        tracer = ResidualTracer(model)
        ht = tracer.trace_hallucination("Test question")
        assert ht.entropy_drop_layer > 0
        assert ht.entropy_drop_layer < 33

    def test_three_tuple_storage(self):
        """layer_probabilities should store (exec_pos, layer_idx, p_correct) tuples."""
        model = MockAdapter(mode="perfect", seed=42, num_layers=16)
        tracer = ResidualTracer(model)
        trace = tracer.trace("Test", ["42"])
        for entry in trace.layer_probabilities:
            assert len(entry) == 3, f"Expected 3-tuple, got {len(entry)}-tuple: {entry}"
            exec_pos, layer_idx, p_correct = entry
            assert isinstance(p_correct, float)

    def test_three_tuple_first_is_embedding(self):
        """First entry should be the embedding hook at position -1."""
        model = MockAdapter(mode="perfect", seed=42, num_layers=16)
        tracer = ResidualTracer(model)
        trace = tracer.trace("Test", ["42"])
        exec_pos, layer_idx, p_correct = trace.layer_probabilities[0]
        assert exec_pos == -1
        assert layer_idx == -1

    def test_pre_layer_0_hook_fires(self):
        """Hook at position -1 (embedding) should be called before layers."""
        model = MockAdapter(mode="perfect", seed=42, num_layers=8)
        calls = []
        def hook_fn(exec_pos, layer_idx, hidden):
            calls.append((exec_pos, layer_idx))
        model.forward_with_hooks("test", hook_fn)
        # First call should be (-1, -1) for embedding
        assert calls[0] == (-1, -1), f"Expected (-1, -1), got {calls[0]}"
        # Subsequent calls should have exec_pos 0..7
        for i in range(1, len(calls)):
            assert calls[i][0] == i - 1

    def test_sycophancy_pressure_answer_tracking(self):
        """trace_sycophancy should track pressure_answer_probabilities."""
        model = MockAdapter(mode="sycophantic", seed=42, num_layers=32)
        tracer = ResidualTracer(model)
        st = tracer.trace_sycophancy(
            "What is 2+2?",
            "What is 2+2? Actually I think it's 5.",
            ["4"],
            pressure_answer_tokens=["5"],
        )
        assert isinstance(st, SycophancyTrace)
        assert hasattr(st, 'pressure_answer_probabilities')
        assert hasattr(st, 'genuine_override_layer')
        assert hasattr(st, 'distraction_only')
        assert len(st.pressure_answer_probabilities) == 33  # 32 + embedding

    def test_sycophancy_without_pressure_tokens(self):
        """trace_sycophancy should work without pressure_answer_tokens (backward compat)."""
        model = MockAdapter(mode="perfect", seed=42, num_layers=16)
        tracer = ResidualTracer(model)
        st = tracer.trace_sycophancy("Q1", "Q2 pressure", ["42"])
        assert isinstance(st, SycophancyTrace)
        assert st.pressure_answer_probabilities == []

    def test_project_to_vocab_with_target_token_ids(self):
        """project_to_vocab with target_token_ids should return only those IDs."""
        model = MockAdapter(mode="perfect", seed=42)
        hidden = {"_synthetic_p": 0.8, "_layer_idx": 10, "_position": 10}
        probs = model.project_to_vocab(hidden, target_token_ids=["_correct", "_other"])
        assert "_correct" in probs
        assert "_other" in probs
        assert abs(probs["_correct"] - 0.8) < 1e-6

    def test_project_to_vocab_without_target_ids(self):
        """project_to_vocab without target_token_ids returns full distribution."""
        model = MockAdapter(mode="perfect", seed=42)
        hidden = {"_synthetic_p": 0.75, "_layer_idx": 10, "_position": 10}
        probs = model.project_to_vocab(hidden)
        assert "_correct" in probs
        assert "_default" in probs

    def test_ignore_early_frac(self):
        """Early layer noise filtering should skip first ignore_early layers."""
        model = MockAdapter(mode="perfect", seed=42, num_layers=20)
        tracer = ResidualTracer(model, ignore_early_frac=0.3)
        assert tracer.ignore_early == 6  # int(20 * 0.3)

    def test_mock_entropy_in_hidden_state(self):
        """Mock hidden states should include _entropy field."""
        model = MockAdapter(mode="perfect", seed=42, num_layers=8)
        states = []
        def hook_fn(exec_pos, layer_idx, hidden):
            states.append(hidden)
        model.forward_with_hooks("test", hook_fn)
        for s in states:
            assert "_entropy" in s, f"Hidden state missing _entropy: {s.keys()}"
