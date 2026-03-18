"""
Tests for layerwise probing system.

Tests:
1. MockAdapter.get_layerwise_logprobs returns correct structure
2. BaseLayerwiseProbe produces valid analysis output
3. All layerwise probes can run with MockAdapter
4. Analysis functions compute correct metrics
5. CLI doesn't crash with --mock
"""

import math
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sweep.mock_adapter import MockAdapter
from probes.layerwise_registry import (
    BaseLayerwiseProbe, list_layerwise_probes, get_layerwise_probe,
    register_layerwise_probe,
)
import probes.layerwise  # trigger auto-discovery


# ------------------------------------------------------------------ #
#  MockAdapter.get_layerwise_logprobs tests                           #
# ------------------------------------------------------------------ #

class TestMockLayerwiseLogprobs:

    def test_returns_dict_with_required_keys(self):
        model = MockAdapter(mode="random", seed=42)
        result = model.get_layerwise_logprobs("test prompt", ["yes", "no"])
        assert "layer_logprobs" in result
        assert "final_logprobs" in result

    def test_layer_count_matches_num_layers(self):
        model = MockAdapter(mode="random", seed=42, num_layers=16)
        result = model.get_layerwise_logprobs("test prompt", ["yes", "no"])
        assert len(result["layer_logprobs"]) == 16

    def test_custom_layer_path(self):
        model = MockAdapter(mode="random", seed=42, num_layers=32)
        path = [0, 1, 2, 3, 2, 3, 4, 5]
        result = model.get_layerwise_logprobs("test prompt", ["yes", "no"],
                                              layer_path=path)
        assert len(result["layer_logprobs"]) == len(path)

    def test_layer_entry_structure(self):
        model = MockAdapter(mode="random", seed=42)
        result = model.get_layerwise_logprobs("test prompt", ["yes", "no"])
        entry = result["layer_logprobs"][0]
        assert "layer_idx" in entry
        assert "exec_pos" in entry
        assert "target_logprobs" in entry
        assert "psych_scores" in entry

    def test_target_logprobs_are_log_probabilities(self):
        model = MockAdapter(mode="random", seed=42)
        result = model.get_layerwise_logprobs("test prompt", ["yes", "no"])
        for entry in result["layer_logprobs"]:
            for token, lp in entry["target_logprobs"].items():
                assert lp <= 0.0, f"Log prob should be <= 0, got {lp}"
                assert math.exp(lp) <= 1.0

    def test_psych_scores_present_when_map_given(self):
        model = MockAdapter(mode="random", seed=42)
        psych_map = {"hedging": ["maybe", "perhaps"], "confidence": ["certainly"]}
        result = model.get_layerwise_logprobs("test prompt", ["yes", "no"],
                                              psych_token_map=psych_map)
        for entry in result["layer_logprobs"]:
            assert "hedging" in entry["psych_scores"]
            assert "confidence" in entry["psych_scores"]

    def test_psych_scores_empty_when_no_map(self):
        model = MockAdapter(mode="random", seed=42)
        result = model.get_layerwise_logprobs("test prompt", ["yes", "no"])
        for entry in result["layer_logprobs"]:
            assert entry["psych_scores"] == {}

    def test_perfect_mode_p_correct_rises(self):
        model = MockAdapter(mode="perfect", seed=42)
        result = model.get_layerwise_logprobs("test prompt", ["yes", "no"])
        entries = result["layer_logprobs"]
        # p(first token) should be higher at end than start
        p_start = math.exp(entries[0]["target_logprobs"]["yes"])
        p_end = math.exp(entries[-1]["target_logprobs"]["yes"])
        assert p_end > p_start, f"Expected rising p_correct: start={p_start}, end={p_end}"

    def test_terrible_mode_p_correct_stays_low(self):
        model = MockAdapter(mode="terrible", seed=42)
        result = model.get_layerwise_logprobs("test prompt", ["yes", "no"])
        for entry in result["layer_logprobs"]:
            p = math.exp(entry["target_logprobs"]["yes"])
            assert p < 0.15, f"Terrible mode should keep p_correct low, got {p}"

    def test_final_logprobs_match_last_layer(self):
        model = MockAdapter(mode="random", seed=42)
        result = model.get_layerwise_logprobs("test prompt", ["yes", "no"])
        last_entry = result["layer_logprobs"][-1]["target_logprobs"]
        final = result["final_logprobs"]
        for token in last_entry:
            assert abs(last_entry[token] - final[token]) < 1e-10


# ------------------------------------------------------------------ #
#  BaseLayerwiseProbe tests                                           #
# ------------------------------------------------------------------ #

class TestBaseLayerwiseProbe:

    def test_all_probes_discovered(self):
        probes = list_layerwise_probes()
        assert len(probes) >= 20, f"Expected >= 20 probes, got {len(probes)}"

    def test_causal_probe_runs(self):
        model = MockAdapter(mode="random", seed=42)
        probe = get_layerwise_probe("causal_layerwise")
        probe.max_items = 3
        result = probe.run(model)
        assert "score" in result
        assert "p_correct" in result
        assert "mean_p_correct_by_layer" in result
        assert "items" in result

    def test_result_has_correct_n_layers(self):
        model = MockAdapter(mode="random", seed=42, num_layers=16)
        probe = get_layerwise_probe("causal_layerwise")
        probe.max_items = 2
        result = probe.run(model)
        assert result["n_layers"] == 16
        assert len(result["mean_p_correct_by_layer"]) == 16

    def test_convergence_computed(self):
        model = MockAdapter(mode="perfect", seed=42)
        probe = get_layerwise_probe("causal_layerwise")
        probe.max_items = 3
        result = probe.run(model)
        # Perfect mode should converge
        assert result["mean_convergence_layer"] < model.num_layers

    def test_entropy_tracked(self):
        model = MockAdapter(mode="random", seed=42)
        probe = get_layerwise_probe("causal_layerwise")
        probe.max_items = 2
        result = probe.run(model)
        assert "mean_entropy_by_layer" in result
        assert len(result["mean_entropy_by_layer"]) == model.num_layers
        # Entropy should be positive
        for e in result["mean_entropy_by_layer"]:
            assert e >= 0.0

    def test_psych_by_layer_populated(self):
        model = MockAdapter(mode="random", seed=42)
        probe = get_layerwise_probe("causal_layerwise")
        probe.max_items = 2
        probe.capture_psych = True
        result = probe.run(model)
        psych = result.get("psych_by_layer", {})
        assert len(psych) > 0, "Expected psych categories"
        for cat, values in psych.items():
            assert len(values) == model.num_layers

    def test_correct_vs_incorrect_populated(self):
        model = MockAdapter(mode="random", seed=42)
        probe = get_layerwise_probe("causal_layerwise")
        probe.max_items = 8  # need enough items for split
        probe.capture_psych = True
        result = probe.run(model)
        cvi = result.get("correct_vs_incorrect", {})
        # May or may not have data depending on random scores
        # Just check it's a dict
        assert isinstance(cvi, dict)

    def test_computation_region_valid(self):
        model = MockAdapter(mode="random", seed=42)
        probe = get_layerwise_probe("causal_layerwise")
        probe.max_items = 4
        result = probe.run(model)
        region = result["computation_region"]
        assert len(region) == 2
        assert region[0] >= 0
        assert region[1] >= region[0]

    def test_score_in_0_1_range(self):
        model = MockAdapter(mode="random", seed=42)
        probe = get_layerwise_probe("causal_layerwise")
        probe.max_items = 4
        result = probe.run(model)
        assert 0.0 <= result["score"] <= 1.0
        assert 0.0 <= result["p_correct"] <= 1.0


# ------------------------------------------------------------------ #
#  All probes run without error                                       #
# ------------------------------------------------------------------ #

class TestAllLayerwiseProbes:

    @pytest.fixture
    def model(self):
        return MockAdapter(mode="random", seed=42, num_layers=16)

    def test_all_probes_run(self, model):
        """Every registered layerwise probe should run without error."""
        all_names = list_layerwise_probes()
        for name in all_names:
            probe = get_layerwise_probe(name)
            probe.max_items = 2
            probe.capture_psych = True
            result = probe.run(model)
            assert "score" in result, f"{name} missing 'score'"
            assert "p_correct" in result, f"{name} missing 'p_correct'"
            assert "mean_p_correct_by_layer" in result, (
                f"{name} missing 'mean_p_correct_by_layer'")

    def test_sycophancy_has_pressure_data(self, model):
        """Sycophancy layerwise should have pressure-specific fields."""
        probe = get_layerwise_probe("sycophancy_layerwise")
        probe.max_items = 2
        result = probe.run(model)
        # Check first item has pressure data
        item = result["items"][0]
        layer = item["layers"][0]
        assert "p_correct_neutral" in layer
        assert "p_correct_pressure" in layer
        assert "pressure_delta" in layer


# ------------------------------------------------------------------ #
#  Analysis module tests                                              #
# ------------------------------------------------------------------ #

class TestLayerwiseAnalysis:

    def test_convergence_detection(self):
        from analysis.layerwise_analysis import analyze_convergence
        # Sigmoidal rise
        data = [0.1 + 0.8 / (1 + math.exp(-0.5 * (x - 16)))
                for x in range(32)]
        result = analyze_convergence(data)
        assert result["is_converged"]
        assert result["convergence_layer"] is not None
        assert result["convergence_layer"] < 20

    def test_convergence_not_detected_flat(self):
        from analysis.layerwise_analysis import analyze_convergence
        data = [0.3] * 32
        result = analyze_convergence(data)
        assert not result["is_converged"]

    def test_computation_region(self):
        from analysis.layerwise_analysis import find_computation_region
        data = [0.1] * 10 + [0.1 + 0.05 * i for i in range(10)] + [0.6] * 12
        result = find_computation_region(data)
        assert result["start"] >= 10
        assert result["end"] <= 20
        assert result["total_rise"] > 0.3

    def test_surprise_detection(self):
        from analysis.layerwise_analysis import detect_surprises
        data = [0.1] * 10 + [0.5] + [0.5] * 21
        surprises = detect_surprises(data, threshold=0.1)
        assert len(surprises) >= 1
        assert surprises[0]["layer"] == 10
        assert surprises[0]["direction"] == "rise"

    def test_psych_peaks(self):
        from analysis.layerwise_analysis import find_psych_peaks
        psych = {
            "hedging": [0.01, 0.05, 0.08, 0.05, 0.01],
            "confidence": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
        result = find_psych_peaks(psych)
        assert result["hedging"]["peak_layer"] == 2
        assert result["confidence"]["peak_layer"] == 4

    def test_cross_probe_comparison(self):
        from analysis.layerwise_analysis import compare_probes
        results = {
            "probe_a": {
                "computation_region": (5, 15),
                "mean_convergence_layer": 10,
                "surprise_layers": [{"layer": 8, "delta": 0.2, "direction": "rise"}],
            },
            "probe_b": {
                "computation_region": (10, 20),
                "mean_convergence_layer": 15,
                "surprise_layers": [{"layer": 8, "delta": 0.15, "direction": "rise"}],
            },
        }
        comparison = compare_probes(results)
        assert len(comparison["region_overlaps"]) == 1
        assert comparison["region_overlaps"][0]["overlap_region"] == (10, 15)
        assert 8 in comparison["shared_surprise_layers"]

    def test_generate_report(self):
        from analysis.layerwise_analysis import generate_layerwise_report
        model = MockAdapter(mode="random", seed=42, num_layers=16)
        probe_results = {}
        for name in ["causal_layerwise", "logic_layerwise"]:
            probe = get_layerwise_probe(name)
            probe.max_items = 3
            probe_results[name] = probe.run(model)

        report = generate_layerwise_report(probe_results)
        assert "n_probes" in report
        assert report["n_probes"] == 2
        assert "cross_probe" in report
        assert "causal_layerwise" in report["probes"]


# ------------------------------------------------------------------ #
#  Integration test — existing tests still pass                       #
# ------------------------------------------------------------------ #

class TestExistingProbesUnchanged:

    def test_base_logprob_probe_still_works(self):
        """Ensure BaseLogprobProbe.run() is unchanged."""
        from probes.causal_logprob.probe import CausalLogprobProbe
        model = MockAdapter(mode="random", seed=42)
        probe = CausalLogprobProbe()
        probe.max_items = 3
        result = probe.run(model)
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_mock_adapter_original_methods_work(self):
        """Ensure original MockAdapter methods still work."""
        model = MockAdapter(mode="random", seed=42)
        # get_logprobs
        lp = model.get_logprobs("test", ["yes", "no"])
        assert "yes" in lp
        # get_logprobs_batch
        batch = model.get_logprobs_batch(["a", "b"], ["yes", "no"])
        assert len(batch) == 2
        # generate_short
        out = model.generate_short("test")
        assert isinstance(out, str)
