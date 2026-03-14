"""
Tests for API adapters, calibration analysis, cost estimator, and baseline runner.

API tests are marked with @pytest.mark.api and skipped when keys are not set.
All other tests run without any API keys.

Run: python -m pytest sweep/test_api_adapters.py -v
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from sweep.api_adapters import (
    BaseAPIAdapter,
    ClaudeAdapter,
    GeminiAdapter,
    GroqAdapter,
    TogetherAdapter,
    OpenRouterAdapter,
    ADAPTER_MAP,
    _ENV_KEYS,
    available_providers,
    _retry_with_backoff,
)
from analysis.calibration import (
    check_ceiling_effects,
    check_floor_effects,
    check_dynamic_range,
    check_orthogonality,
    generate_calibration_report,
    _pearson_correlation,
)
from scripts.estimate_cost import (
    estimate_baseline_cost,
    estimate_sweep_cost,
    ALL_MODELS,
    PRICING,
)
from scripts.run_baselines import MODEL_REGISTRY, ALL_PROBES


# ===================================================================
#  Synthetic baseline data for calibration tests
# ===================================================================

SYNTHETIC_BASELINES = {
    "model_a": {
        "math":       {"score": 0.90, "n_items": 12, "latency_seconds": 5.0, "error": None},
        "eq":         {"score": 0.50, "n_items": 12, "latency_seconds": 3.0, "error": None},
        "code":       {"score": 0.70, "n_items": 7,  "latency_seconds": 8.0, "error": None},
        "spatial":    {"score": 0.30, "n_items": 20, "latency_seconds": 6.0, "error": None},
        "factual":    {"score": 0.60, "n_items": 14, "latency_seconds": 4.0, "error": None},
    },
    "model_b": {
        "math":       {"score": 0.60, "n_items": 12, "latency_seconds": 4.0, "error": None},
        "eq":         {"score": 0.70, "n_items": 12, "latency_seconds": 2.5, "error": None},
        "code":       {"score": 0.40, "n_items": 7,  "latency_seconds": 7.0, "error": None},
        "spatial":    {"score": 0.55, "n_items": 20, "latency_seconds": 5.0, "error": None},
        "factual":    {"score": 0.80, "n_items": 14, "latency_seconds": 3.5, "error": None},
    },
    "model_c": {
        "math":       {"score": 0.75, "n_items": 12, "latency_seconds": 6.0, "error": None},
        "eq":         {"score": 0.60, "n_items": 12, "latency_seconds": 3.0, "error": None},
        "code":       {"score": 0.55, "n_items": 7,  "latency_seconds": 9.0, "error": None},
        "spatial":    {"score": 0.40, "n_items": 20, "latency_seconds": 6.5, "error": None},
        "factual":    {"score": 0.70, "n_items": 14, "latency_seconds": 4.5, "error": None},
    },
}


# ===================================================================
#  Adapter interface tests (no API keys needed)
# ===================================================================

class TestAdapterMissingKeys:
    """Test that each adapter raises ValueError when API key is missing."""

    def test_claude_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Also need to mock the import so it doesn't fail on ImportError
            with patch.dict("sys.modules", {"anthropic": MagicMock()}):
                with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                    ClaudeAdapter()

    def test_gemini_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            mock_genai = MagicMock()
            with patch.dict("sys.modules", {
                "google": MagicMock(),
                "google.generativeai": mock_genai,
            }):
                with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                    GeminiAdapter()

    def test_groq_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict("sys.modules", {"groq": MagicMock()}):
                with pytest.raises(ValueError, match="GROQ_API_KEY"):
                    GroqAdapter()

    def test_together_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict("sys.modules", {"together": MagicMock()}):
                with pytest.raises(ValueError, match="TOGETHER_API_KEY"):
                    TogetherAdapter()

    def test_openrouter_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            mock_openai = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_openai}):
                with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                    OpenRouterAdapter()


class TestAdapterInterface:
    """Test that all adapters expose the same interface as MockAdapter."""

    REQUIRED_ATTRS = ["generate_short", "get_logprobs", "set_layer_path", "num_layers"]

    def test_base_class_has_interface(self):
        for attr in self.REQUIRED_ATTRS:
            assert hasattr(BaseAPIAdapter, attr), f"BaseAPIAdapter missing '{attr}'"

    def test_all_adapters_in_map(self):
        assert "claude" in ADAPTER_MAP
        assert "gemini" in ADAPTER_MAP
        assert "groq" in ADAPTER_MAP
        assert "together" in ADAPTER_MAP
        assert "openrouter" in ADAPTER_MAP

    def test_adapter_classes_have_interface(self):
        for name, cls in ADAPTER_MAP.items():
            for attr in self.REQUIRED_ATTRS:
                assert hasattr(cls, attr), f"{name} adapter missing '{attr}'"

    def test_env_keys_complete(self):
        for provider in ADAPTER_MAP:
            assert provider in _ENV_KEYS, f"No env key mapping for '{provider}'"

    def test_available_providers_empty_without_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            assert available_providers() == []


class TestRetryWithBackoff:
    """Test the retry helper."""

    def test_succeeds_immediately(self):
        result = _retry_with_backoff(lambda: 42)
        assert result == 42

    def test_retries_on_rate_limit(self):
        call_count = 0
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                exc = Exception("429 rate limit exceeded")
                raise exc
            return "ok"
        result = _retry_with_backoff(flaky, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        def always_fail():
            raise Exception("429 rate limit")
        with pytest.raises(Exception, match="429"):
            _retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)

    def test_non_retryable_error_raises_immediately(self):
        call_count = 0
        def bad():
            nonlocal call_count
            call_count += 1
            raise ValueError("invalid input")
        with pytest.raises(ValueError):
            _retry_with_backoff(bad, max_retries=3, base_delay=0.01)
        assert call_count == 1


# ===================================================================
#  Calibration function tests (no API keys needed)
# ===================================================================

class TestCeilingEffects:
    def test_detects_ceiling(self):
        findings = check_ceiling_effects(SYNTHETIC_BASELINES, threshold=0.85)
        probes_flagged = {f["probe"] for f in findings}
        assert "math" in probes_flagged  # model_a scores 0.90

    def test_no_false_positives(self):
        findings = check_ceiling_effects(SYNTHETIC_BASELINES, threshold=0.95)
        # No score above 0.95
        assert len(findings) == 0


class TestFloorEffects:
    def test_no_floor_in_synthetic(self):
        findings = check_floor_effects(SYNTHETIC_BASELINES, threshold=0.20)
        assert len(findings) == 0

    def test_detects_floor(self):
        data = {
            "m1": {"hard_probe": {"score": 0.05, "n_items": 10, "latency_seconds": 1, "error": None}},
            "m2": {"hard_probe": {"score": 0.10, "n_items": 10, "latency_seconds": 1, "error": None}},
            "m3": {"hard_probe": {"score": 0.15, "n_items": 10, "latency_seconds": 1, "error": None}},
        }
        findings = check_floor_effects(data, threshold=0.20)
        assert len(findings) == 1
        assert findings[0]["probe"] == "hard_probe"


class TestDynamicRange:
    def test_reports_range(self):
        results = check_dynamic_range(SYNTHETIC_BASELINES)
        assert len(results) == 5  # 5 probes
        for r in results:
            assert "probe" in r
            assert "range" in r
            assert "ok" in r

    def test_narrow_range_flagged(self):
        data = {
            "m1": {"narrow": {"score": 0.50, "n_items": 1, "latency_seconds": 1, "error": None}},
            "m2": {"narrow": {"score": 0.52, "n_items": 1, "latency_seconds": 1, "error": None}},
            "m3": {"narrow": {"score": 0.51, "n_items": 1, "latency_seconds": 1, "error": None}},
        }
        results = check_dynamic_range(data, min_range=0.25)
        assert len(results) == 1
        assert not results[0]["ok"]


class TestOrthogonality:
    def test_returns_pairs(self):
        all_pairs, flagged = check_orthogonality(SYNTHETIC_BASELINES)
        # 5 probes -> C(5,2) = 10 pairs
        assert len(all_pairs) == 10
        for p in all_pairs:
            assert "probe_a" in p
            assert "probe_b" in p
            assert "correlation" in p

    def test_perfect_correlation_flagged(self):
        data = {
            "m1": {
                "a": {"score": 0.1, "n_items": 1, "latency_seconds": 1, "error": None},
                "b": {"score": 0.1, "n_items": 1, "latency_seconds": 1, "error": None},
            },
            "m2": {
                "a": {"score": 0.5, "n_items": 1, "latency_seconds": 1, "error": None},
                "b": {"score": 0.5, "n_items": 1, "latency_seconds": 1, "error": None},
            },
            "m3": {
                "a": {"score": 0.9, "n_items": 1, "latency_seconds": 1, "error": None},
                "b": {"score": 0.9, "n_items": 1, "latency_seconds": 1, "error": None},
            },
        }
        _, flagged = check_orthogonality(data, threshold=0.7)
        assert len(flagged) == 1
        assert flagged[0]["correlation"] > 0.99


class TestPearsonCorrelation:
    def test_perfect_positive(self):
        r = _pearson_correlation([1, 2, 3], [2, 4, 6])
        assert abs(r - 1.0) < 0.001

    def test_perfect_negative(self):
        r = _pearson_correlation([1, 2, 3], [6, 4, 2])
        assert abs(r - (-1.0)) < 0.001

    def test_zero_variance(self):
        r = _pearson_correlation([5, 5, 5], [1, 2, 3])
        assert r == 0.0


class TestCalibrationReport:
    def test_generates_report(self, tmp_path):
        report_path = tmp_path / "CALIBRATION_REPORT.md"
        report = generate_calibration_report(SYNTHETIC_BASELINES, str(report_path))
        assert report_path.exists()
        assert "# Calibration Report" in report
        assert "Score Summary" in report
        assert "Ceiling Effects" in report
        assert "Floor Effects" in report
        assert "Dynamic Range" in report
        assert "Orthogonality" in report


# ===================================================================
#  Cost estimator tests
# ===================================================================

class TestCostEstimator:
    def test_baseline_cost_structure(self):
        est = estimate_baseline_cost(["claude-sonnet"])
        assert "breakdown" in est
        assert "total_usd" in est
        assert "claude-sonnet" in est["breakdown"]
        assert est["total_usd"] > 0

    def test_baseline_cost_all_models(self):
        est = estimate_baseline_cost(ALL_MODELS)
        assert len(est["breakdown"]) == len(ALL_MODELS)
        assert est["total_usd"] > 0

    def test_sweep_cost(self):
        est = estimate_sweep_cost(n_layers=32, max_block=8)
        assert est["n_configs"] > 0
        assert est["total_hours"] > 0
        assert est["total_usd"] > 0

    def test_sweep_cost_full(self):
        est = estimate_sweep_cost(n_layers=48)
        assert est["n_configs"] > 100  # 48 layers = lots of configs
        assert est["n_layers"] == 48

    def test_unknown_model_handled(self):
        est = estimate_baseline_cost(["unknown-model-xyz"])
        assert "unknown-model-xyz" in est["breakdown"]
        assert "note" in est["breakdown"]["unknown-model-xyz"]


# ===================================================================
#  Baseline runner validation tests
# ===================================================================

class TestModelRegistry:
    def test_all_models_have_provider(self):
        for name, (provider, model_id) in MODEL_REGISTRY.items():
            assert provider in ADAPTER_MAP, f"{name}: provider '{provider}' not in ADAPTER_MAP"
            assert isinstance(model_id, str) and len(model_id) > 0

    def test_registry_completeness(self):
        assert len(MODEL_REGISTRY) >= 5  # at least the 5 defined models

    def test_probe_list_matches_registered(self):
        from probes.registry import list_probes
        registered = list_probes()
        for p in ALL_PROBES:
            assert p in registered, f"Probe '{p}' in ALL_PROBES but not registered"

    def test_all_probes_count(self):
        assert len(ALL_PROBES) == 20


# ===================================================================
#  API integration tests (skipped without keys)
# ===================================================================

_has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
_has_google_key = bool(os.environ.get("GOOGLE_API_KEY"))
_has_groq_key = bool(os.environ.get("GROQ_API_KEY"))
_has_together_key = bool(os.environ.get("TOGETHER_API_KEY"))
_has_openrouter_key = bool(os.environ.get("OPENROUTER_API_KEY"))


@pytest.mark.api
@pytest.mark.skipif(not _has_anthropic_key, reason="ANTHROPIC_API_KEY not set")
class TestClaudeAdapterLive:
    def test_instantiation(self):
        adapter = ClaudeAdapter()
        assert adapter.num_layers is None
        assert adapter.model == "claude-sonnet-4-20250514"

    def test_generate_short(self):
        adapter = ClaudeAdapter()
        result = adapter.generate_short("What is 2+2? Answer with just the number.", max_new_tokens=5)
        assert "4" in result

    def test_set_layer_path_noop(self):
        adapter = ClaudeAdapter()
        adapter.set_layer_path([0, 1, 2])  # should not raise

    def test_get_logprobs_returns_dict(self):
        adapter = ClaudeAdapter()
        result = adapter.get_logprobs("test")
        assert isinstance(result, dict)


@pytest.mark.api
@pytest.mark.skipif(not _has_google_key, reason="GOOGLE_API_KEY not set")
class TestGeminiAdapterLive:
    def test_instantiation(self):
        adapter = GeminiAdapter()
        assert adapter.num_layers is None

    def test_generate_short(self):
        adapter = GeminiAdapter()
        result = adapter.generate_short("What is 2+2? Answer with just the number.", max_new_tokens=5)
        assert "4" in result


@pytest.mark.api
@pytest.mark.skipif(not _has_groq_key, reason="GROQ_API_KEY not set")
class TestGroqAdapterLive:
    def test_instantiation(self):
        adapter = GroqAdapter()
        assert adapter.num_layers is None

    def test_generate_short(self):
        adapter = GroqAdapter()
        result = adapter.generate_short("What is 2+2? Answer with just the number.", max_new_tokens=5)
        assert "4" in result


@pytest.mark.api
@pytest.mark.skipif(not _has_together_key, reason="TOGETHER_API_KEY not set")
class TestTogetherAdapterLive:
    def test_instantiation(self):
        adapter = TogetherAdapter()
        assert adapter.num_layers is None

    def test_generate_short(self):
        adapter = TogetherAdapter()
        result = adapter.generate_short("What is 2+2? Answer with just the number.", max_new_tokens=5)
        assert "4" in result


@pytest.mark.api
@pytest.mark.skipif(not _has_openrouter_key, reason="OPENROUTER_API_KEY not set")
class TestOpenRouterAdapterLive:
    def test_instantiation(self):
        adapter = OpenRouterAdapter()
        assert adapter.num_layers is None
        assert adapter.model == "meta-llama/llama-3.1-8b-instruct"

    def test_generate_short(self):
        adapter = OpenRouterAdapter()
        result = adapter.generate_short("What is 2+2? Answer with just the number.", max_new_tokens=5)
        assert "4" in result

    def test_set_layer_path_noop(self):
        adapter = OpenRouterAdapter()
        adapter.set_layer_path([0, 1, 2])  # should not raise

    def test_get_logprobs_returns_dict(self):
        adapter = OpenRouterAdapter()
        result = adapter.get_logprobs("test")
        assert isinstance(result, dict)
