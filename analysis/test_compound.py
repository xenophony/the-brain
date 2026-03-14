"""
Tests for compound analysis, taxonomy, path optimizer, and hierarchical router.

Run: python -m pytest analysis/test_compound.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


# ------------------------------------------------------------------ #
#  Taxonomy tests                                                      #
# ------------------------------------------------------------------ #

class TestTaxonomy:
    def test_get_domain_math(self):
        from analysis.taxonomy import get_domain
        domains = get_domain("math")
        assert "REASONING" in domains

    def test_get_domain_sycophancy_multi(self):
        from analysis.taxonomy import get_domain
        domains = get_domain("sycophancy")
        assert "SOCIAL" in domains
        assert "SAFETY" in domains

    def test_get_domain_unknown(self):
        from analysis.taxonomy import get_domain
        domains = get_domain("nonexistent_probe")
        assert domains == []

    def test_get_probes_in_domain(self):
        from analysis.taxonomy import get_probes_in_domain
        probes = get_probes_in_domain("EXECUTION")
        assert "code" in probes
        assert "tool_use" in probes

    def test_get_probes_in_domain_unknown(self):
        from analysis.taxonomy import get_probes_in_domain
        probes = get_probes_in_domain("NONEXISTENT")
        assert probes == []

    def test_get_all_probe_names_unique(self):
        from analysis.taxonomy import get_all_probe_names
        names = get_all_probe_names()
        assert len(names) == len(set(names)), "Duplicate probe names"
        assert len(names) > 10, "Expected many probes"

    def test_instruction_in_multiple_domains(self):
        from analysis.taxonomy import get_domain
        domains = get_domain("instruction")
        assert "LANGUAGE" in domains
        assert "SAFETY" in domains


# ------------------------------------------------------------------ #
#  Synthetic delta matrices for testing                                 #
# ------------------------------------------------------------------ #

def _make_matrices(n=8):
    """Create synthetic delta matrices for 3 probes on an 8-layer model."""
    np.random.seed(42)
    matrices = {}
    # math: layers 2-4 are beneficial
    m = np.full((n, n + 1), np.nan)
    for i in range(n):
        for j in range(i + 1, n + 1):
            if 2 <= i <= 3 and 4 <= j <= 5:
                m[i, j] = 0.08  # strong improvement
            else:
                m[i, j] = np.random.uniform(-0.02, 0.02)
    matrices["math"] = m

    # code: layers 2-4 also beneficial (synergistic with math)
    c = np.full((n, n + 1), np.nan)
    for i in range(n):
        for j in range(i + 1, n + 1):
            if 2 <= i <= 3 and 4 <= j <= 5:
                c[i, j] = 0.06
            else:
                c[i, j] = np.random.uniform(-0.02, 0.02)
    matrices["code"] = c

    # eq: layers 2-4 are harmful (antagonistic with math)
    e = np.full((n, n + 1), np.nan)
    for i in range(n):
        for j in range(i + 1, n + 1):
            if 2 <= i <= 3 and 4 <= j <= 5:
                e[i, j] = -0.05  # degradation
            else:
                e[i, j] = np.random.uniform(-0.01, 0.01)
    matrices["eq"] = e

    return matrices


# ------------------------------------------------------------------ #
#  Compound analysis tests                                              #
# ------------------------------------------------------------------ #

class TestCompoundAnalysis:
    def test_synergistic_finds_math_code_overlap(self):
        from analysis.compound import find_synergistic_circuits
        matrices = _make_matrices()
        results = find_synergistic_circuits(matrices, min_probes=2, threshold=0.03)
        assert len(results) > 0
        # The (2,5) or (3,5) region should be synergistic for math+code
        probes_found = set()
        for r in results:
            probes_found.update(r["improving_probes"])
        assert "math" in probes_found
        assert "code" in probes_found

    def test_synergistic_empty_with_high_threshold(self):
        from analysis.compound import find_synergistic_circuits
        matrices = _make_matrices()
        results = find_synergistic_circuits(matrices, min_probes=2, threshold=0.5)
        assert len(results) == 0

    def test_antagonistic_finds_math_eq_conflict(self):
        from analysis.compound import find_antagonistic_circuits
        matrices = _make_matrices()
        results = find_antagonistic_circuits(matrices, threshold=0.03)
        assert len(results) > 0
        # Should find math improving while eq degrades
        found_conflict = False
        for r in results:
            if "math" in r["improved"] and "eq" in r["degraded"]:
                found_conflict = True
        assert found_conflict, "Expected math/eq antagonism"

    def test_cascade_finds_overlapping_regions(self):
        from analysis.compound import find_cascade_candidates
        matrices = _make_matrices()
        results = find_cascade_candidates(matrices, threshold=0.03)
        # math and code both improve in (2-3, 4-5) range, so there should be overlap
        assert len(results) > 0

    def test_inhibitory_finds_eq_suppression(self):
        from analysis.compound import find_inhibitory_circuits
        matrices = _make_matrices()
        results = find_inhibitory_circuits(matrices, threshold=0.03)
        # In the 2-4 region, math is strong but mean is dragged down by eq
        assert len(results) > 0

    def test_empty_matrices(self):
        from analysis.compound import (
            find_synergistic_circuits,
            find_antagonistic_circuits,
            find_cascade_candidates,
            find_inhibitory_circuits,
        )
        assert find_synergistic_circuits({}) == []
        assert find_antagonistic_circuits({}) == []
        assert find_cascade_candidates({}) == []
        assert find_inhibitory_circuits({}) == []


# ------------------------------------------------------------------ #
#  Path optimizer tests                                                 #
# ------------------------------------------------------------------ #

class TestPathOptimizer:
    def test_empty_weights_returns_identity(self):
        from analysis.path_optimizer import PathOptimizer
        optimizer = PathOptimizer({"synergistic": [], "antagonistic": [], "inhibitory": []}, n_layers=8)
        path = optimizer.recommend_path({})
        assert path == list(range(8))

    def test_weighted_path_includes_synergistic(self):
        from analysis.path_optimizer import PathOptimizer
        compound = {
            "synergistic": [{
                "i": 2, "j": 5,
                "improving_probes": ["math", "code"],
                "mean_delta": 0.07,
                "n_improving": 2,
            }],
            "antagonistic": [],
            "inhibitory": [],
        }
        optimizer = PathOptimizer(compound, n_layers=8)
        path = optimizer.recommend_path({"math": 1.0})
        # Layers 2-4 should appear more than once (duplicated)
        assert path.count(2) >= 2 or path.count(3) >= 2 or path.count(4) >= 2

    def test_router_features_returns_all_layers(self):
        from analysis.path_optimizer import PathOptimizer
        matrices = _make_matrices()
        optimizer = PathOptimizer({}, n_layers=8)
        features = optimizer.recommend_router_features(matrices)
        assert len(features) == 8
        # All values should be non-negative
        assert all(v >= 0 for v in features.values())

    def test_router_features_empty(self):
        from analysis.path_optimizer import PathOptimizer
        optimizer = PathOptimizer({}, n_layers=8)
        features = optimizer.recommend_router_features({})
        assert features == {}


# ------------------------------------------------------------------ #
#  Hierarchical router tests                                            #
# ------------------------------------------------------------------ #

class TestHierarchicalRouter:
    def test_classify_math_prompt(self):
        from router.hierarchical import HierarchicalRouter
        router = HierarchicalRouter(n_layers=32)
        result = router.classify("Calculate the sum of 17 and 23")
        assert result["domain"] == "REASONING"
        assert result["method"] == "heuristic"
        assert "domain_scores" in result
        assert "path" in result

    def test_classify_code_prompt(self):
        from router.hierarchical import HierarchicalRouter
        router = HierarchicalRouter(n_layers=32)
        result = router.classify("Write a function to implement binary search")
        assert result["domain"] == "EXECUTION"

    def test_classify_spatial_prompt(self):
        from router.hierarchical import HierarchicalRouter
        router = HierarchicalRouter(n_layers=32)
        result = router.classify("What is at position B3 on the grid board?")
        assert result["domain"] == "SPATIAL"

    def test_classify_empty_defaults_to_reasoning(self):
        from router.hierarchical import HierarchicalRouter
        router = HierarchicalRouter(n_layers=32)
        result = router.classify("")
        assert result["domain"] == "REASONING"

    def test_path_length_matches_layers(self):
        from router.hierarchical import HierarchicalRouter
        router = HierarchicalRouter(n_layers=16)
        result = router.classify("solve this equation")
        # Path should have at least n_layers entries
        assert len(result["path"]) >= 16


class TestFuzzyDomainMatcher:
    def test_basic_matching(self):
        from router.hierarchical import FuzzyDomainMatcher
        matcher = FuzzyDomainMatcher({
            "math": ["calculate the sum", "solve the equation", "compute the product"],
            "code": ["write a function", "implement the algorithm", "debug this program"],
        })
        scores = matcher.match("calculate the product of these numbers")
        assert "math" in scores
        assert "code" in scores
        assert scores["math"] > scores["code"]

    def test_no_match_returns_uniform(self):
        from router.hierarchical import FuzzyDomainMatcher
        matcher = FuzzyDomainMatcher({
            "math": ["calculate sum"],
            "code": ["write function"],
        })
        scores = matcher.match("xyzzy plugh")
        # Should be uniform when nothing matches
        assert abs(scores["math"] - scores["code"]) < 0.01

    def test_scores_sum_to_one(self):
        from router.hierarchical import FuzzyDomainMatcher
        matcher = FuzzyDomainMatcher({
            "math": ["calculate the sum of numbers"],
            "code": ["write a function to implement"],
            "eq": ["understand the emotional tone"],
        })
        scores = matcher.match("calculate and implement the function")
        total = sum(scores.values())
        assert abs(total - 1.0) < 0.01, f"Scores sum to {total}, expected 1.0"


# ------------------------------------------------------------------ #
#  Difficulty-aware analysis tests                                     #
# ------------------------------------------------------------------ #

class TestDifficultyAnalysis:
    def _make_synthetic_matrices(self):
        """Create synthetic easy/hard matrices for testing."""
        n = 6
        easy = np.full((n, n + 1), np.nan)
        hard = np.full((n, n + 1), np.nan)
        # Fast-path circuit at (1,3): easy improves, hard degrades
        easy[1, 3] = 0.15
        hard[1, 3] = -0.05
        # Complexity circuit at (2,5): hard improves, easy flat
        easy[2, 5] = 0.0
        hard[2, 5] = 0.12
        # General circuit at (3,5): both improve
        easy[3, 5] = 0.08
        hard[3, 5] = 0.10
        # Neutral at (0,2): both flat
        easy[0, 2] = 0.01
        hard[0, 2] = 0.01
        return easy, hard

    def test_find_fastpath_circuits(self):
        from analysis.heatmap import find_fastpath_circuits
        easy, hard = self._make_synthetic_matrices()
        fast = find_fastpath_circuits(easy, hard, threshold=0.03)
        assert len(fast) == 1
        assert fast[0]["i"] == 1
        assert fast[0]["j"] == 3
        assert fast[0]["easy_delta"] > 0.1
        assert fast[0]["hard_delta"] < 0

    def test_find_complexity_circuits(self):
        from analysis.heatmap import find_complexity_circuits
        easy, hard = self._make_synthetic_matrices()
        complexity = find_complexity_circuits(easy, hard, threshold=0.03)
        assert len(complexity) == 1
        assert complexity[0]["i"] == 2
        assert complexity[0]["j"] == 5
        assert complexity[0]["hard_delta"] > 0.1

    def test_build_difficulty_matrices_structure(self):
        from analysis.heatmap import build_difficulty_matrices
        # Create minimal results with difficulty scores
        results = [
            {"i": 0, "j": 0, "probe_scores": {"math": 0.5, "math_easy": 0.7, "math_hard": 0.3},
             "probe_deltas": {"math": 0.0}},
            {"i": 1, "j": 3, "probe_scores": {"math": 0.6, "math_easy": 0.8, "math_hard": 0.4},
             "probe_deltas": {"math": 0.1}},
        ]
        matrices = build_difficulty_matrices(results, "math", 4)
        assert "overall" in matrices
        assert "easy" in matrices
        assert "hard" in matrices
        assert "diff" in matrices
        assert matrices["overall"].shape == (4, 5)
        # Check (1,3) has correct values
        assert not np.isnan(matrices["easy"][1, 3])
        assert not np.isnan(matrices["hard"][1, 3])

    def test_fastpath_empty_when_no_signal(self):
        from analysis.heatmap import find_fastpath_circuits
        n = 4
        easy = np.full((n, n + 1), 0.01)
        hard = np.full((n, n + 1), 0.01)
        fast = find_fastpath_circuits(easy, hard, threshold=0.03)
        assert len(fast) == 0

    def test_complexity_empty_when_no_signal(self):
        from analysis.heatmap import find_complexity_circuits
        n = 4
        easy = np.full((n, n + 1), 0.01)
        hard = np.full((n, n + 1), 0.01)
        complexity = find_complexity_circuits(easy, hard, threshold=0.03)
        assert len(complexity) == 0
