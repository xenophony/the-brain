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
