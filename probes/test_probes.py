"""
Tests for all probes using MockAdapter, plus sweep runner tests.

Run: python -m pytest probes/test_probes.py -v
"""

import sys
import json
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from sweep.mock_adapter import MockAdapter
from probes.registry import get_probe


# ------------------------------------------------------------------ #
#  Perfect mode tests — each probe should score > 0.9                 #
# ------------------------------------------------------------------ #

@pytest.fixture
def perfect_model():
    return MockAdapter(mode="perfect", seed=42)


@pytest.fixture
def terrible_model():
    return MockAdapter(mode="terrible", seed=42)


class TestMathProbe:
    def test_perfect_score(self, perfect_model):
        probe = get_probe("math")
        score = probe.run(perfect_model)
        assert score > 0.9, f"Math probe perfect score too low: {score}"

    def test_terrible_score(self, terrible_model):
        probe = get_probe("math")
        score = probe.run(terrible_model)
        assert score < 0.3, f"Math probe terrible score too high: {score}"


class TestCodeProbe:
    def test_perfect_score(self, perfect_model):
        probe = get_probe("code")
        score = probe.run(perfect_model)
        assert score > 0.9, f"Code probe perfect score too low: {score}"

    def test_terrible_score(self, terrible_model):
        probe = get_probe("code")
        score = probe.run(terrible_model)
        assert score < 0.3, f"Code probe terrible score too high: {score}"


class TestEQProbe:
    def test_perfect_score(self, perfect_model):
        probe = get_probe("eq")
        score = probe.run(perfect_model)
        assert score > 0.9, f"EQ probe perfect score too low: {score}"

    def test_terrible_score(self, terrible_model):
        probe = get_probe("eq")
        score = probe.run(terrible_model)
        assert score < 0.3, f"EQ probe terrible score too high: {score}"


class TestFactualProbe:
    def test_perfect_score(self, perfect_model):
        probe = get_probe("factual")
        score = probe.run(perfect_model)
        assert score > 0.9, f"Factual probe perfect score too low: {score}"

    def test_terrible_score(self, terrible_model):
        probe = get_probe("factual")
        score = probe.run(terrible_model)
        assert score < 0.3, f"Factual probe terrible score too high: {score}"


class TestLanguageProbe:
    def test_perfect_score(self, perfect_model):
        probe = get_probe("language")
        score = probe.run(perfect_model)
        assert score > 0.9, f"Language probe perfect score too low: {score}"

    def test_terrible_score(self, terrible_model):
        probe = get_probe("language")
        score = probe.run(terrible_model)
        assert score < 0.3, f"Language probe terrible score too high: {score}"


class TestToolUseProbe:
    def test_perfect_score(self, perfect_model):
        probe = get_probe("tool_use")
        score = probe.run(perfect_model)
        assert score > 0.9, f"Tool use probe perfect score too low: {score}"

    def test_terrible_score(self, terrible_model):
        probe = get_probe("tool_use")
        score = probe.run(terrible_model)
        assert score < 0.3, f"Tool use probe terrible score too high: {score}"


class TestHolisticProbe:
    def test_perfect_score(self, perfect_model):
        probe = get_probe("holistic")
        score = probe.run(perfect_model)
        assert score > 0.9, f"Holistic probe perfect score too low: {score}"

    def test_terrible_score(self, terrible_model):
        probe = get_probe("holistic")
        score = probe.run(terrible_model)
        assert score < 0.3, f"Holistic probe terrible score too high: {score}"


class TestPlanningProbe:
    def test_perfect_score(self, perfect_model):
        probe = get_probe("planning")
        score = probe.run(perfect_model)
        assert score > 0.9, f"Planning probe perfect score too low: {score}"

    def test_terrible_score(self, terrible_model):
        probe = get_probe("planning")
        score = probe.run(terrible_model)
        assert score < 0.3, f"Planning probe terrible score too high: {score}"


class TestInstructionProbe:
    def test_perfect_score(self, perfect_model):
        probe = get_probe("instruction")
        score = probe.run(perfect_model)
        assert score > 0.9, f"Instruction probe perfect score too low: {score}"

    def test_terrible_score(self, terrible_model):
        probe = get_probe("instruction")
        score = probe.run(terrible_model)
        assert score < 0.3, f"Instruction probe terrible score too high: {score}"


class TestSpatialProbe:
    def test_perfect_score(self, perfect_model):
        probe = get_probe("spatial")
        score = probe.run(perfect_model)
        # Spatial is harder — mock may not find optimal cell but should be > 0
        assert score > 0.0, f"Spatial probe perfect score is zero"

    def test_terrible_score(self, terrible_model):
        probe = get_probe("spatial")
        score = probe.run(terrible_model)
        assert score < 0.3, f"Spatial probe terrible score too high: {score}"

    def test_board_generation_determinism(self):
        """Same seed must produce same boards."""
        from probes.spatial.probe import generate_boards
        boards1 = generate_boards(seed=42, n=5)
        boards2 = generate_boards(seed=42, n=5)
        assert len(boards1) == len(boards2)
        for (b1, s1, _), (b2, s2, _) in zip(boards1, boards2):
            assert b1 == b2, "Boards differ with same seed"
            assert s1 == s2, "Ship cells differ with same seed"

    def test_board_generation_different_seeds(self):
        """Different seeds must produce different boards."""
        from probes.spatial.probe import generate_boards
        boards1 = generate_boards(seed=42, n=5)
        boards2 = generate_boards(seed=99, n=5)
        # At least one board should differ
        any_different = False
        for (b1, _, _), (b2, _, _) in zip(boards1, boards2):
            if b1 != b2:
                any_different = True
                break
        assert any_different, "Different seeds produced identical boards"


# ------------------------------------------------------------------ #
#  Individual scoring function tests                                   #
# ------------------------------------------------------------------ #

class TestScoringFunctions:
    def test_math_scoring(self):
        from probes.math.probe import score_math
        assert score_math("391", 391) == 1.0
        assert score_math("17 * 23 = 391", 391) == 1.0  # extracts last number
        assert score_math("392", 391) == 0.9  # <1% error
        assert score_math("banana", 391) == 0.0

    def test_factual_scoring(self):
        from probes.factual.probe import score_factual
        q_num = {"answer": "3422", "type": "number"}
        assert score_factual("3422", q_num) == 1.0
        assert score_factual("3423", q_num) == 0.5  # off by one
        assert score_factual("banana", q_num) == 0.0

        q_word = {"answer": "osmium", "type": "word", "alternates": []}
        assert score_factual("osmium", q_word) == 1.0
        assert score_factual("Osmium", q_word) == 1.0
        assert score_factual("iron", q_word) == 0.0

    def test_language_scoring(self):
        from probes.language.probe import score_language
        assert score_language("grammatical", "grammatical") == 1.0
        assert score_language("ungrammatical", "ungrammatical") == 1.0
        assert score_language("grammatical", "ungrammatical") == 0.0
        assert score_language("banana", "grammatical") == 0.0

    def test_tool_use_scoring(self):
        from probes.tool_use.probe import score_tool_use
        assert score_tool_use("calculator", "calculator") == 1.0
        assert score_tool_use("The best tool is calculator", "calculator") == 1.0
        assert score_tool_use("banana", "calculator") == 0.0

    def test_analogy_scoring(self):
        from probes.holistic.probe import score_analogy
        assert score_analogy("composer", ["composer", "musician"]) == 1.0
        assert score_analogy("musician", ["composer", "musician"]) == 1.0
        assert score_analogy("banana", ["composer", "musician"]) == 0.0

    def test_planning_scoring(self):
        from probes.planning.probe import score_planning
        assert score_planning("ABCD", "ABCD") == 1.0
        assert score_planning("DCBA", "ABCD") == 0.0
        # Partial: "ABDC" vs "ABCD" — AB correct, BD wrong, DC wrong
        score = score_planning("ABDC", "ABCD")
        assert 0.0 < score < 1.0

    def test_instruction_checkers(self):
        from probes.instruction.probe import (
            _has_uppercase, _has_number, _ends_with_exclamation,
            _ends_with_period, _word_count, _no_vowels,
            _contains_color, _contains_animal, _is_palindrome_word,
            _all_words_same_first_letter,
        )
        assert _has_uppercase("HELLO WORLD")
        assert not _has_uppercase("Hello World")
        assert _has_number("abc 42 def")
        assert not _has_number("no digits here")
        assert _ends_with_exclamation("hello!")
        assert not _ends_with_exclamation("hello.")
        assert _ends_with_period("hello.")
        assert _word_count("one two three", 3)
        assert not _word_count("one two", 3)
        assert _no_vowels("RHYTHMS 7")
        assert not _no_vowels("HELLO")
        assert _contains_color("the red fox")
        assert _contains_animal("the red fox")
        assert _is_palindrome_word("racecar is great")
        assert not _is_palindrome_word("hello world")
        assert _all_words_same_first_letter("Peter Piper Picked")
        assert not _all_words_same_first_letter("Peter Paul Mary")

    def test_spatial_board_ascii_format(self):
        from probes.spatial.probe import generate_boards, board_to_ascii
        boards = generate_boards(seed=42, n=1)
        board, _, _ = boards[0]
        ascii_board = board_to_ascii(board)
        # Check format
        assert "+--------------------+" in ascii_board
        assert ". = unknown, H = hit, M = miss" in ascii_board
        lines = [l for l in ascii_board.split("\n") if l.startswith("|")]
        assert len(lines) == 10

    def test_spatial_probability_density(self):
        from probes.spatial.probe import compute_probability_density, _make_empty_board
        # Simple test: board with one hit should produce non-zero density
        # Uses only visible board state — no ground truth positions needed
        board = _make_empty_board()
        board[5][5] = 'H'
        density = compute_probability_density(board)
        assert len(density) > 0
        # Cells adjacent to the hit should have higher density
        assert density.get((5, 6), 0) > 0 or density.get((5, 4), 0) > 0


# ------------------------------------------------------------------ #
#  Sweep runner tests — skip mode, build paths, optimized path         #
# ------------------------------------------------------------------ #

class TestSweepRunner:
    def test_build_skip_path_identity(self):
        """(0,0) skip path should be identity."""
        from sweep.runner import SweepRunner, SweepConfig
        config = SweepConfig(model_path="mock", output_dir="/tmp/test_skip",
                             probe_names=["math"])
        runner = SweepRunner(config, adapter_class=MockAdapter)
        runner.load_model()
        path = runner.build_skip_path(0, 0)
        assert path == list(range(runner.n_layers))

    def test_build_skip_path_removes_layers(self):
        """Skip path should exclude layers i..j-1."""
        from sweep.runner import SweepRunner, SweepConfig
        config = SweepConfig(model_path="mock", output_dir="/tmp/test_skip",
                             probe_names=["math"])
        runner = SweepRunner(config, adapter_class=MockAdapter)
        runner.load_model()
        # Skip layers 2,3,4 (i=2, j=5)
        path = runner.build_skip_path(2, 5)
        assert 2 not in path
        assert 3 not in path
        assert 4 not in path
        assert 0 in path
        assert 1 in path
        assert 5 in path
        expected = [0, 1] + list(range(5, runner.n_layers))
        assert path == expected

    def test_build_layer_path_duplicates_layers(self):
        """Duplicate path should include layers i..j-1 twice."""
        from sweep.runner import SweepRunner, SweepConfig
        config = SweepConfig(model_path="mock", output_dir="/tmp/test_dup",
                             probe_names=["math"])
        runner = SweepRunner(config, adapter_class=MockAdapter)
        runner.load_model()
        path = runner.build_layer_path(2, 5)
        # Layers 2,3,4 should appear twice
        assert path.count(2) == 2
        assert path.count(3) == 2
        assert path.count(4) == 2
        # Layer 0,1 appear once
        assert path.count(0) == 1
        assert path.count(1) == 1

    def test_skip_sweep_runs(self, tmp_path):
        """Skip mode sweep should complete and produce results."""
        from sweep.runner import SweepRunner, SweepConfig
        config = SweepConfig(
            model_path="mock", output_dir=str(tmp_path),
            probe_names=["math"], max_layers=4, max_block_size=2,
            mode="skip",
        )
        runner = SweepRunner(config, adapter_class=MockAdapter)
        results = runner.run()
        assert len(results) > 1  # baseline + configs
        # All non-baseline results should have mode="skip"
        for r in results:
            assert r.mode == "skip"

    def test_both_mode_produces_separate_files(self, tmp_path):
        """Mode=both should produce both result files."""
        from sweep.runner import SweepRunner, SweepConfig
        config = SweepConfig(
            model_path="mock", output_dir=str(tmp_path),
            probe_names=["math"], max_layers=4, max_block_size=2,
            mode="both",
        )
        runner = SweepRunner(config, adapter_class=MockAdapter)
        runner.run()
        assert (tmp_path / "sweep_results_duplicate.json").exists()
        assert (tmp_path / "sweep_results_skip.json").exists()
        assert (tmp_path / "sweep_results.json").exists()

        # Check each file has correct mode
        with open(tmp_path / "sweep_results_duplicate.json") as f:
            dup = json.load(f)
        with open(tmp_path / "sweep_results_skip.json") as f:
            skip = json.load(f)
        for r in dup:
            assert r["mode"] == "duplicate"
        for r in skip:
            assert r["mode"] == "skip"

    def test_skip_path_length(self):
        """Skip path should be shorter than original."""
        from sweep.runner import SweepRunner, SweepConfig
        config = SweepConfig(model_path="mock", output_dir="/tmp/test_skip",
                             probe_names=["math"])
        runner = SweepRunner(config, adapter_class=MockAdapter)
        runner.load_model()
        N = runner.n_layers
        original = runner.build_layer_path(0, 0)
        skip = runner.build_skip_path(2, 5)
        assert len(original) == N
        assert len(skip) == N - 3  # removed 3 layers


class TestBuildOptimizedPath:
    def test_no_modifications(self):
        """No skip or dup regions should return identity."""
        from sweep.runner import build_optimized_path
        path = build_optimized_path(10, [], [])
        assert path == list(range(10))

    def test_skip_only(self):
        """Skip regions should remove layers."""
        from sweep.runner import build_optimized_path
        path = build_optimized_path(10, skip_regions=[(3, 6)], duplicate_regions=[])
        assert 3 not in path
        assert 4 not in path
        assert 5 not in path
        assert path == [0, 1, 2, 6, 7, 8, 9]

    def test_duplicate_only(self):
        """Duplicate regions should repeat layers."""
        from sweep.runner import build_optimized_path
        path = build_optimized_path(10, skip_regions=[], duplicate_regions=[(2, 4)])
        assert path.count(2) == 2
        assert path.count(3) == 2
        assert path.count(0) == 1

    def test_skip_takes_priority(self):
        """Overlapping skip and dup — skip wins."""
        from sweep.runner import build_optimized_path
        path = build_optimized_path(10, skip_regions=[(3, 5)], duplicate_regions=[(3, 6)])
        # Layers 3,4 should be skipped (not duplicated)
        assert 3 not in path
        assert 4 not in path
        # Layer 5 should be duplicated (not in skip range)
        assert path.count(5) == 2

    def test_combined_path_structure(self):
        """Combined skip+dup should produce valid layer sequence."""
        from sweep.runner import build_optimized_path
        path = build_optimized_path(
            n_layers=12,
            skip_regions=[(8, 10)],
            duplicate_regions=[(2, 4)],
        )
        # Layers 8,9 removed, layers 2,3 doubled
        assert 8 not in path
        assert 9 not in path
        assert path.count(2) == 2
        assert path.count(3) == 2
        # First and last layers present
        assert 0 in path
        assert 11 in path


class TestOverlayAnalysis:
    def test_classify_region(self):
        from analysis.heatmap import classify_region
        assert classify_region(0.1, 0.0, threshold=0.03) == "double"
        assert classify_region(0.0, 0.1, threshold=0.03) == "skip"
        assert classify_region(0.0, 0.0, threshold=0.03) == "neutral"
        assert classify_region(0.1, 0.1, threshold=0.03) == "ambiguous"
        assert classify_region(0.01, 0.01, threshold=0.03) == "neutral"

    def test_overlay_produces_recommendations(self, tmp_path):
        """Overlay analysis should produce optimized_path_recommendations.json."""
        from sweep.runner import SweepRunner, SweepConfig
        # Run both mode
        config = SweepConfig(
            model_path="mock", output_dir=str(tmp_path),
            probe_names=["math"], max_layers=4, max_block_size=2,
            mode="both",
        )
        runner = SweepRunner(config, adapter_class=MockAdapter)
        runner.run()

        from analysis.heatmap import generate_overlay_analysis
        dup_path = str(tmp_path / "sweep_results_duplicate.json")
        skip_path = str(tmp_path / "sweep_results_skip.json")
        analysis_dir = str(tmp_path / "analysis")
        generate_overlay_analysis(dup_path, skip_path, analysis_dir)

        recs_path = tmp_path / "analysis" / "optimized_path_recommendations.json"
        assert recs_path.exists()
        with open(recs_path) as f:
            recs = json.load(f)
        assert "math" in recs
        assert "counts" in recs["math"]
        assert "skip_regions" in recs["math"]
        assert "double_regions" in recs["math"]
        assert "optimized_path_hint" in recs["math"]
