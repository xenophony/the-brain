"""
Tests for all probes using MockAdapter.

Run: python -m pytest probes/test_probes.py -v
"""

import sys
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
        assert score_math("392", 391) == 0.9  # <1% error
        assert score_math("banana", 391) == 0.0

    def test_factual_scoring(self):
        from probes.factual.probe import score_factual
        q_num = {"answer": "206", "type": "number"}
        assert score_factual("206", q_num) == 1.0
        assert score_factual("207", q_num) == 0.5  # off by one
        assert score_factual("banana", q_num) == 0.0

        q_word = {"answer": "olympus", "type": "word", "alternates": ["olympus mons"]}
        assert score_factual("Olympus", q_word) == 1.0
        assert score_factual("Olympus Mons", q_word) == 1.0
        assert score_factual("mars", q_word) == 0.0

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
        board = _make_empty_board()
        board[5][5] = 'H'
        ship_cells = {(5, 5), (5, 6)}
        placements = [({(5, 5), (5, 6)}, 2)]
        density = compute_probability_density(board, ship_cells, placements)
        assert len(density) > 0
        # The cell adjacent to the hit should have density > 0
        assert density.get((5, 6), 0) > 0 or density.get((5, 4), 0) > 0
