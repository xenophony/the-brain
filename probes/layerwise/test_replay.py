"""
Tests for the harvest-replay probe system.

Tests:
1. Harvest helpers: extract_thinking, extract_answer, check_correct
2. ReplayLayerwiseProbe with mock harvest data
3. ReplayLayerwiseProbe with MockAdapter end-to-end
4. harvest_responses.py with --mock (integration)
5. run_replay.py with --mock (integration)
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sweep.mock_adapter import MockAdapter


# ------------------------------------------------------------------ #
#  Harvest helper tests                                               #
# ------------------------------------------------------------------ #

class TestExtractThinking:

    def test_with_think_tags(self):
        from scripts.harvest_responses import extract_thinking
        thinking, answer = extract_thinking(
            "<think>Glass is fragile</think>\nyes")
        assert thinking == "Glass is fragile"
        assert answer == "yes"

    def test_without_think_tags(self):
        from scripts.harvest_responses import extract_thinking
        thinking, answer = extract_thinking("yes")
        assert thinking == ""
        assert answer == "yes"

    def test_multiline_thinking(self):
        from scripts.harvest_responses import extract_thinking
        response = "<think>Line 1\nLine 2\nLine 3</think>\nno"
        thinking, answer = extract_thinking(response)
        assert "Line 1" in thinking
        assert "Line 3" in thinking
        assert answer == "no"

    def test_empty_response(self):
        from scripts.harvest_responses import extract_thinking
        thinking, answer = extract_thinking("")
        assert thinking == ""
        assert answer == ""


class TestExtractAnswer:

    def test_simple_answer(self):
        from scripts.harvest_responses import extract_answer
        assert extract_answer("yes") == "yes"

    def test_with_thinking(self):
        from scripts.harvest_responses import extract_answer
        assert extract_answer("<think>reasoning</think>\nyes") == "yes"

    def test_case_insensitive(self):
        from scripts.harvest_responses import extract_answer
        assert extract_answer("Yes") == "yes"

    def test_strip_punctuation(self):
        from scripts.harvest_responses import extract_answer
        assert extract_answer("Yes.") == "yes"

    def test_multi_word_takes_first(self):
        from scripts.harvest_responses import extract_answer
        assert extract_answer("yes it will") == "yes"

    def test_empty(self):
        from scripts.harvest_responses import extract_answer
        assert extract_answer("") == ""


class TestCheckCorrect:

    def test_exact_match(self):
        from scripts.harvest_responses import check_correct
        assert check_correct("yes", "yes")
        assert check_correct("no", "no")

    def test_case_insensitive(self):
        from scripts.harvest_responses import check_correct
        assert check_correct("Yes", "yes")
        assert check_correct("NO", "no")

    def test_yes_true_equivalence(self):
        from scripts.harvest_responses import check_correct
        assert check_correct("true", "yes")
        assert check_correct("yes", "true")

    def test_no_false_equivalence(self):
        from scripts.harvest_responses import check_correct
        assert check_correct("false", "no")
        assert check_correct("no", "false")

    def test_mismatch(self):
        from scripts.harvest_responses import check_correct
        assert not check_correct("yes", "no")
        assert not check_correct("true", "false")

    def test_arbitrary_strings(self):
        from scripts.harvest_responses import check_correct
        assert check_correct("valid", "valid")
        assert not check_correct("valid", "invalid")


# ------------------------------------------------------------------ #
#  ReplayLayerwiseProbe tests                                         #
# ------------------------------------------------------------------ #

def _make_mock_harvest(probe_name="causal_logprob", n_items=4):
    """Create mock harvest data for testing."""
    items = []
    for i in range(n_items):
        expected = "yes" if i % 2 == 0 else "no"
        thinking = f"Let me think about item {i}. The answer is {expected}."
        items.append({
            "prompt": f"Test question {i}?",
            "expected": expected,
            "choices": ["yes", "no"],
            "full_response": f"<think>{thinking}</think>\n{expected}",
            "extracted_answer": expected,
            "correct": True,
            "thinking": thinking,
            "difficulty": "easy" if i < 2 else "hard",
        })
    return {
        "probe_name": probe_name,
        "model": "mock",
        "n_items": n_items,
        "n_correct": n_items,
        "accuracy": 1.0,
        "items": items,
    }


class TestReplayLayerwiseProbe:

    def test_init(self):
        from probes.layerwise.replay import ReplayLayerwiseProbe
        probe = ReplayLayerwiseProbe(
            probe_name="causal_logprob",
            harvest_data=_make_mock_harvest(),
        )
        assert probe.name == "causal_logprob_replay"

    def test_run_with_mock_adapter(self):
        from probes.layerwise.replay import ReplayLayerwiseProbe
        model = MockAdapter(mode="random", seed=42, num_layers=16)
        probe = ReplayLayerwiseProbe(
            probe_name="causal_logprob",
            harvest_data=_make_mock_harvest(),
        )
        result = probe.run(model)

        assert "score" in result
        assert "p_correct" in result
        assert "raw_convergence_layer" in result
        assert "replay_convergence_layer" in result
        assert "mean_convergence_delta" in result
        assert "comparison_items" in result
        assert result["n_correct_items"] == 4

    def test_raw_and_replay_both_populated(self):
        from probes.layerwise.replay import ReplayLayerwiseProbe
        model = MockAdapter(mode="random", seed=42, num_layers=16)
        probe = ReplayLayerwiseProbe(
            probe_name="causal_logprob",
            harvest_data=_make_mock_harvest(),
        )
        result = probe.run(model)

        raw_p = result.get("raw_mean_p_correct_by_layer", [])
        replay_p = result.get("replay_mean_p_correct_by_layer", [])
        assert len(raw_p) == 16
        assert len(replay_p) == 16

    def test_comparison_items_structure(self):
        from probes.layerwise.replay import ReplayLayerwiseProbe
        model = MockAdapter(mode="random", seed=42, num_layers=16)
        probe = ReplayLayerwiseProbe(
            probe_name="causal_logprob",
            harvest_data=_make_mock_harvest(n_items=3),
        )
        result = probe.run(model)

        items = result["comparison_items"]
        assert len(items) == 3
        for item in items:
            assert "prompt" in item
            assert "answer" in item
            assert "raw_convergence_layer" in item
            assert "replay_convergence_layer" in item
            assert "raw_final_p" in item
            assert "replay_final_p" in item

    def test_only_correct_items_used(self):
        from probes.layerwise.replay import ReplayLayerwiseProbe
        harvest = _make_mock_harvest(n_items=4)
        # Mark some as incorrect
        harvest["items"][1]["correct"] = False
        harvest["items"][3]["correct"] = False

        model = MockAdapter(mode="random", seed=42)
        probe = ReplayLayerwiseProbe(
            probe_name="causal_logprob",
            harvest_data=harvest,
        )
        result = probe.run(model)
        assert result["n_correct_items"] == 2

    def test_items_without_thinking_skipped(self):
        from probes.layerwise.replay import ReplayLayerwiseProbe
        harvest = _make_mock_harvest(n_items=4)
        # Remove thinking from some items
        harvest["items"][0]["thinking"] = ""
        harvest["items"][2]["thinking"] = "   "

        model = MockAdapter(mode="random", seed=42)
        probe = ReplayLayerwiseProbe(
            probe_name="causal_logprob",
            harvest_data=harvest,
        )
        result = probe.run(model)
        assert result["n_correct_items"] == 2

    def test_no_correct_items_returns_empty(self):
        from probes.layerwise.replay import ReplayLayerwiseProbe
        harvest = _make_mock_harvest(n_items=2)
        harvest["items"][0]["correct"] = False
        harvest["items"][1]["correct"] = False

        model = MockAdapter(mode="random", seed=42)
        probe = ReplayLayerwiseProbe(
            probe_name="causal_logprob",
            harvest_data=harvest,
        )
        result = probe.run(model)
        assert result["n_correct_items"] == 0
        assert result["score"] == 0.0

    def test_load_from_file(self):
        from probes.layerwise.replay import ReplayLayerwiseProbe
        harvest = _make_mock_harvest()

        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False) as f:
            json.dump(harvest, f)
            tmp_path = f.name

        try:
            model = MockAdapter(mode="random", seed=42, num_layers=16)
            probe = ReplayLayerwiseProbe(
                harvest_file=tmp_path,
                probe_name="causal_logprob",
            )
            result = probe.run(model)
            assert result["n_correct_items"] == 4
        finally:
            Path(tmp_path).unlink()

    def test_max_items_respected(self):
        from probes.layerwise.replay import ReplayLayerwiseProbe
        model = MockAdapter(mode="random", seed=42, num_layers=16)
        probe = ReplayLayerwiseProbe(
            probe_name="causal_logprob",
            harvest_data=_make_mock_harvest(n_items=10),
        )
        probe.max_items = 3
        result = probe.run(model)
        assert result["n_correct_items"] == 3

    def test_perfect_mode_convergence(self):
        from probes.layerwise.replay import ReplayLayerwiseProbe
        model = MockAdapter(mode="perfect", seed=42, num_layers=32)
        probe = ReplayLayerwiseProbe(
            probe_name="causal_logprob",
            harvest_data=_make_mock_harvest(n_items=4),
        )
        result = probe.run(model)
        # In perfect mode, both should converge somewhere
        raw_conv = result.get("raw_convergence_layer")
        replay_conv = result.get("replay_convergence_layer")
        assert raw_conv is not None or replay_conv is not None

    def test_choices_from_harvest_data(self):
        from probes.layerwise.replay import ReplayLayerwiseProbe
        harvest = _make_mock_harvest()
        probe = ReplayLayerwiseProbe(
            probe_name="nonexistent_probe",
            harvest_data=harvest,
        )
        choices = probe._get_choices(harvest)
        assert "yes" in choices
        assert "no" in choices


# ------------------------------------------------------------------ #
#  Integration tests: CLI scripts                                     #
# ------------------------------------------------------------------ #

class TestHarvestScript:

    def test_harvest_mock_runs(self):
        """harvest_responses.py --mock should run without errors."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "harvest_responses.py"),
             "--mock", "--probes", "causal_logprob", "--max-items", "3",
             "--output-dir", str(Path(tempfile.gettempdir()) / "test_harvest")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Harvesting causal_logprob" in result.stdout

    def test_harvest_creates_output_file(self):
        """harvest_responses.py should create the expected JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import subprocess
            result = subprocess.run(
                [sys.executable,
                 str(project_root / "scripts" / "harvest_responses.py"),
                 "--mock", "--probes", "causal_logprob", "--max-items", "2",
                 "--output-dir", tmpdir],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"

            output_file = Path(tmpdir) / "causal_logprob_responses.json"
            assert output_file.exists()

            with open(output_file) as f:
                data = json.load(f)
            assert data["probe_name"] == "causal_logprob"
            assert len(data["items"]) == 2
            assert "thinking" in data["items"][0]
            assert "correct" in data["items"][0]


class TestRunReplayScript:

    def test_replay_mock_runs(self):
        """run_replay.py --mock should run without errors."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "run_replay.py"),
             "--mock", "--probes", "causal_logprob", "--max-items", "2",
             "--output-dir", str(Path(tempfile.gettempdir()) / "test_replay")],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Raw convergence layer" in result.stdout
        assert "Replay convergence layer" in result.stdout

    def test_replay_creates_output_files(self):
        """run_replay.py should create replay JSON outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import subprocess
            result = subprocess.run(
                [sys.executable,
                 str(project_root / "scripts" / "run_replay.py"),
                 "--mock", "--probes", "causal_logprob", "--max-items", "2",
                 "--output-dir", tmpdir],
                capture_output=True, text=True, timeout=60,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"

            replay_file = Path(tmpdir) / "causal_logprob_replay.json"
            assert replay_file.exists()

            with open(replay_file) as f:
                data = json.load(f)
            assert "raw_convergence_layer" in data
            assert "replay_convergence_layer" in data
            assert "comparison_items" in data

            summary_file = Path(tmpdir) / "replay_summary.json"
            assert summary_file.exists()


# ------------------------------------------------------------------ #
#  End-to-end: harvest then replay                                    #
# ------------------------------------------------------------------ #

class TestEndToEnd:

    def test_harvest_then_replay(self):
        """Full pipeline: harvest with mock, then replay with mock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harvest_dir = Path(tmpdir) / "harvested"
            replay_dir = Path(tmpdir) / "replay"

            import subprocess

            # Step 1: Harvest
            r1 = subprocess.run(
                [sys.executable,
                 str(project_root / "scripts" / "harvest_responses.py"),
                 "--mock", "--probes", "causal_logprob", "--max-items", "3",
                 "--output-dir", str(harvest_dir)],
                capture_output=True, text=True, timeout=30,
            )
            assert r1.returncode == 0, f"Harvest failed: {r1.stderr}"

            # Step 2: Replay using harvested data
            r2 = subprocess.run(
                [sys.executable,
                 str(project_root / "scripts" / "run_replay.py"),
                 "--mock", "--probes", "causal_logprob", "--max-items", "3",
                 "--harvest-dir", str(harvest_dir),
                 "--output-dir", str(replay_dir)],
                capture_output=True, text=True, timeout=60,
            )
            # Note: with --mock, run_replay generates its own mock data,
            # so harvest_dir doesn't need to exist. This tests the mock path.
            assert r2.returncode == 0, f"Replay failed: {r2.stderr}"
