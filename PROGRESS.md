# Progress Log

[00:00] DONE: Read all existing code files and CLAUDE.md
[00:01] DONE: Created directory structure per CLAUDE.md spec
[00:01] DONE: Moved runner.py → sweep/runner.py
[00:01] DONE: Moved exllama_adapter.py → sweep/exllama_adapter.py
[00:01] DONE: Moved probe.py → probes/spatial/probe.py
[00:01] DONE: Moved run_sweep.py → scripts/run_sweep.py
[00:02] DONE: Created probes/registry.py (BaseProbe, register_probe, get_probe, list_probes)
[00:02] DONE: Created probes/math/probe.py (COMPLETE — 12 questions with partial credit scoring)
[00:02] DONE: Created probes/code/probe.py (COMPLETE — 6 challenges with unit test scoring)
[00:02] DONE: Created stub probes: eq, factual, language, tool_use, holistic, planning, instruction
[00:02] DONE: Created analysis/heatmap.py (heatmap, skyline, circuit boundary detection)
[00:02] DONE: Created scripts/bootstrap_cloud.sh and scripts/setup_repo.sh
[00:02] DONE: Created all __init__.py files for Python imports
[00:02] DONE: Created requirements.txt
[00:03] NEXT: Verify imports resolve correctly, then spawn parallel subagents

## Subagent 1 — Infrastructure
[02:05] BLOCKED: ExLlamaV2 adapter validation — exllamav2 not installed (no GPU setup) — needs pip install exllamav2 + CUDA GPU
[02:05] SKIPPING TO: Mock adapter compatibility and sweep runner fixes
[02:06] DONE: Updated sweep/mock_adapter.py — added model_path parameter for API compatibility with runner
[02:06] DONE: Updated sweep/runner.py — SweepRunner now accepts optional adapter_class parameter (defaults to ExLlamaV2LayerAdapter)
[02:06] DONE: Added timeout_seconds to SweepConfig (default 30s) and threading-based per-config timeout in SweepRunner
[02:07] DONE: Added --mock and --timeout flags to scripts/run_sweep.py
[02:08] DONE: Smoke test passed — ran full sweep with MockAdapter (mock model, math+spatial probes, max-block=3, max-layers=6, 15 configs)
[02:08] DONE: Results saved to results/latest/sweep_results.json (16 entries including baseline)
[02:08] DONE: Heatmap generation verified — heatmap_math.png, heatmap_spatial.png, skyline.png, circuit_boundaries.json all generated
[02:08] DONE: Standalone heatmap generation also works via generate_all_plots()
[02:09] NEXT: All infrastructure tasks complete. Ready for real model testing when exllamav2 is available.

## Subagent 2 — Probes
[02:10] DONE: Upgraded sweep/mock_adapter.py — 3 modes (random/perfect/terrible), smart prompt detection for all probes
[02:12] DONE: Implemented EQ probe (12 scenarios: guilt, anger, surprise, relief, jealousy, pride, betrayal, gratitude, shame, anxiety, contentment, frustration)
[02:12] DONE: Implemented Factual probe (12 obscure facts with number/word exact match + off-by-one scoring)
[02:12] DONE: Implemented Language probe (16 sentences, 8 grammatical + 8 ungrammatical, subtle violations)
[02:12] DONE: Implemented Tool Use probe (12 scenarios with abstract tool names, exact match scoring)
[02:12] DONE: Implemented Holistic/Analogy probe (12 non-obvious analogies with accept lists)
[02:13] DONE: Implemented Planning probe (10 step-ordering scenarios, pairwise scoring)
[02:13] DONE: Implemented Instruction Following probe (10 multi-constraint scenarios, programmatic checkers)
[02:14] DONE: Redesigned Spatial/Battleship probe — generated boards (seed=42), fleet placement, mid-game simulation, probability density oracle scoring
[02:15] DONE: Created probes/test_probes.py — 31 tests (perfect/terrible mode for all 10 probes + scoring function unit tests + board determinism)
[02:15] DONE: All 31 tests passing
[02:15] DONE: Updated scripts/run_sweep.py — added "planning" and "instruction" to ALL_PROBES list
[02:15] NEXT: All probe tasks complete. No NotImplementedError stubs remaining.

## Merge and Final Validation
[02:16] DONE: Merged all subagent work (infrastructure + probes) into main branch
[02:16] DONE: Applied mid-session guidance — added 3 five-step deep dependency chain planning scenarios
[02:16] DONE: Applied mid-session guidance — added 3 conflicting-constraint instruction scenarios (lowercase, no-spaces, embedded numbers)
[02:16] DONE: Fixed planning scoring to reject natural-language prose containing step letters
[02:16] DONE: Updated mock adapter for new planning and instruction scenarios
[02:17] DONE: All 31 tests passing after refinements
[02:17] DONE: Full mock sweep with all 10 probes completed (26 configs, 8 layers)
[02:17] DONE: All 10 heatmaps + skyline + circuit_boundaries.json generated
[02:17] DONE: MORNING_SUMMARY.md written with status, blockers, next commands, and decisions needed
[02:18] DONE: Committed all probe and infrastructure work.

## Pruning Sweep Mode
[02:20] DONE: Added build_skip_path() to sweep/runner.py — skip config (i,j) removes layers i..j-1
[02:21] DONE: Added SweepConfig.mode field ("duplicate", "skip", "both") and ConfigResult.mode field
[02:22] DONE: Refactored run() into run() + _run_single_sweep() to support both modes
[02:22] DONE: Added _save_results() for mode=both to write separate result files
[02:23] DONE: Added --mode flag to scripts/run_sweep.py (choices: duplicate, skip, both)
[02:24] DONE: Added classify_region() and generate_overlay_analysis() to analysis/heatmap.py
[02:24] DONE: Overlay produces: overlay scatter plots, quadrant map heatmaps, optimized_path_recommendations.json
[02:25] DONE: Added build_optimized_path() to runner.py — combines skip and duplicate regions into single path
[02:26] DONE: Added 13 new tests: TestSweepRunner (6), TestBuildOptimizedPath (5), TestOverlayAnalysis (2)
[02:27] DONE: All 44 tests passing
[02:28] DONE: Smoke test mode=both completed — overlay_math.png, quadrant_map_math.png, optimized_path_recommendations.json generated
[02:28] DONE: Updated CLAUDE.md with pruning sweep documentation (modes, quadrant table, optimized path builder)
[02:29] DONE: Committed pruning sweep work.

## Pre-Production Fixes
[02:30] DONE: BLOCKER 1 — Rewrote ExLlamaV2 adapter KV cache: prefill/decode separation, position-based cache tracking
[02:31] DONE: BLOCKER 2 — Fixed num_layers detection: dynamic ExLlamaV2DecoderLayer isinstance check instead of modules_dict
[02:31] DONE: BLOCKER 3 — Replaced monkey-patched generator with manual autoregressive decode loop (proper prefill/decode)
[02:32] DONE: BLOCKER 4 — Replaced 6 trivial code challenges with 7 harder ones (flatten, LCS, balanced brackets, merge intervals, spiral, RPN eval, permutations)
[02:33] DONE: BLOCKER 5 — Rewrote spatial oracle to use ONLY visible board state, no ground truth ship positions
[02:34] DONE: HIGH — Added baseline variance estimation (3 repeats, mean+std per probe)
[02:34] DONE: HIGH — Added checkpoint resume (--resume flag, skip completed configs)
[02:35] DONE: MEDIUM — Math extraction now uses last number in response, not first
[02:35] DONE: MEDIUM — Atomic checkpointing (write .tmp then rename)
[02:35] DONE: MEDIUM — Replaced factual probe questions with genuinely obscure facts
[02:36] DONE: Updated mock adapter for all changed probes
[02:36] DONE: All 44 tests passing
[02:37] DONE: FIXES_SUMMARY.md written with detailed descriptions of each fix
[02:37] NEXT: Commit pre-production fixes.
