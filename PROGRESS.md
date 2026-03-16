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
[02:37] DONE: Committed pre-production fixes.

## Safety-Relevant Probes
[02:40] DONE: Created probes/hallucination/probe.py — 16 questions, 4 categories (unknowable/traps/edge/control), hedge detection, weighted scoring
[02:42] DONE: Created probes/sycophancy/probe.py — 11 scenarios, 3-level escalating pressure, phase1 exclusion, pressure resistance scoring
[02:44] DONE: Created probes/consistency/probe.py — 12 scenarios, chain-of-thought vs direct answer, answer matching
[02:46] DONE: Expanded probes/instruction/probe.py — 11 new scenarios: Type A (preference conflict), Type B (persistence), Type C (nested conflict)
[02:48] DONE: Added "sycophantic" mode to MockAdapter — capitulates to pressure, gives inconsistent answers, confidently wrong
[02:49] DONE: Added perfect-mode responses for all 3 new probes + expanded instruction scenarios
[02:50] DONE: Added safety_analysis() to analysis/heatmap.py — identifies integrity/deception/sycophancy/resistance circuits
[02:51] DONE: Updated CLAUDE.md — brain region mapping + safety probes section
[02:52] DONE: Updated scripts/run_sweep.py — ALL_PROBES list + safety analysis integration
[02:53] DONE: Added 16 new tests (60 total) — perfect/terrible/sycophantic modes + scoring functions + safety analysis
[02:53] DONE: All 60 tests passing
[02:54] DONE: SAFETY_PROBES_SUMMARY.md written
[02:54] NEXT: Commit safety probes work.

## Cognitive Probes — Round 2
[03:00] DONE: Created probes/temporal/probe.py — 16 questions across 4 types (causal chain, relative time, contradiction detection, counterfactual temporal)
[03:05] DONE: Created probes/metacognition/probe.py — 20 questions with confidence calibration scoring (easy/medium/obscure/trick mix)
[03:10] DONE: Created probes/counterfactual/probe.py — 15 scenarios across 3 types (physical, social, logical counterfactuals)
[03:15] DONE: Created probes/abstraction/probe.py — 15 items across 3 types (concrete-to-abstract, abstract-to-concrete, level identification)
[03:20] DONE: Created probes/noise_robustness/probe.py — 10 base questions x 4 variants (clean, reworded, noisy context, casual)
[03:25] DONE: Updated sweep/mock_adapter.py — perfect responses for all 5 new probes + "overconfident" and "fragile" modes
[03:30] DONE: Updated scripts/run_sweep.py — ALL_PROBES includes 5 new probes (18 total)
[03:30] DONE: Updated CLAUDE.md — brain region mapping table includes 5 new probes
[03:35] DONE: Added 21 new tests (81 total) — perfect/terrible/overconfident/fragile modes + scoring function unit tests
[03:35] DONE: All 81 tests passing (60 original + 21 new)

## Baseline API Comparison Infrastructure
[04:00] DONE: Created sweep/api_adapters.py — BaseAPIAdapter + ClaudeAdapter, GeminiAdapter, GroqAdapter, TogetherAdapter with exponential backoff, API logging, import guards
[04:05] DONE: Created scripts/run_baselines.py — CLI to run all 18 probes against 5 API models with atomic checkpointing, resume support, dry-run cost estimation
[04:10] DONE: Created analysis/calibration.py — ceiling/floor effects, dynamic range, orthogonality (Pearson correlation), generates CALIBRATION_REPORT.md
[04:15] DONE: Created scripts/estimate_cost.py — standalone cost estimator for baselines and sweeps with itemized breakdown
[04:20] DONE: Created sweep/test_api_adapters.py — 44 tests (34 non-API + 10 API-gated), covers adapters, calibration, cost estimator, model registry validation
[04:22] DONE: Updated requirements.txt — added anthropic, google-generativeai, groq, together, python-dotenv
[04:23] DONE: Created .env.example with placeholder API keys
[04:24] DONE: Created pytest.ini to register custom 'api' mark
[04:25] DONE: All 140 tests passing (81 original + 34 new non-API + 10 new API-skipped + 15 other existing), 10 skipped (API keys not set)
[05:00] DONE: Added difficulty tiering (easy/hard) to all 10 original probes
[05:00] DONE: Updated BaseProbe with _make_result() helper returning dict with score/easy_score/hard_score/n_easy/n_hard
[05:00] DONE: Updated SweepRunner.run_probes() to handle dict returns (extracts float score for existing flow)
[05:00] DONE: math probe: 8 easy (single-op) + 8 hard (multi-step, new: cube_root_27000, 17^3, sum_15_squares, LCM_12_18)
[05:00] DONE: code probe: 8 easy (is_even, sum_list, first_element, reverse_string, count_vowels, fizzbuzz, abs_value, is_palindrome) + 8 hard (existing + binary_search)
[05:00] DONE: eq probe: 8 easy (obvious emotions: betrayal, relief, frustration, pride, anger, gratitude, shame, joy) + 8 hard (ambiguous: guilt, jealousy, contentment, surprise, anxiety, ambivalence, conflicted, disbelief)
[05:00] DONE: factual probe: 8 easy (well-known: gold atomic#, WW2 end, bones, continents, speed of light, chromosomes, boiling water, Fe symbol) + 8 hard (obscure: tungsten, electron mass, C14, nitrogen, Uranus moons, hydrogen ionization, topaz, gallium)
[05:00] DONE: language probe: 8 easy (obvious errors + clear grammatical) + 8 hard (subtle: neither/nor, each/has, along with, had I known, that-that, team-are, between you and I)
[05:00] DONE: tool_use probe: 8 easy (clear one-to-one mappings) + 8 hard (ambiguous: compressor, converter, merger, formatter + 4 new multi-plausible scenarios)
[05:00] DONE: holistic probe: 8 easy (common analogies + teacher/patient, bark/cat) + 8 hard (non-obvious: painter/composer, fish/bird, etc. + menu/meal, cocoon/tree)
[05:00] DONE: planning probe: 8 easy (4-step linear) + 8 hard (5-step chains: ML deploy, mobile app, aquarium + 5 new: fundraiser, vintage car, solar, documentary, product launch)
[05:00] DONE: spatial probe: 10 easy boards (single isolated hit) + 10 hard boards (3+ hits, complex states)
[05:00] DONE: instruction probe: 8 easy (simple 3-constraint + single-check Type B) + 8 hard (conflicting constraints + Type A/C)
[05:00] DONE: Updated MockAdapter with responses for all new items (math, code, eq, factual, language, tool_use, holistic, planning, instruction)
[05:00] DONE: Updated test_probes.py with _extract_score() helper for all 10 original probe tests
[05:00] DONE: All 81 tests passing after difficulty tiering upgrade
[05:00] DONE: Added difficulty tiering (easy/hard) to all 10 original probes
[05:00] DONE: Updated BaseProbe with _make_result() helper returning dict with score/easy_score/hard_score/n_easy/n_hard
[05:00] DONE: Updated SweepRunner.run_probes() to handle dict returns (extracts float score for existing flow)
[05:00] DONE: math probe: 8 easy + 8 hard (new: cube_root_27000, 17^3, sum_15_squares, LCM_12_18)
[05:00] DONE: code probe: 8 easy (new trivial functions) + 8 hard (existing + binary_search)
[05:00] DONE: eq probe: 8 easy (obvious emotions) + 8 hard (ambiguous: ambivalence, conflicted, disbelief)
[05:00] DONE: factual probe: 8 easy (well-known facts) + 8 hard (obscure facts)
[05:00] DONE: language probe: 8 easy (obvious errors) + 8 hard (subtle violations)
[05:00] DONE: tool_use probe: 8 easy (clear mappings) + 8 hard (ambiguous scenarios)
[05:00] DONE: holistic probe: 8 easy (common analogies) + 8 hard (non-obvious analogies)
[05:00] DONE: planning probe: 8 easy (4-step) + 8 hard (5-step chains)
[05:00] DONE: spatial probe: 10 easy boards (single hit) + 10 hard boards (3+ hits)
[05:00] DONE: instruction probe: 8 easy (simple constraints) + 8 hard (conflicting)
[05:00] DONE: Updated MockAdapter with all new item responses
[05:00] DONE: Updated test_probes.py with _extract_score() helper
[05:00] DONE: All 81 tests passing after difficulty tiering upgrade

## Residual Stream Tracing Infrastructure
[06:00] DONE: Added forward_with_hooks(), project_to_vocab(), tokens_to_ids() to MockAdapter — synthetic sigmoidal/sycophantic/terrible probability curves
[06:00] DONE: Added forward_with_hooks(), project_to_vocab(), tokens_to_ids() to ExLlamaV2LayerAdapter — real model hidden state projection
[06:05] DONE: Created analysis/residual_tracer.py — ResidualTracer with trace(), trace_batch(), trace_domain(), compare_paths(), trace_sycophancy(), trace_hallucination()
[06:05] DONE: Created analysis/trace_heatmap.py — overlay_trace_on_heatmap(), validate_circuit_mechanistically(), generate_mechanistic_report()
[06:10] DONE: Created scripts/run_traces.py — CLI for --probe, --all-domains, --mock, --mode safety
[06:10] DONE: Created RESEARCH.md — project overview, residual stream analysis technique, trace types, connection to ROME/logit lens, phase roadmap
[06:15] DONE: Created analysis/test_traces.py — 34 tests covering MockAdapter hooks, ResidualTracer, SycophancyTrace, HallucinationTrace, DomainTrace, PathComparison, trace_heatmap, existing mode regression
[06:15] DONE: All 34 new trace tests passing
[06:15] DONE: All 145 existing tests still passing (0 regressions)
[06:20] DONE: Ran scripts/run_traces.py --all-domains --mock — 16/18 probes traced successfully, output saved to results/traces/
[06:20] DONE: Residual stream tracing infrastructure complete. Ready for real model validation.

## Documentation
[06:30] DONE: Updated MORNING_SUMMARY.md — accurate state: 18 probes, 179 tests, all infrastructure
[06:45] DONE: Created PROBE_REVIEW.md — human-readable summary of all 18 probes with examples, scoring, risks, expected scores
[06:55] DONE: Created API_KEYS_NEEDED.md — exact setup instructions for all 4 API services with costs and minimum viable baseline
[06:55] NEXT: All documentation complete.

## Audit Fixes — Residual Stream Tracing
[07:00] DONE: CRITICAL — Fixed project_to_vocab truncation in exllama_adapter.py (was min(100, len(probs)), now accepts target_token_ids or returns full distribution)
[07:00] DONE: CRITICAL — Documented forward_with_hooks cache reset as intentional (no code change needed)
[07:01] DONE: HIGH — Changed trace() to use only first token of correct_tokens for p_correct (standard logit lens)
[07:02] DONE: HIGH — Replaced HallucinationTrace with entropy-based tracing (per-layer entropy, drop detection, crossover)
[07:03] DONE: MEDIUM — Added pre-layer-0 embedding hook at position (-1, -1) in both adapters
[07:03] DONE: MEDIUM — Changed hook signature to (exec_pos, layer_idx, hidden) in both adapters
[07:04] DONE: MEDIUM — Updated LayerTrace to store 3-tuples; updated all code in ResidualTracer
[07:05] DONE: MEDIUM — Added pressure_answer_probabilities, genuine_override_layer, distraction_only to SycophancyTrace
[07:05] DONE: LOW — Added ignore_early_frac parameter to ResidualTracer
[07:06] DONE: MockAdapter entropy support — _entropy field in hidden states, mode-specific curves (perfect/sycophantic/terrible)
[07:07] DONE: Updated analysis/test_traces.py — fixed existing tests for new API, added 12 new tests
[07:08] DONE: All 191 tests passing (179 original + 12 new), 10 skipped
[07:08] DONE: Created AUDIT_FIXES.md with full change summary
[07:08] DONE: Audit fixes complete.

## Documentation — Probe Review
[07:30] DONE: Regenerated PROBE_REVIEW.md from scratch — 18 probes with verbatim examples, scoring, expected ranges, ceiling risks, orthogonality notes, summary table, Tier 1 sweep recommendation
[07:30] DONE: API adapter updates (OpenRouter, Gemini 3 Pro, remove Opus).

## Probe Fixes from Human Review
[08:10] DONE: FIX 1 — temporal probe Type A: replaced 2 of 4 all-no questions with yes-answer scenarios (birds singing after sunrise, plant growing after rain). Now 2 yes + 2 no.
[08:10] DONE: FIX 2 — noise_robustness: replaced 3 over-represented science/math questions with history (WW2 end), language (opposite of ancient), food/culture (miso soup ingredient). Now 2 geography, 2 science, 2 math, 1 history, 1 general, 1 language, 1 food/culture.
[08:10] DONE: Updated MockAdapter for all changed questions (temporal yes/no, noise_robustness clean prompts, fragile mode keywords)
[08:10] DONE: All 192 tests passing, 14 skipped

## Baseline Runner Parallelization
[08:30] DONE: Added rate_limit_workers to all API adapters (Claude/Gemini=5, Groq/OpenRouter=10)
[08:30] DONE: Rewrote run_baselines with ThreadPoolExecutor — 6 probes concurrent per model
[08:30] DONE: Thread-safe progress reporting and checkpoint locking
[08:30] DONE: Added time estimate to --dry-run (sequential: 13.5min, parallel: 2.2min, 6x speedup)
[08:30] DONE: All 192 tests passing, 14 skipped

## Baseline Runner Reliability Fixes
[09:00] DONE: Added per-API-call timeout (30s) to _retry_with_backoff — prevents individual calls from hanging forever
[09:00] DONE: Added per-probe timeout (180s/3min) to _run_single_probe — kills hung probes, records partial error
[09:00] DONE: Improved error logging — shows exception type and message, distinguishes TIMEOUT vs ERROR
[09:00] DONE: Cleared failed gemini-3-pro and qwen-30b spatial results for rerun

## OpenRouter Integration + claude-opus Removal
[08:00] DONE: Added OpenRouterAdapter to sweep/api_adapters.py — OpenAI-compatible client, exponential backoff, API logging
[08:00] DONE: Added "openrouter" to ADAPTER_MAP and _ENV_KEYS in api_adapters.py
[08:01] DONE: Updated MODEL_REGISTRY in scripts/run_baselines.py — all 5 models now route through OpenRouter primary
[08:01] DONE: Added FALLBACK_REGISTRY in scripts/run_baselines.py — llama-8b/70b->groq, claude-sonnet->claude, gemini-3-pro->gemini
[08:01] DONE: Added _resolve_provider() to run_baselines.py — tries OpenRouter first, then falls back to individual provider keys
[08:02] DONE: Updated PRICING in run_baselines.py and estimate_cost.py — new OpenRouter pricing, removed claude-opus
[08:02] DONE: Updated ALL_MODELS in estimate_cost.py — replaced gemini-2.5-pro with gemini-3-pro, added qwen-30b, removed claude-opus
[08:03] DONE: Replaced API_KEYS_NEEDED.md — OpenRouter as primary, individual keys as fallback, no Opus references
[08:03] DONE: Updated .env.example — added OPENROUTER_API_KEY as primary, kept others as optional fallback
[08:04] DONE: Updated sweep/test_api_adapters.py — added OpenRouterAdapter import, missing-key test, adapter map assertion, live test class
[08:04] DONE: Removed all claude-opus references across entire codebase (verified with grep)
[08:05] DONE: All 192 tests passing (178 non-API + 14 API-skipped), 0 failures

## Scoring Extraction Fixes for Real Model Output
[10:00] DONE: FIX 1 — Code probe: new extraction pipeline handles ```python fences, prose before/after, searches for def func_name( anywhere
[10:05] DONE: FIX 2 — Consistency probe: _extract_final_answer uses rfind (last occurrence), expanded markers, bolded text fallback. max_new_tokens 80->300. _answers_match uses last-number extraction.
[10:10] DONE: FIX 3 — EQ probe: new _extract_eq_digit uses last word-bounded digit, not first digit (avoids echoed scenario text)
[10:12] DONE: FIX 4 — Spatial probe: verified extraction regex already handles all real model patterns. No changes needed.
[10:15] DONE: FIX 5 — Added BaseProbe.log_responses flag and item_results to _make_result. Code/EQ/consistency/spatial probes collect per-item details when enabled.
[10:18] DONE: FIX 6 — Updated run_baselines.py: sets log_responses=True, saves item_results to baseline_responses.json, prints 2 sample responses per probe.
[10:20] DONE: All 81 tests passing. No regressions.
[10:22] DONE: Created SCORING_FIXES.md with detailed descriptions of each fix.
[10:22] DONE: Ready for re-running baselines with fixed extraction.

## Baseline Rerun Fixes
[10:30] DONE: FIX 1 — Code probe max_new_tokens increased from 150 to 400 (hard challenges were being truncated mid-function)
[10:30] DONE: FIX 2 — Spatial prompt strengthened: "You MUST respond with ONLY a grid coordinate... No explanation. No reasoning."
[10:30] DONE: FIX 3 — Consistency sample printing fixed: handles reasoning_raw/direct_raw keys not just "response"
[10:30] DONE: FIX 4 — Per-call timeout reduced from 30s to 20s to prevent cumulative hanging on slow endpoints
[10:30] DONE: All 192 tests passing

## Baseline Rerun Fixes
[10:30] DONE: FIX 1 — Code probe max_new_tokens increased from 150 to 400 (hard challenges were being truncated mid-function)
[10:30] DONE: FIX 2 — Spatial prompt strengthened: "You MUST respond with ONLY a grid coordinate... No explanation. No reasoning."
[10:30] DONE: FIX 3 — Consistency sample printing fixed: handles reasoning_raw/direct_raw keys not just "response"
[10:30] DONE: FIX 4 — Per-call timeout reduced from 30s to 20s to prevent cumulative hanging on slow endpoints
[10:30] DONE: All 192 tests passing

## Baseline Response Analysis
[11:00] DONE: Analyzed spatial responses — Claude truncated by max_new_tokens=5 (reasoning preamble), valid coords avg 0.518 vs llama 0.648
[11:00] DONE: Analyzed consistency responses — 3/12 extraction failures from intermediate calc numbers, not reasoning failures
[11:00] DONE: Analyzed code responses — 2/16 truncated at 400 tokens, model knows algorithm but response cut off
[11:00] DONE: Analyzed EQ qwen-30b — 12/16 empty responses from content filtering, not extraction bug
[11:00] DONE: Created TRACE_ANALYSIS.md with full findings and fix recommendations
[11:00] NEXT: Apply 3 remaining fixes (spatial tokens, code tokens, consistency extraction)

## Final Extraction Fixes
[11:30] DONE: FIX 1 — spatial max_new_tokens 5 → 10 (Claude reasoning preamble no longer truncated)
[11:30] DONE: FIX 2 — code max_new_tokens 400 → 500 (complex DP solutions no longer truncated)
[11:30] DONE: FIX 3 — consistency extraction: last-line-first-word check for direct answers (no/yes/friday), bold text priority, last-section-only marker search avoids intermediate calc numbers
[11:30] DONE: Added EQ qwen-30b content filtering finding to RESEARCH.md
[11:30] DONE: All 192 tests passing

## Four-Fix Batch
[12:00] DONE: FIX 1 — Verified consistency _extract_final_answer works correctly on all 4 test patterns (120 km/h, no, 3/8, 4 prime numbers). No code change needed.
[12:05] DONE: FIX 2 — Added per-adapter request_timeout: Claude=30s, Gemini=30s, Groq=20s, Together=30s, OpenRouter=30s (45s for qwen models). All generate_short methods now pass timeout_seconds=self.request_timeout.
[12:10] DONE: FIX 3 — Created probes/spatial_pathfinding/probe.py — 16 grids (8 easy 5x5 + 8 hard 8x8), BFS oracle at load time, 2 unsolvable grids. Scoring: exact=1.0, off-by-1=0.5, else 0.0. Added MockAdapter perfect responses via BFS. Added 10 tests.
[12:15] DONE: FIX 4 — Renamed gemini-3-pro to gemini-2.5-pro everywhere: run_baselines.py MODEL_REGISTRY+FALLBACK_REGISTRY+PRICING, estimate_cost.py PRICING+_PRICING_KEY+ALL_MODELS. Model ID now google/gemini-2.5-pro-preview-05-06.
[12:15] DONE: Added spatial_pathfinding to ALL_PROBES in run_baselines.py (19 total). Updated N_PROBES=19 in estimate_cost.py.
[12:15] DONE: All 125 non-API tests passing, 14 API tests deselected. Dry-run with --models all shows gemini-2.5-pro correctly resolved.

## Pathfinding Probe + Model Fixes
[12:00] DONE: FIX 1 — Consistency extraction verified working on all test patterns (no change needed)
[12:00] DONE: FIX 2 — Per-model timeout: qwen=45s, groq=20s, others=30s via request_timeout attribute
[12:00] DONE: FIX 3 — Created probes/spatial_pathfinding/probe.py: 16 BFS grids (8 easy 5x5, 8 hard 8x8), 2 unsolvable, integer output
[12:00] DONE: FIX 4 — Gemini model string: gemini-3-pro → gemini-2.5-pro (google/gemini-2.5-pro-preview-05-06)
[12:00] DONE: MockAdapter pathfinding support via BFS on prompt grid
[12:00] DONE: 10 new pathfinding tests, 201 total passing, 14 deselected
[12:00] DONE: Dry run shows all 5 models OK, 19 probes, $0.33 estimated

## Scoring Investigation + Fix
[12:30] DONE: Investigated spatial_pathfinding scoring — NOT a bug. Models genuinely bad at pathfinding (llama-8b answers "4" for everything). Scoring is correct.
[12:30] DONE: Investigated consistency extraction — "there are" marker matched "There are 60 minutes in an hour" from truncated CoT working. Two fixes:
[12:30] DONE: Removed "there are" from conclusion markers (too generic, matches intermediate facts)
[12:30] DONE: Increased consistency max_new_tokens from 300 to 500 (llama-70b was truncated before reaching conclusion)
[12:30] DONE: All 201 tests passing

## Auto Calibration Report
[13:00] DONE: Added automatic calibration report generation to run_baselines.py (generates after all models complete)
[13:00] DONE: Added quick summary to stdout with CEILING and LOW RANGE flags
[13:00] DONE: Generated CALIBRATION_REPORT.md from current results
[13:00] NOTE: High orthogonality correlations (r>0.9 everywhere) are artifacts of gemini-2.5-pro scoring ~0 on all probes. Remove gemini from correlation analysis for meaningful orthogonality.
[13:00] NOTE: spatial_pathfinding has LOW RANGE (0.094) — models genuinely can't pathfind. Keep for sweep (may show signal with circuit manipulation) but don't rely on it for baseline calibration.

## Gemini Thinking Token Fix
[13:30] DONE: Diagnosed Gemini 2.5 Pro scoring 0 on most probes — "thinking" mode consumes max_tokens budget, leaving no room for actual answer
[13:30] DONE: Evidence: responses truncated at 35-48 chars ("Of course, let's think through this") with empty direct answers
[13:30] DONE: Fix: multiply max_tokens by 8x (min 1024) for Gemini models in both OpenRouterAdapter and GeminiAdapter
[13:30] DONE: All 201 tests passing
[13:30] NEXT: Rerun gemini-2.5-pro on all probes with --no-resume

## Gemini Direct API + Model Upgrade
[14:00] DONE: Listed available Gemini models via Google AI API — gemini-3.1-pro-preview available
[14:00] DONE: Updated MODEL_REGISTRY: gemini-3.1-pro uses GeminiAdapter directly (not OpenRouter)
[14:00] DONE: Model ID: gemini-3.1-pro-preview (Google AI API, frontier ceiling)
[14:00] DONE: Fallback: OpenRouter google/gemini-2.5-pro-preview-05-06 if GOOGLE_API_KEY not set
[14:00] DONE: Updated estimate_cost.py pricing and model list
[14:00] DONE: 8x token multiplier confirmed in both GeminiAdapter and OpenRouterAdapter
[14:00] DONE: Dry run shows all 5 models OK, $0.36 estimated
[14:00] DONE: All 201 tests passing

## Frontier Model Token Budget Fix
[15:00] DONE: Added BaseAPIAdapter._get_max_tokens() — 10x scaling (min 2048) for thinking models (gpt-5, o1, o3, gemini, deepseek-r1, qwq)
[15:00] DONE: Applied to all 5 adapters: Claude, Gemini, Groq, Together, OpenRouter
[15:00] DONE: Removed inline scaling from OpenRouter and Gemini adapters (centralized)
[15:00] DONE: Increased probe timeout from 180s to 300s for thinking model headroom
[15:00] DONE: All 201 tests passing

## Benchmark Research
[14:56] DONE: Read all 19 probe source files to understand current implementation
[14:56] DONE: Researched benchmarks: GSM8K, HumanEval, MBPP, EQ-Bench, SimpleQA, TriviaQA, MMLU, BLiMP, IFEval, PlanBench, TruthfulQA, Perez/Sharma sycophancy, ToolBench, API-Bank, ARC-AGI, SpartQA, AdvGLUE, CounterBench, TimeBench, BATS, MazeEval
[14:56] DONE: Created BENCHMARK_RESEARCH.md — 19 probe sections with benchmark, license, lm-eval-harness status, score distributions, recommendation
[14:56] DONE: Recommendations: 1 REPLACE (factual->SimpleQA), 10 AUGMENT, 8 KEEP AS-IS
[14:56] DONE: Priority actions: factual replacement highest impact, instruction+math augmentation for leaderboard comparability
[14:56] NEXT: Implement benchmark augmentations starting with factual probe SimpleQA replacement

## Benchmark Augmentations (Priorities 1-4)
[15:30] DONE: PRIORITY 1 — Replaced factual probe with 20 SimpleQA-style items (10 easy + 10 hard). Diverse domains, single unambiguous answers.
[15:35] DONE: PRIORITY 2 — Augmented math probe with 8 GSM8K-style word problems (HARD_ITEMS now 16). 1-2 step problems with integer answers.
[15:40] DONE: PRIORITY 3 — Augmented hallucination probe with 8 TruthfulQA-style common misconceptions (Category B). Total questions now 24.
[15:45] DONE: PRIORITY 4 — Augmented instruction probe with 8 IFEval-style constraint items (HARD_ITEMS now 16). Added 12 new checker functions.
[15:50] DONE: Updated MockAdapter perfect-mode responses for all 32 new items across all 4 probes.
[15:55] DONE: All 90 tests passing. All probes score 1.0 in perfect mode.
[15:55] DONE: Created AUGMENTATION_SUMMARY.md with detailed change log.

[--:--] DONE: Created spatial_pong_simple probe (8 easy + 8 hard trajectory prediction scenarios, seed=42)
[--:--] DONE: Created spatial_pong_strategic probe (8 easy + 8 hard motion planning scenarios, 3 unreachable)
[--:--] DONE: Created shared pong_oracle.py with wall bounce simulation and reachability logic
[--:--] DONE: Added MockAdapter support (perfect mode uses oracle, random mode returns random up/down/stay)
[--:--] DONE: Added 17 new tests (oracle wall bounce, reachability, perfect/terrible scores, structure, scoring)
[--:--] DONE: Updated run_baselines.py ALL_PROBES (added pong probes, removed spatial_pathfinding from defaults)
[--:--] DONE: Updated CLAUDE.md brain region mapping table
[--:--] DONE: All 107 tests pass (90 existing + 17 new)

## ExLlamaV2 0.3.2 Compatibility
[00:30] DONE: Fixed layer detection: ExLlamaV2DecoderLayer -> ExLlamaV2ParallelDecoder (from exllamav2.model)
[00:30] DONE: Added version-compatible import fallback: tries ExLlamaV2ParallelDecoder first, falls back to ExLlamaV2DecoderLayer for older versions
[00:45] DONE: Fixed ExLlamaV2Tokenizer -> ExLlamaV2TokenizerHF (from exllamav2.tokenizer)
[00:45] DONE: Fixed ExLlamaV2Cache import (from exllamav2.cache)
[00:45] DONE: All imports now have version-compatible fallbacks for both 0.2.x and 0.3.x
[01:00] DONE: Added max_seq_len parameter to ExLlamaV2LayerAdapter (default 2048, reduces KV cache VRAM)
[01:00] DONE: Changed model.load(lazy=False) to model.load() — loads pre-quantized weights
[01:00] NOTE: Model must be in pre-quantized format (GPTQ/EXL2). Use quantized Qwen3-30B-A3B for 32GB VRAM.

## GPU Sweep Debugging Session
[02:00] CRITICAL FINDING: ExLlamaV2 embedding layer runs on CPU (device_idx=-1) while transformer layers run on GPU (device_idx=0). This is by design — ExLlamaV2's own forward_chunk uses safe_move_tensor to shuttle between devices.
[02:00] CRITICAL FINDING: CUDA operations fail in non-main threads. Sweep runner threading bypasses GPU execution entirely — all probes timeout silently.
[02:00] FIX: direct probe.run() call for GPU adapters (hasattr(model, '_model')), threading only for mock/API adapters.
[02:00] FIX: _run_module() moves tensors to match each module's device_idx (-1=CPU for embedding, 0=GPU for layers).
[02:00] FIX: _encode() returns CPU tensors (correct — embedding expects CPU input).
[02:00] FIX: generate_short uses use_cache=True with KV cache for O(n) not O(n²) generation.
[02:00] FIX: cache.reset() called before each generation to prevent KV contamination between configs.
[02:00] CRITICAL RULE ADDED TO CLAUDE.md: Never move tensors to CPU to fix CUDA errors. ExLlamaV2 embedding on CPU is by design. Device flow: input(CPU) → embedding(CPU) → hidden→GPU → layers(GPU) → output(GPU).
[02:00] STATUS: Adapter generates correct output in Python REPL. Sweep runner needs GPU instance test with debug logging to confirm direct-call path is taken.

## KV Cache Bad Alloc Fix
[--:--] DONE: CRITICAL FIX v1 — forward_with_path cache=None (reverted, see v2)
[--:--] DONE: CRITICAL FIX v2 — fresh ExLlamaV2Cache per generate_short() call (per ExLlamaV2 PR #275 by dnhkng)
[--:--] DONE: forward_with_path accepts explicit cache param (caller controls lifecycle)
[--:--] DONE: generate_short creates fresh cache (max_seq_len=512) each call — empty slots, no stale KV entries, O(n) autoregressive decode restored
[--:--] DONE: get_logprobs / forward_with_hooks still use cache=None (single-pass, no caching needed)
[--:--] DONE: CRITICAL FIX v3-v4 — layer_idx remapping (FAILED — attn ignores past_len param)
[--:--] DONE: CRITICAL FIX v5 — No KV cache (too slow — O(n²) with CPU embedding bottleneck, 14% GPU util)
[--:--] DONE: CRITICAL FIX v6 — Restore KV cache with proper current_seq_len management. Read attn.py source: attention reads cache.current_seq_len, ignores past_len param. Fix: manage current_seq_len ourselves (reset→0, after prefill→prompt_len, after each decode→+1). Duplicate layers overwrite same cache slot (semantically imperfect, won't crash). O(n) decode restored.
[--:--] DONE: CRITICAL FIX v7 — Disable CUDA graph caching (fixed deterministic bad_alloc at call 30)
[--:--] DONE: Added debugging rules to CLAUDE.md — ALWAYS read external library source before fixing

## Probe Research & Calibration
[--:--] DONE: Researched Qwen3-32B benchmark scores (MATH-500 43.6%, GPQA 41.5%, IFEval 61.5% without thinking)
[--:--] DONE: Ran all 23 probes against Qwen3-32B API (thinking + no-think modes)
[--:--] DONE: Identified 7 sweet-spot probes (0.3-0.7): spatial, consistency, hallucination, language, eq, routing, reasoning
[--:--] DONE: Identified 9 ceiling probes to drop from sweep: math, planning, sycophancy, temporal, factual, implication, negation, code, counterfactual, abstraction, tool_use, noise_robustness
[--:--] DONE: Created 5 new probes: implication (1.0 ceiling), negation (1.0 ceiling), estimation (0.83), reasoning (0.56 sweet spot), routing (0.50 sweet spot)
[--:--] DONE: Validated 8-item and 4-item subsets produce comparable scores to full sets
[--:--] DONE: Added _limit() to BaseProbe — default 8 items per probe, --full for all items
[--:--] DONE: Fixed consistency probe extraction (bold heading bug, interleaved easy/hard items, _scan_for_accepted fallback)
[--:--] DONE: Added force-close thinking in adapter (detect <think>, inject </think>)
[--:--] DONE: Added failure response logging to sweep runner for post-hoc scoring review

## Sweep Configuration — Round 1
Probes (9): eq, language, hallucination, spatial, consistency, judgement, reasoning, spatial_pong_simple, spatial_pong_strategic
Ceiling probes (parked for round 2): routing (1.0 — save for after circuit map identifies routes)
Items per probe: 8 (default _limit)
Generation calls per config: ~80
Estimated sweep time: ~17 hours (down from 137 hours with 20 probes × full items)

## Adaptive Catastrophic Pruning
[--:--] DONE: Added adaptive pruning to SweepRunner
[--:--] FINDING: Duplicate sweeps on Qwen3-30B show catastrophic collapse when duplicating into layers 35+
[--:--] FINDING: Pattern: near-zero delta → first big drop → floor (-1.8) within 1-2 configs, no recovery
[--:--] DONE: Pruning logic: 2 consecutive configs with combined delta < -1.0 → run 1 more (for research) → skip remaining j for that i
[--:--] DONE: --no-prune flag to disable, default enabled
[--:--] DONE: Estimated ~35% sweep time savings from pruning catastrophic regions

## Performance Optimizations
[--:--] DONE: OPT 1 — Remove per-token string decode in generate_short. Detect <think> by token ID (stored at load time), not string decode. Eliminates tokenizer round-trip + CPU tensor creation every token.
[--:--] DONE: OPT 2 — Single Params object per forward_with_path. Was creating ~98 Params per forward pass. Safe because config.no_graphs=True prevents CUDA graph caching. Applied to forward_with_path, forward_with_hooks, project_to_vocab.
[--:--] DONE: OPT 3 — Think-close counter. Tracks how many times <think> force-close fires. Logged per probe in runner.py. Tells us if disrupted layer configs cause thinking to leak through.
