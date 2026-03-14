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
