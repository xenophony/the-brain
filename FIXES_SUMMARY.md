# Fixes Summary — Pre-Production Sweep

## BLOCKER 1 — KV Cache Architecture (sweep/exllama_adapter.py)
**Problem:** Cache keyed by layer_index. When layer i runs twice in a duplicated path, both executions write to the same cache slot, corrupting attention.
**Fix:** Restructured `forward_with_path()` to accept a `prefill` flag. During prefill (full prompt), cache is reset. During decode (single new token), cache accumulates normally. Each forward call passes through layers sequentially, and the cache grows by sequence position, not by layer identity. The adapter now handles the cache position tracking correctly for paths where layers repeat.
**Status:** Code updated, needs validation with real ExLlamaV2 on GPU.

## BLOCKER 2 — num_layers Detection (sweep/exllama_adapter.py)
**Problem:** `modules_dict` does not exist in ExLlamaV2.
**Fix:** Dynamic detection using `isinstance(m, ExLlamaV2DecoderLayer)`:
```python
layer_indices = [i for i, m in enumerate(all_modules) if isinstance(m, ExLlamaV2DecoderLayer)]
```
Pre/post-layer modules (embedding, norm, lm_head) are now detected dynamically rather than assumed at fixed positions.
**Status:** Code updated, needs validation on GPU.

## BLOCKER 3 — Autoregressive Cache Reset (sweep/exllama_adapter.py)
**Problem:** `generate_short` used the ExLlamaV2 generator which reset cache on every forward call, making each token independent.
**Fix:** Replaced with manual autoregressive decoding loop. Prefill runs full prompt with `cache.current_seq_len = 0`, then each decode step runs a single token without resetting. Proper EOS detection added.
**Status:** Code updated, needs validation on GPU.

## BLOCKER 4 — Code Probe Ceiling (probes/code/probe.py)
**Problem:** All 6 challenges trivially solvable by 30B models → 1.0 baseline → flat heatmap.
**Fix:** Replaced with 7 harder challenges:
- `flatten(lst)` — deep recursive list flattening
- `lcs(a, b)` — longest common subsequence (DP)
- `balanced(s)` — bracket matching with 3 types
- `merge_intervals()` — overlapping interval merging
- `spiral_order(matrix)` — 2D spiral traversal
- `eval_rpn(tokens)` — reverse Polish notation evaluator
- `permutations(nums)` — generate all permutations

Increased `max_new_tokens` from 80 to 150 for longer functions.

## BLOCKER 5 — Spatial Oracle Ground Truth Leak (probes/spatial/probe.py)
**Problem:** `compute_probability_density()` used actual ship positions (`ship_cells`, `placements`) to determine which ships are remaining. This leaks hidden information — a move could score low because the oracle knows the ship is elsewhere.
**Fix:** Rewrote oracle to use ONLY visible board state (hits/misses/unknowns). Enumerates all possible standard fleet placements (5,4,3,3,2) consistent with visible board. No ground truth positions used. `score_response()` signature changed to `score_response(response, board)` — no longer accepts `ship_cells` or `placements`.

## HIGH PRIORITY — Baseline Variance Estimation (sweep/runner.py)
**Problem:** Single baseline measurement can't distinguish signal from noise.
**Fix:** Added `baseline_repeats` config (default 3). Baseline runs N times, stores mean and std per probe. CLI flag: `--baseline-repeats N`.

## HIGH PRIORITY — Checkpoint Resume (sweep/runner.py)
**Problem:** Crash during long sweep = restart from zero.
**Fix:** Added `_load_checkpoint()` that reads existing `sweep_results.json`, populates `_completed_configs` set, and skips already-completed (i,j,mode) triples. CLI flag: `--resume`.

## MEDIUM — Math Extraction (probes/math/probe.py)
**Problem:** "17 * 23 = 391" parsed as 17 (first number), not 391.
**Fix:** Changed `re.search` to `re.findall`, take `matches[-1]` (last number).

## MEDIUM — Atomic Checkpointing (sweep/runner.py)
**Problem:** Writing JSON directly to output file → corrupt file on crash mid-write.
**Fix:** Write to `.tmp` file first, then `tmp.replace(out)` for atomic rename.

## MEDIUM — Factual Probe Difficulty (probes/factual/probe.py)
**Problem:** Common facts (bones in body, speed of light) → ceiling on 30B.
**Fix:** Replaced with genuinely obscure facts: gallium discovery year, Mohs hardness of topaz, capital of Vanuatu, ionization energy of hydrogen, densest element, SI unit of magnetic flux, etc. Also switched to last-number extraction.

## Mock Adapter Updates (sweep/mock_adapter.py)
Updated perfect-mode responses for all changed probes (new code challenges, new factual questions).
