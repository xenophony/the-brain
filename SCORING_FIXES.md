# Scoring Extraction Fixes (2026-03-14)

## Problem

Real model baselines showed impossibly low scores on two probes:
- **consistency**: claude-sonnet=0.167 (expected >0.4)
- **code**: claude-sonnet=0.000, llama-70b=0.000 (expected >0.5)

Root cause: scoring functions were only tested against MockAdapter's clean output.
Real models format responses differently (prose, code fences, verbose CoT).

## Fixes Applied

### 1. Code Probe (`probes/code/probe.py`)

**Problem**: `score_code()` failed when real models:
- Wrapped code in ` ```python ... ``` ` blocks
- Added prose before/after: "Here's the function:\n```python\ndef lcs..."
- The `body.startswith("def ")` check failed with any leading prose

**Fix**: New extraction pipeline:
1. Try extracting from ` ```python ... ``` ` code fence first
2. Search for `def func_name(` anywhere in the response
3. Extract from function start to end of indented body
4. Fall back to prepending the header (original behavior)

### 2. Consistency Probe (`probes/consistency/probe.py`)

**Problem**: `_extract_final_answer()` failed because:
- Used first occurrence of markers (caught mid-reasoning text, not final answer)
- Limited marker set missed "= 120 km/h", "**120**", "Final answer: 120"
- `max_new_tokens=80` too low for CoT -- models need 200+ tokens to reason

**Fixes**:
- Increased `max_new_tokens` for reasoning phase from 80 to 300
- Rewrote `_extract_final_answer()`:
  - Uses `rfind` (last occurrence) instead of first for all markers
  - Expanded marker set with domain-specific patterns
  - Falls back to `**bolded**` text, then last short line
- Rewrote `_answers_match()`:
  - New `_extract_key_value()` helper extracts the last number from each answer
  - Avoids false positives from substring matching on verbose responses

### 3. EQ Probe (`probes/eq/probe.py`)

**Problem**: `expected_digit_score()` in BaseProbe extracts the FIRST digit.
Real models write "I would rate this a 7" -- first digit might be from echoed text.

**Fix**: New `_extract_eq_digit()` function:
- Finds the LAST word-bounded single digit in the response
- Falls back to last digit character anywhere
- Dedicated `_eq_digit_score()` wrapper with same partial-credit logic

### 4. Spatial Probe (`probes/spatial/probe.py`)

**Verified**: The existing regex `re.search(r'([A-J])\s*(\d{1,2})', response)` already
handles all real model patterns ("The best move is B4.", "I'll target F3", etc.)
because it searches anywhere in the string. No changes needed.

### 5. Response Logging Infrastructure

- Added `BaseProbe.log_responses: bool = False` class-level flag
- `_make_result()` now accepts optional `item_results` parameter
- Code, EQ, consistency, and spatial probes collect per-item details when `log_responses=True`
- `run_baselines.py` sets `probe.log_responses = True` and saves item_results to
  `results/baselines/baseline_responses.json`
- Prints 2 sample responses per probe (one good score, one zero score if exists)

## Files Changed

- `probes/registry.py` -- `log_responses` flag, `_make_result()` item_results param
- `probes/code/probe.py` -- new extraction pipeline, logging
- `probes/consistency/probe.py` -- robust extraction, increased tokens, logging
- `probes/eq/probe.py` -- last-digit extraction, logging
- `probes/spatial/probe.py` -- logging only (extraction was fine)
- `scripts/run_baselines.py` -- enables logging, saves responses, prints samples

## Verification

All 81 existing tests pass. No changes to what probes measure -- only string parsing.
MockAdapter responses (clean format) still score perfectly.
