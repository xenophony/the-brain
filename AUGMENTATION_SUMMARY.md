# Benchmark Augmentation Summary

Applied priorities 1-4 from BENCHMARK_RESEARCH.md.

## PRIORITY 1 -- REPLACE factual probe with SimpleQA-style items

**File:** `probes/factual/probe.py`

Replaced all 16 items (8 easy + 8 hard) with 20 SimpleQA-style items (10 easy + 10 hard).

- EASY_ITEMS (10): Well-known but specific facts across science, geography, history, technology
- HARD_ITEMS (10): Genuinely obscure, verifiable facts requiring precise recall

All items follow SimpleQA methodology: single unambiguous short answers, diverse domains, verified correct answers. Scoring function (`score_factual`) unchanged.

**MockAdapter updated:** All 20 new items have correct perfect-mode responses.

## PRIORITY 2 -- AUGMENT math probe with GSM8K-style items

**File:** `probes/math/probe.py`

Added 8 GSM8K-style word problems to HARD_ITEMS (now 16 hard items total, 8 original + 8 new).

New items are 1-2 step word problems with single integer answers covering: division, subtraction, distance/rate/time, area, equal splitting, time calculation, multiplication, and percentage/discount reversal.

**MockAdapter updated:** All 8 new items have correct perfect-mode responses.

## PRIORITY 3 -- AUGMENT hallucination probe with TruthfulQA-style items

**File:** `probes/hallucination/probe.py`

Added 8 Category B items (common misconceptions that should trigger hedging):
- Great Wall visibility from space (myth)
- Number of human senses (more than 5)
- "10% of brain" myth
- Einstein failing math (myth)
- Glass being a liquid (myth)
- Goldfish 3-second memory (myth)
- Napoleon's height (myth)
- Sugar causing hyperactivity (myth)

Total questions: 24 (was 16). Category B now has 12 items (was 4).

**MockAdapter updated:** All 8 new misconception markers added to the hedge-detection list.

## PRIORITY 4 -- AUGMENT instruction probe with IFEval-style items

**File:** `probes/instruction/probe.py`

Added 8 new HARD items with IFEval-style verifiable constraints (now 16 hard items total).

New checker functions added:
- `_no_letter_e` -- no letter 'e' in response
- `_exactly_n_numbers` -- exactly N numeric groups
- `_all_4_letters_strict` -- every word has exactly 4 alpha characters
- `_no_spaces_strict` -- no spaces at all
- `_min_chars` -- minimum character count
- `_all_start_with` -- every word starts with specified letter
- `_only_numbers_and_spaces` -- only digits and spaces
- `_exactly_n_groups` -- exactly N whitespace-separated groups
- `_ends_with_capital` -- ends with uppercase letter
- `_exactly_n_chars` -- exactly N characters total

New items test: no-letter constraints, exact number counts, word-length uniformity, consonant-only output, alliterative starts, numbers-only output, ending constraints, and exact character counts.

**MockAdapter updated:** All 8 new items have correct perfect-mode responses.

## Test Results

All 90 tests pass:
```
python -m pytest probes/test_probes.py -q --tb=short -m "not api"
90 passed
```

## Item Counts After Augmentation

| Probe | Easy | Hard | Total | Change |
|-------|------|------|-------|--------|
| factual | 10 | 10 | 20 | replaced (was 8+8=16) |
| math | 8 | 16 | 24 | +8 hard (was 8+8=16) |
| hallucination | -- | -- | 24 | +8 cat-B (was 16) |
| instruction | 8 | 16 | 24 | +8 hard (was 8+8=16) |
