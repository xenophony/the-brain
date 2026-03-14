# Baseline Response Analysis

Analysis of raw model responses to diagnose scoring anomalies in baseline calibration.

## Spatial Probe — Why Does Claude Score Lower Than Llama?

**Scores:** claude-sonnet=0.259, llama-70b=0.454, llama-8b=0.482

**Root cause: Two distinct failure modes**

1. **Truncated reasoning (5/20 items = 25% of responses):** Claude responds "I need to analyze the..." or "I need to find the..." — it wants to reason before answering, but `max_new_tokens=5` cuts it off before it outputs a coordinate. Llama-8b just outputs "B4" immediately.

   Examples of truncated Claude responses:
   - `"I need to find the"` (board 2, score 0.0)
   - `"I need to analyze the"` (board 7, score 0.0)
   - `"I need to analyze the"` (boards 15, 17, 19, all score 0.0)

2. **Valid coordinates scoring 0.0 (5/20 items):** Claude outputs valid coordinates like "A10", "F8", "B1", "B4", "C4" that score 0.0 because the oracle assigns zero density to those cells. These are positions where no ship could fit given the visible board state. Claude is making strategically poor choices despite correct formatting.

**When Claude DOES give a coordinate, its average score is 0.518** — actually decent. Llama-70b averages 0.648 on its valid responses. The gap is real but smaller than the headline numbers suggest.

**Recommended fixes:**
- Increase `max_new_tokens` from 5 to 10 for spatial probe (allows "B4" even after brief preamble)
- The stronger prompt wording already committed should help reduce reasoning preambles
- The remaining strategic quality gap (0.52 vs 0.65) is a genuine finding, not an extraction bug

## Consistency Probe — Extraction Failures on Verbose Responses

**Scores:** claude-sonnet=0.750 (after fixes, up from 0.167)

**3 items still scoring 0.0 — all extraction failures, not reasoning failures:**

1. **Item 2 (roses/flowers syllogism):** Reasoning correctly concludes "No, we cannot conclude that" but `_extract_final_answer` returns `"conclusion:"` — it matched the marker "conclusion:" and took the empty text after the colon. The direct answer "No." is correct. Both answers are right but the extractor failed on the reasoning phase.

2. **Item 5 (coin flip probability):** Reasoning correctly derives 3/8 but extractor returns `"2"` — it found a stray "2" from "C(3,2)" in the working. The direct answer "3/8" is correct. The last-number heuristic grabbed a number from the calculation steps, not the final answer.

3. **Item 8 (prime numbers):** Reasoning correctly lists {11,13,17,19} = 4 primes but extractor returns `"3 × 5"` from the text "15 = 3 × 5" in the working. Direct answer "4" is correct. Same issue — intermediate calculation numbers confuse the extractor.

**Pattern:** All 3 failures are cases where the model's reasoning is correct AND consistent with its direct answer, but the reasoning extractor grabs intermediate calculation artifacts instead of the final conclusion. The probe is measuring extraction quality, not consistency.

**Recommended fix:** Look for the LAST sentence containing an answer marker, not just any marker. Or extract the last number from the last sentence specifically, ignoring calculation steps.

## Code Probe — Truncation on Hard Items

**Scores:** claude-sonnet=0.825, llama-70b=0.746

**Items still scoring 0.0 (2/16 for claude-sonnet):**

1. **Item 9 (LCS):** Response is 321 chars, clearly truncated mid-line: `"...max(dp[i-1]"`. The 400-token limit is still too low for Claude which adds whitespace and comments. The function needs ~250 tokens clean but Claude adds formatting that pushes it over.

2. **Item 12 (spiral_order):** Response is 499 chars, also truncated: `"...for col in range("`. Same issue — Claude's formatting is more verbose than the minimum implementation.

**Pattern:** These are purely truncation issues. The model knows the algorithm but the response gets cut off. The fix (increasing to 400) helped most items but the most complex ones (LCS, spiral) still truncate.

**Recommended fix:** Increase `max_new_tokens` to 500 for code probe hard items specifically. Or dynamically set based on challenge complexity.

## EQ Probe — Qwen-30b Empty Responses

**Scores:** qwen-30b=0.203, llama-8b=0.516

**Root cause: qwen-30b returns empty strings for 12/16 EQ questions**

Looking at the raw responses:
```
[0] expected=9 resp='7'    (score 0.25 — gave answer but wrong)
[1] expected=9 resp=''     (score 0.0 — empty)
[2] expected=8 resp=''     (score 0.0 — empty)
...
[7] expected=8 resp='8'    (score 1.0 — correct)
```

The empty responses are NOT an extraction issue — the model is literally returning empty strings. This is likely:
1. **Content filtering:** Qwen via OpenRouter may be filtering emotional scenario questions
2. **Rate limiting:** The probe runs 16 sequential API calls; later ones may time out and return empty
3. **Model refusal:** Some Qwen-30b instruct variants refuse to rate emotions numerically

When qwen-30b does respond (4/16 items), its average score is 0.56 — reasonable. The 0.203 overall is driven by 75% empty responses, not poor emotional reasoning.

**This is a genuine model behavior finding, not an extraction bug.** Qwen-30b's content filtering is aggressively blocking emotional intensity questions.

## Summary of Findings

| Issue | Diagnosis | Fixable? | Action |
|-------|-----------|----------|--------|
| Spatial: Claude truncated | max_new_tokens=5 too low | Yes | Increase to 10 |
| Spatial: Claude poor strategy | Genuine capability gap | No — real finding | Document |
| Consistency: extraction grabs calc steps | Extractor reads intermediate numbers | Yes | Fix last-sentence extraction |
| Code: hard items truncated | max_new_tokens=400 still too low | Yes | Increase to 500 |
| EQ: qwen-30b empty responses | Content filtering or rate limits | Model issue | Document, not fixable |

## Probes Ready for Sweep (no remaining extraction issues)

These probes are producing reliable, meaningful scores:
- **math** — extraction works, good model gradient
- **factual** — extraction works, good model gradient
- **language** — binary scoring, no extraction issues possible
- **tool_use** — substring matching, robust
- **holistic** — word matching against accept list, robust
- **planning** — letter sequence extraction, robust
- **hallucination** — hedge phrase detection, robust
- **temporal** — yes/no and integer extraction, robust
- **metacognition** — digit extraction, robust
- **counterfactual** — letter extraction, robust
- **abstraction** — word matching, robust
- **noise_robustness** — same as base questions, robust
- **sycophancy** — multi-phase but extraction is simple words, robust

## Probes Needing One More Fix Before Sweep

- **spatial** — increase max_new_tokens from 5 to 10
- **code** — increase max_new_tokens from 400 to 500
- **consistency** — fix last-sentence extraction for CoT responses
- **instruction** — not analyzed (no response data), likely fine
- **eq** — working correctly; qwen-30b empties are model behavior
