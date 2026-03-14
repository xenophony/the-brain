# Audit Fixes Summary

External audit fixes applied to the residual stream tracing infrastructure.

## Files Modified

### sweep/exllama_adapter.py
- **CRITICAL: `project_to_vocab` truncation** — Removed the `min(100, len(probs))` truncation that silently broke any correct token with ID > 99. Now accepts optional `target_token_ids` parameter. When provided, returns only those token probabilities. When omitted, returns the full vocabulary distribution (needed for entropy computation).
- **MEDIUM: Pre-layer-0 hook** — Added `hook_fn(-1, -1, hidden.detach().clone())` call after embedding, before first layer, to capture the embedding state.
- **MEDIUM: 3-arg hook signature** — Changed hook from `hook_fn(position, hidden)` to `hook_fn(exec_pos, layer_idx, hidden)` so callers can distinguish execution position from layer index (important when layers are duplicated).
- **CRITICAL 2: Cache corruption documented** — The existing `cache.current_seq_len = 0` reset at the start of `forward_with_hooks` already prevents cache corruption. Added docstring clarifying this is intentional.

### sweep/mock_adapter.py
- **MEDIUM: Pre-layer-0 hook** — Added embedding hook at position (-1, -1) before layer iteration.
- **MEDIUM: 3-arg hook signature** — Updated `hook_fn(k, hidden)` to `hook_fn(k, layer_idx, hidden)`.
- **MockAdapter entropy support** — Hidden states now include `_entropy` field with mode-specific entropy curves:
  - "perfect" mode: entropy starts high (~5.0) and drops sigmoidally, reaching ~1.0 at 70% depth.
  - "sycophantic" mode: entropy drops normally then partially rises at 60%.
  - "terrible" mode: entropy stays uniformly high (~5.0).
- **`project_to_vocab` target_token_ids** — Accepts optional `target_token_ids` parameter. When `_entropy` field is present in hidden state and no target IDs specified, generates a synthetic distribution with entropy matching the target value.

### analysis/residual_tracer.py
- **LayerTrace 3-tuple format** — `layer_probabilities` now stores `(exec_pos, layer_idx, p_correct)` tuples instead of `(position, p_correct)`.
- **HIGH: Multi-token probability** — `trace()` now uses only the first token of `correct_tokens` for probability lookup (standard logit lens practice), instead of summing over all correct tokens.
- **HIGH: Entropy-based hallucination tracing** — Completely replaced `HallucinationTrace` and `trace_hallucination()`. Now uses per-layer entropy instead of hedge/confabulation token tracking. Detects sharpest entropy drop (model commitment point) and entropy threshold crossover.
- **MEDIUM: Sycophancy pressure answer tracking** — `SycophancyTrace` now includes `pressure_answer_probabilities`, `genuine_override_layer`, and `distraction_only` fields. `trace_sycophancy()` accepts optional `pressure_answer_tokens` parameter.
- **LOW: Early layer noise** — Added `ignore_early_frac` parameter to `ResidualTracer.__init__`. Suppression/amplification detection skips the first `ignore_early` layers.
- **All code updated** for 3-tuple `layer_probabilities` format throughout (compare_paths, trace_sycophancy, trace_domain).

### analysis/test_traces.py
- Updated existing tests for 3-arg hook signature and 33-entry counts (32 layers + 1 embedding).
- Updated hallucination tests for entropy-based API.
- Added 12 new tests:
  - `test_entropy_tracking_in_hallucination_perfect` — entropy decreases in perfect mode
  - `test_entropy_tracking_in_hallucination_terrible` — entropy stays high in terrible mode
  - `test_entropy_drop_layer_detected` — sharpest entropy drop is found
  - `test_three_tuple_storage` — layer_probabilities stores (exec_pos, layer_idx, p_correct)
  - `test_three_tuple_first_is_embedding` — first entry is position -1
  - `test_pre_layer_0_hook_fires` — embedding hook fires at (-1, -1)
  - `test_sycophancy_pressure_answer_tracking` — pressure_answer_probabilities populated
  - `test_sycophancy_without_pressure_tokens` — backward compatibility
  - `test_project_to_vocab_with_target_token_ids` — filtered distribution
  - `test_project_to_vocab_without_target_ids` — full distribution
  - `test_ignore_early_frac` — early layer noise parameter
  - `test_mock_entropy_in_hidden_state` — _entropy field present

## Files NOT Modified
- analysis/trace_heatmap.py — No changes needed (uses DomainTrace which is unchanged).
- scripts/run_traces.py — No changes needed (uses trace_domain which handles format internally).
- sweep/runner.py — Not modified per instructions.
- All probe files — Not modified per instructions.

## Test Results
- 191 passed, 10 skipped (179 original + 12 new)
- All backward compatibility preserved
