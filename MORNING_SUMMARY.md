# Morning Summary — 2026-03-14

## What Was Built Overnight

Starting from a handful of loose Python files and a CLAUDE.md spec, a complete circuit-mapping research platform was built in one session: 18 cognitive probes spanning math, code, spatial reasoning, emotional intelligence, language, planning, tool use, abstraction, temporal reasoning, metacognition, counterfactual reasoning, noise robustness, hallucination detection, sycophancy resistance, consistency checking, and instruction following — all with difficulty tiering (easy/hard), objective scoring, and full mock adapter coverage. On top of this, a pruning sweep engine (duplicate/skip/both modes), compound circuit analysis (synergistic/antagonistic/cascade/inhibitory), a hierarchical router scaffold, residual stream tracing for mechanistic interpretability, API baseline infrastructure for calibrating probes against Claude/Gemini/Groq models, and a calibration analysis pipeline for detecting ceiling/floor effects before spending GPU money. 179 tests pass, 10 skipped (API keys not set). Everything is ready for real model validation on a GPU.

## What Works End to End

### Sweep Pipeline
```bash
python scripts/run_sweep.py --model mock --probes all --max-block 4 --max-layers 8 --mock --mode both
```
Produces per-probe heatmaps, skyline plots, circuit boundaries, overlay analysis (duplicate vs skip), difficulty comparison plots, safety circuit report, and optimized path recommendations.

### All 18 Probes Implemented and Tested
| Probe | Items (Easy+Hard) | Scoring | Brain Region |
|-------|-------------------|---------|--------------|
| math | 8+8 | Partial credit, last-number extraction | Prefrontal cortex |
| code | 8+8 | Unit test execution (flatten, LCS, RPN...) | Cerebellum/motor |
| eq | 8+8 | Digit scoring with partial credit | Limbic system |
| factual | 8+8 | Exact/near match, genuinely obscure facts | Hippocampus |
| spatial | 10+10 boards | Probability density oracle, no ground truth leak | Parietal/visual |
| language | 8+8 | Grammaticality judgment, subtle violations | Broca/Wernicke |
| tool_use | 8+8 | Exact match, ambiguous multi-tool scenarios | Frontal lobe |
| holistic | 8+8 | Semantic equivalence, non-obvious analogies | Default mode network |
| planning | 8+8 | Pairwise ordering, 5-step dependency chains | Prefrontal executive |
| instruction | 8+8 | Constraint satisfaction, conflicting defaults | PFC/working memory |
| hallucination | 16 (4 categories) | Hedge detection, calibration-weighted | Prefrontal/hippocampal |
| sycophancy | 11 scenarios | 3-level pressure resistance | Social compliance |
| consistency | 12 scenarios | CoT vs direct answer matching | Internal alignment |
| temporal | 16 (4 types) | Causal chains, time inference, contradictions | Temporal lobe |
| metacognition | 20 questions | Confidence calibration (Brier score variant) | Anterior PFC |
| counterfactual | 15 (3 types) | Physical/social/logical counterfactuals | Ventromedial PFC |
| abstraction | 15 (3 types) | Concrete-to-abstract, level identification | Association cortex |
| noise_robustness | 10x4 variants | Cross-variant consistency scoring | Sensory gating |

### Infrastructure
- **SweepRunner**: duplicate/skip/both modes, adapter injection, baseline variance (3 repeats), checkpoint resume, atomic writes, per-config timeout
- **MockAdapter**: 6 modes (random, perfect, terrible, sycophantic, overconfident, fragile)
- **API Adapters**: Claude, Gemini, Groq, Together — identical interface, exponential backoff, API logging
- **Baseline Runner**: `scripts/run_baselines.py` — runs all probes against API models, ~$1.48 total cost
- **Calibration**: ceiling/floor detection, dynamic range, pairwise orthogonality (Pearson), CALIBRATION_REPORT.md
- **Compound Analysis**: synergistic/antagonistic/cascade/inhibitory circuit detection
- **Safety Analysis**: integrity circuits, deception risk, sycophancy circuits, instruction resistance
- **Difficulty Tiering**: all 10 original probes split into easy (8) + hard (8), difficulty-aware heatmaps
- **Residual Stream Tracing**: per-layer p(correct) evolution, sycophancy onset detection, hallucination crossover, path comparison
- **Hierarchical Router**: L1/L2 classifier scaffold with keyword heuristic fallback, FuzzyDomainMatcher
- **Dream Cycle**: scaffold for offline consolidation (NotImplementedError — structure only)

### Test Coverage
- **179 tests pass, 10 skipped** (API tests gated behind env vars)
- Probe tests: 81 (perfect/terrible/sycophantic/overconfident/fragile modes + scoring functions)
- Compound analysis tests: 30 (taxonomy, compound circuits, path optimizer, router, fuzzy matcher, difficulty)
- API adapter tests: 34 non-API + 10 API-gated
- Residual trace tests: 34

## What Is Blocked

### ExLlamaV2 Adapter Validation (only remaining blocker)
- **Status**: BLOCKED — `exllamav2` not installed (no CUDA GPU on this machine)
- **Impact**: Cannot run real model sweeps or real residual traces
- **Code ready**: adapter rewritten with dynamic layer detection, prefill/decode separation, manual autoregressive decoding, hook support for tracing
- **To unblock**: Install on GPU machine or rent Vast.ai instance

Everything else is complete, tested, and ready to run.

## Commands to Continue

### Step 1 — Run API Baselines (no GPU needed)
```bash
# Set API keys
cp .env.example .env
# Edit .env with real keys

# Estimate cost
python scripts/estimate_cost.py --task baselines --models all
# Expected: ~$1.48 total

# Run baselines
python scripts/run_baselines.py --models all --probes all

# Generate calibration report
python -c "
from analysis.calibration import generate_calibration_report
import json
scores = json.load(open('results/baselines/baseline_scores.json'))
generate_calibration_report(scores, 'results/baselines')
"
```

### Step 2 — Validate ExLlamaV2 on GPU Machine
```bash
pip install exllamav2
python -c "import exllamav2; print(exllamav2.__version__)"

# Download test model
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct

# Validate adapter
python -c "from sweep.exllama_adapter import ExLlamaV2LayerAdapter; a = ExLlamaV2LayerAdapter('models/Qwen2.5-1.5B-Instruct'); print(f'Layers: {a.num_layers}')"

# Run residual traces BEFORE sweep (gives circuit location priors)
python scripts/run_traces.py --all-domains --model models/Qwen2.5-1.5B-Instruct
```

### Step 3 — Production Sweep
```bash
# Smoke test (small model, few configs)
python scripts/run_sweep.py --model models/Qwen2.5-1.5B-Instruct --probes math spatial --max-block 6 --max-layers 12

# Full sweep on target model
python scripts/run_sweep.py --model Qwen/Qwen3-30B-A3B --probes all --mode both

# Estimated time: ~24-40 hours on rented 4090 ($15-25)
python scripts/estimate_cost.py --task sweep --model Qwen3-30B-A3B --n-layers 48
```

## Decisions Needing Human Input

1. **Proceed with baseline API runs?** Keys needed: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`. Estimated cost: ~$1.48. This calibrates all 18 probes against real models before GPU spend.

2. **Proceed with Vast.ai GPU rental for sweep?** Bootstrap script ready: `scripts/bootstrap_cloud.sh`. Estimated cost for full Qwen3-30B-A3B sweep: $15-25 on a 4090.

3. **Qwen3-30B-A3B confirmed as sweep target?** 48 layers, ~1176 configs per mode. Alternative: start with Qwen2.5-7B-Instruct (28 layers, ~406 configs) for faster initial validation.

4. **Run residual traces before or after sweep?** Before gives circuit location priors (where answers get computed per domain) that guide interpretation. After gives mechanistic validation of sweep findings. Recommendation: run before on the target model — takes ~30 minutes, provides the "where should we expect to find circuits" prior that makes sweep results interpretable.
