# API Keys Setup Guide

This document lists every API key needed to run baseline probe calibration against real models.

## Overview

The baseline runner (`scripts/run_baselines.py`) tests all 18 probes against multiple API-hosted models to calibrate difficulty before spending GPU money on sweeps.

**Recommended approach:** Use a single **OpenRouter** API key to access all models through one provider. Total estimated cost: **~$0.35-0.40** for all models.

---

## Primary: OpenRouter (recommended)

| | |
|---|---|
| **Service** | OpenRouter |
| **Models unlocked** | All 5: `llama-8b`, `llama-70b`, `qwen-30b`, `claude-sonnet`, `gemini-3-pro` |
| **Get it at** | https://openrouter.ai/keys |
| **Free tier** | Some open-weight models have free tiers; paid models billed per token |
| **Cost for baseline** | **~$0.35-0.40** (all 5 models combined) |
| **Environment variable** | `OPENROUTER_API_KEY` |

**Why OpenRouter:** One key replaces 4 separate provider keys. Unified billing, unified rate limits, access to all model families (Llama, Qwen, Claude, Gemini) through a single OpenAI-compatible API.

---

## Fallback: Individual Provider Keys (optional)

If you don't have an OpenRouter key, the baseline runner will fall back to individual provider keys for supported models.

### Groq API Key (fallback for llama-8b, llama-70b)

| | |
|---|---|
| **Service** | Groq Cloud |
| **Models unlocked** | `llama-8b` (Llama-3.1-8B), `llama-70b` (Llama-3.3-70B) |
| **Get it at** | https://console.groq.com/keys |
| **Free tier** | Yes — generous free tier with rate limits (30 req/min on free plan) |
| **Environment variable** | `GROQ_API_KEY` |

### Anthropic API Key (fallback for claude-sonnet)

| | |
|---|---|
| **Service** | Anthropic API |
| **Models unlocked** | `claude-sonnet` (Claude Sonnet 4) |
| **Get it at** | https://console.anthropic.com/settings/keys |
| **Free tier** | $5 free credit on signup |
| **Environment variable** | `ANTHROPIC_API_KEY` |

### Google AI API Key (fallback for gemini-3-pro)

| | |
|---|---|
| **Service** | Google AI Studio / Gemini API |
| **Models unlocked** | `gemini-3-pro` (Gemini 2.5 Pro) |
| **Get it at** | https://aistudio.google.com/apikey |
| **Free tier** | Yes — 15 req/min free, generous daily quota |
| **Environment variable** | `GOOGLE_API_KEY` |

### Together AI API Key (additional open-weight models)

| | |
|---|---|
| **Service** | Together AI |
| **Models unlocked** | Open-weight models (Qwen, Mistral, etc.) via API |
| **Get it at** | https://api.together.ai/settings/api-keys |
| **Free tier** | $5 free credit on signup |
| **Environment variable** | `TOGETHER_API_KEY` |

**Note:** `qwen-30b` has no individual fallback — it is only available via OpenRouter.

---

## Cost Summary

| Model | Provider | Estimated Cost | Required? |
|-------|----------|---------------|-----------|
| llama-8b | OpenRouter | $0.004 | Recommended |
| llama-70b | OpenRouter | $0.022 | Recommended |
| qwen-30b | OpenRouter | $0.016 | Optional |
| claude-sonnet | OpenRouter | $0.223 | Recommended |
| gemini-3-pro | OpenRouter | $0.117 | Optional |
| **Total (all)** | | **~$0.38** | |
| **Total (minimum: llama-8b + claude-sonnet)** | | **~$0.23** | |

---

## Setup Instructions

### Step 1: Create .env file

```bash
cp .env.example .env
```

### Step 2: Add your keys

Edit `.env` with your actual keys:

```
# Primary — one key for all models (recommended)
OPENROUTER_API_KEY=sk-or-...

# Fallback — individual provider keys (optional)
ANTHROPIC_API_KEY=sk-ant-api03-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
TOGETHER_API_KEY=...
```

Only add keys for services you have accounts with. The baseline runner will try OpenRouter first, then fall back to individual provider keys.

### Step 3: Verify cost estimate

```bash
python scripts/estimate_cost.py --task baselines --models all
```

### Step 4: Dry run (verify setup without spending money)

```bash
python scripts/run_baselines.py --dry-run --models all
```

This will show which models are available (key present) and which will be skipped. No API calls are made.

### Step 5: Run baselines

```bash
# All available models
python scripts/run_baselines.py --models all

# Or specific models only
python scripts/run_baselines.py --models claude-sonnet llama-8b

# Or specific probes only
python scripts/run_baselines.py --models all --probes math factual eq spatial
```

Results are saved to `results/baselines/baseline_scores.json` with atomic checkpointing — if the script is interrupted, it resumes from where it left off.

### Step 6: Generate calibration report

```bash
python -c "
from analysis.calibration import generate_calibration_report
import json
scores = json.load(open('results/baselines/baseline_scores.json'))
generate_calibration_report(scores, 'results/baselines')
"
```

This produces `results/baselines/CALIBRATION_REPORT.md` identifying:
- Probes that are too easy (ceiling effects on best model)
- Probes that are too hard (floor effects on all models)
- Probes with low dynamic range (don't discriminate between models)
- Probe pairs that are too correlated (may be measuring the same thing)

---

## Minimum Viable Baseline

If you want to spend the absolute minimum:

1. Get an **OpenRouter key** at https://openrouter.ai/keys
2. Run: `python scripts/run_baselines.py --models llama-8b claude-sonnet`
3. Cost: **~$0.23** (almost entirely Claude Sonnet)

This gives you a weak-vs-strong comparison for all 18 probes.
