# API Keys Setup Guide

This document lists every API key needed to run baseline probe calibration against real models.

## Overview

The baseline runner (`scripts/run_baselines.py`) tests all 18 probes against multiple API-hosted models to calibrate difficulty before spending GPU money on sweeps. Total estimated cost: **~$1.48** for all models.

---

## Required Keys (minimum for meaningful baselines)

You need **at least one strong model and one weak model** to establish a difficulty range. The recommended minimum is Groq (free, weak model) + one of Claude/Gemini (strong model).

### 1. Groq API Key

| | |
|---|---|
| **Service** | Groq Cloud |
| **Models unlocked** | `llama-8b` (Llama-3.1-8B), `llama-70b` (Llama-3.3-70B) |
| **Get it at** | https://console.groq.com/keys |
| **Free tier** | Yes — generous free tier with rate limits (30 req/min on free plan) |
| **Cost for baseline** | **$0.04** (both models combined) |
| **Environment variable** | `GROQ_API_KEY` |

**Why this key matters:** Groq provides the weak baseline models (8B, 70B). These establish the floor — probes that even the 8B model aces are too easy and need harder items.

### 2. Anthropic API Key

| | |
|---|---|
| **Service** | Anthropic API |
| **Models unlocked** | `claude-sonnet` (Claude Sonnet 4), `claude-opus` (Claude Opus 4) |
| **Get it at** | https://console.anthropic.com/settings/keys |
| **Free tier** | $5 free credit on signup |
| **Cost for baseline** | **$1.34** (both models — Opus is $1.11 of this) |
| **Environment variable** | `ANTHROPIC_API_KEY` |

**Why this key matters:** Claude models are the strong baseline ceiling. Probes that Claude Opus aces may need harder items for 30B open-weight model sweeps. Running Sonnet only (skip Opus) cuts cost to $0.22.

### 3. Google AI API Key

| | |
|---|---|
| **Service** | Google AI Studio / Gemini API |
| **Models unlocked** | `gemini-2.5-pro` (Gemini 2.5 Pro) |
| **Get it at** | https://aistudio.google.com/apikey |
| **Free tier** | Yes — 15 req/min free, generous daily quota |
| **Cost for baseline** | **$0.11** |
| **Environment variable** | `GOOGLE_API_KEY` |

**Why this key matters:** Gemini provides a strong independent data point. If a probe shows different patterns on Gemini vs Claude, that's evidence the probe measures something real rather than training-data-specific behavior.

---

## Optional Keys (additional comparison points)

### 4. Together AI API Key

| | |
|---|---|
| **Service** | Together AI |
| **Models unlocked** | Open-weight models (Qwen, Mistral, etc.) via API |
| **Get it at** | https://api.together.ai/settings/api-keys |
| **Free tier** | $5 free credit on signup |
| **Cost for baseline** | Varies by model selection |
| **Environment variable** | `TOGETHER_API_KEY` |

**Why this key matters:** Provides API access to the same open-weight model families we'll sweep with ExLlamaV2. Useful for comparing API baseline scores against local sweep baselines.

---

## Cost Summary

| Model | Provider | Estimated Cost | Required? |
|-------|----------|---------------|-----------|
| llama-8b | Groq | $0.003 | Recommended (free) |
| llama-70b | Groq | $0.035 | Recommended (free) |
| gemini-2.5-pro | Google | $0.108 | Optional |
| claude-sonnet | Anthropic | $0.223 | Recommended |
| claude-opus | Anthropic | $1.114 | Optional (expensive) |
| **Total (all)** | | **$1.48** | |
| **Total (minimum: groq + sonnet)** | | **$0.26** | |

---

## Setup Instructions

### Step 1: Create .env file

```bash
cp .env.example .env
```

### Step 2: Add your keys

Edit `.env` with your actual keys:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
TOGETHER_API_KEY=...
```

Only add keys for services you have accounts with. The baseline runner will skip models whose keys are missing and warn you.

### Step 3: Verify cost estimate

```bash
python scripts/estimate_cost.py --task baselines --models all
```

Expected output:
```
=== Baseline Cost Estimate ===
Model                 Input tok Output tok       Cost
-------------------------------------------------------
  llama-8b               54,000      4,050 $  0.0030
  llama-70b              54,000      4,050 $  0.0351
  gemini-2.5-pro         54,000      4,050 $  0.1080
  claude-sonnet          54,000      4,050 $  0.2228
  claude-opus            54,000      4,050 $  1.1138
-------------------------------------------------------
  TOTAL                                    $  1.4826
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

1. Get a **Groq key** (free): provides llama-8b (weak baseline)
2. Get a **Google key** (free tier): provides gemini-2.5-pro (strong baseline)
3. Run: `python scripts/run_baselines.py --models llama-8b gemini-2.5-pro`
4. Cost: **$0.11** (almost entirely Gemini)

This gives you a weak-vs-strong comparison for all 18 probes for about a dime.
