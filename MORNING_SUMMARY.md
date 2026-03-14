# Morning Summary — 2026-03-14

## What Works End to End

The complete sweep pipeline runs from CLI invocation through heatmap generation:

```bash
python scripts/run_sweep.py --model mock --probes all --max-block 4 --max-layers 8 --mock
```

This produces:
- `results/latest/sweep_results.json` — full sweep results
- `results/latest/analysis/heatmap_*.png` — per-probe delta heatmaps (10 probes)
- `results/latest/analysis/skyline.png` — best delta per block size
- `results/latest/analysis/circuit_boundaries.json` — detected circuit regions

### All 10 Probes Implemented and Tested
| Probe | Status | Test Result |
|-------|--------|-------------|
| math | Complete (12 questions, partial credit) | PASS |
| code | Complete (6 challenges, unit test scoring) | PASS |
| eq | Complete (12 EQ-Bench scenarios, digit scoring) | PASS |
| factual | Complete (12 obscure facts, exact/near match) | PASS |
| spatial | Complete (20 generated boards, probability density oracle) | PASS |
| language | Complete (16 sentences, grammaticality judgment) | PASS |
| tool_use | Complete (12 scenarios, exact match) | PASS |
| holistic | Complete (12 analogies, semantic equivalence) | PASS |
| planning | Complete (13 scenarios incl. 5-step chains, pairwise scoring) | PASS |
| instruction | Complete (13 scenarios incl. 3 conflicting-constraint, fraction scoring) | PASS |

31/31 pytest tests pass.

### Infrastructure
- SweepRunner supports `adapter_class` injection (MockAdapter or ExLlamaV2)
- `--mock` and `--timeout` CLI flags added
- Per-config timeout with threading (default 30s)
- Checkpointing every 10 configs
- Heatmap generation with circuit boundary detection

## What Is Blocked

### ExLlamaV2 Adapter Validation
- **Status**: BLOCKED — `exllamav2` not installed (no CUDA GPU on this machine)
- **Error**: `ModuleNotFoundError: No module named 'exllamav2'`
- **To unblock**: Install on a machine with CUDA GPU, or rent a cloud GPU (Vast.ai)
- The adapter code at `sweep/exllama_adapter.py` is structurally sound but untested against the real library

### Real Model Sweep
- Cannot run until ExLlamaV2 adapter is validated
- Smallest viable test model: Qwen2.5-1.5B-Instruct

## Commands to Continue

```bash
# On a GPU machine:
pip install exllamav2
python -c "import exllamav2; print(exllamav2.__version__)"

# Download test model:
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct

# Validate adapter:
python -c "from sweep.exllama_adapter import ExLlamaV2LayerAdapter; a = ExLlamaV2LayerAdapter('models/Qwen2.5-1.5B-Instruct')"

# Smoke test with real model:
python scripts/run_sweep.py --model models/Qwen2.5-1.5B-Instruct --probes math spatial --max-block 6 --max-layers 12

# Full sweep:
python scripts/run_sweep.py --model models/Qwen2.5-1.5B-Instruct --probes all
```

## Decisions Needing Human Input

1. **Model selection for smoke test**: CLAUDE.md says 7B model, but 1.5B would be faster for validation. Which to use?
2. **Cloud GPU rental**: Ready to deploy via `scripts/bootstrap_cloud.sh` on Vast.ai. Proceed?
3. **ExLlamaV2 version**: The adapter's module access pattern (`model.modules`, `module.forward(hidden, cache, None, None)`) may differ across versions. Need to verify against the installed version's API.
4. **Spatial probe board count**: Currently generates 20 boards per run. Increase for production sweeps? (slower but more statistically stable)
