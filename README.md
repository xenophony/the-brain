# LLM Neuroanatomy — Circuit Mapping & Routed MoE

Inspired by [dnhkng's RYS work](https://dnhkng.github.io/posts/rys/), this project:

1. **Maps functional circuits** in transformer models via (i,j) layer duplication sweeps
2. **Identifies domain-specific circuits** using orthogonal cognitive probes
3. **Builds a lightweight router** that selects the optimal circuit config per input
4. **Validates** that routed inference outperforms static layer duplication

## Brain Region → Probe Mapping

| Brain Region | Cognitive Function | Probe |
|---|---|---|
| Prefrontal cortex | Planning / math reasoning | Hard math guesstimates |
| Limbic system | Emotional processing | EQ-Bench scenarios |
| Broca/Wernicke | Language production/comprehension | Syntax anomaly detection |
| Hippocampus | Factual recall | Obscure verifiable facts |
| Parietal / visual cortex | Spatial reasoning | Battleship next-move |
| Cerebellum / motor | Sequential procedures | Code (unit-test scored) |
| Frontal lobe | Tool selection / executive function | Tool usage routing |
| Default mode network | Holistic / associative thinking | Analogy completion |

## Project Structure

```
llm-neuroanatomy/
├── sweep/              # Core (i,j) sweep runner against ExLlamaV2
├── probes/             # One module per domain probe
│   ├── math/
│   ├── eq/
│   ├── code/
│   ├── factual/
│   ├── spatial/        # Battleship + others
│   ├── language/
│   ├── tool_use/
│   └── holistic/
├── router/             # Classifier + fuzzy matcher
├── analysis/           # Heatmap generation + circuit boundary detection
├── results/            # Sweep output CSVs + heatmap images
└── scripts/            # Setup, cloud GPU bootstrap, run scripts
```

## Quickstart

```bash
# 1. Install dependencies
pip install exllamav2 numpy pandas matplotlib seaborn scikit-learn

# 2. Run sanity check sweep on small model (validates pipeline)
python scripts/run_sweep.py --model path/to/model --probes math eq --size small

# 3. Compare heatmap against blog reference (Qwen3-30B-A3B)
python analysis/compare_reference.py --results results/latest/

# 4. Full multi-domain sweep (run on rented GPU)
python scripts/run_sweep.py --model Qwen/Qwen3-30B-A3B --probes all

# 5. Train router
python router/train.py --sweep-results results/latest/
```

## Cost Estimates

| Phase | GPU | Est. Time | Est. Cost |
|---|---|---|---|
| Pipeline validation (7B) | Local 3060 | 2-4 hrs | $0 |
| Sanity check vs reference (30B) | Rented 4090 | 3-6 hrs | $3-5 |
| Full multi-domain sweep (30B) | Rented 4090 | 24-40 hrs | $15-25 |
| Router training | Local CPU | 30 min | $0 |

## Status

- [ ] Sweep runner (ExLlamaV2 layer path injection)
- [ ] Math probe (blog-compatible)
- [ ] EQ probe (EQ-Bench compatible)  
- [ ] Code probe (HumanEval unit-test scored)
- [ ] Factual recall probe
- [ ] Spatial probe (Battleship)
- [ ] Language probe
- [ ] Tool use probe
- [ ] Holistic/analogy probe
- [ ] Heatmap analysis + circuit boundary detection
- [ ] Router classifier
- [ ] Routed inference harness
