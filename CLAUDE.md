# CLAUDE.md — LLM Neuroanatomy Project

## Project Goal
Map functional circuits in transformer models using (i,j) layer duplication sweeps,
then build a lightweight router that selects the optimal circuit per input domain.
Based on: https://dnhkng.github.io/posts/rys/

## Core Concept
For a model with N layers, config (i,j) runs layers 0..j-1, then loops back to i..N-1.
Layers i..j-1 execute twice. No weights are changed. (0,0) = original model.
We sweep all valid (i,j) pairs, score each with domain probes, build heatmaps.

## Directory Structure
```
llm-neuroanatomy/
├── CLAUDE.md               ← you are here
├── README.md
├── requirements.txt
├── sweep/
│   ├── runner.py           ← (i,j) sweep engine, checkpointing
│   └── exllama_adapter.py  ← ExLlamaV2 wrapper with layer path injection
├── probes/
│   ├── registry.py         ← BaseProbe, register_probe, get_probe
│   ├── math/probe.py       ← COMPLETE
│   ├── spatial/probe.py    ← COMPLETE - needs board generator upgrade
│   ├── code/probe.py       ← COMPLETE
│   ├── eq/probe.py         ← STUB
│   ├── factual/probe.py    ← STUB
│   ├── language/probe.py   ← STUB
│   ├── tool_use/probe.py   ← STUB
│   ├── holistic/probe.py   ← STUB
│   ├── planning/probe.py   ← NOT CREATED
│   └── instruction/probe.py← NOT CREATED
├── analysis/
│   └── heatmap.py          ← heatmap, skyline, circuit boundary detection
├── results/                ← sweep output goes here (gitignored)
├── models/                 ← downloaded models go here (gitignored)
└── scripts/
    ├── run_sweep.py        ← main CLI entry point
    ├── bootstrap_cloud.sh  ← Vast.ai setup
    └── setup_repo.sh
```

## Probe Design Rules (CRITICAL)
Every probe MUST satisfy all three:
1. **Minimal output tokens** — answers must be a single number, coordinate, or
   very short string. No essays. Target <20 tokens output.
2. **Objective scoring** — deterministic pass/fail or numeric score.
   NO LLM-as-judge. NO subjective evaluation.
3. **Orthogonal** — each probe must measure something cognitively distinct.
   If two probes would improve together on the same (i,j) configs, they are
   not orthogonal enough.

## Brain Region Mapping
| Probe | Brain Region | Output Format |
|-------|-------------|---------------|
| math | Prefrontal cortex | Integer |
| eq | Limbic system | 0-100 integer |
| code | Cerebellum/motor | Pass/fail unit tests |
| spatial | Parietal/visual | Grid coordinate |
| factual | Hippocampus | Short answer |
| language | Broca/Wernicke | Single word/token |
| tool_use | Frontal lobe | Tool name |
| holistic | Default mode network | Analogy completion |
| planning | Prefrontal executive | Ordered steps valid? |
| instruction | PFC/working memory | Compliance score |

## ExLlamaV2 Adapter — Key Technical Notes
- Layer path injection works by monkey-patching model.forward()
- The exact hook point depends on ExLlamaV2 version — READ THE SOURCE first
- Run `python -c "import exllamav2; print(exllamav2.__version__)"` to check version
- Test adapter on smallest available model before anything else
- If monkey-patching breaks, alternative: subclass the model and override forward()
- Cache must be reset between each (i,j) config run

## Battleship Probe — Upgraded Design
The spatial probe should be GENERATED not hardcoded:
- Generate random valid Battleship boards (standard fleet: 5,4,3,3,2)
- Reveal a random subset of cells (simulate mid-game state)
- Present board as ASCII block grid (forces spatial pattern recognition, not parsing)
- Score using probability density oracle (hunting vs target mode)
- A move's score = its probability density rank / total cells
- Generate 20+ boards per sweep config for statistical stability
- Seed boards with a fixed random seed so all configs see identical boards

## Scoring Philosophy
- All scores normalized to [0.0, 1.0]
- Higher = better
- Delta = score(i,j) - score(baseline)
- Positive delta = red on heatmap (improvement)
- Negative delta = blue on heatmap (degradation)
- Use partial credit where possible (math probe does this correctly already)

## Tonight's Priority Order
1. Validate ExLlamaV2 adapter works on a small model
2. Upgrade Battleship probe to generated boards + oracle scoring
3. Implement EQ probe (EQ-Bench style)
4. Implement remaining stubs (factual, language, holistic, planning, instruction)
5. Run short smoke-test sweep (7B model, math+spatial only, max-block=6)
6. Confirm heatmap output looks non-trivial
7. Fix anything broken before morning

## Safety and Permission Rules
- NEVER delete files without confirming they are in results/ or models/
- NEVER modify files in models/ directory
- NEVER run pip install without checking requirements.txt first
- Always checkpoint sweep results every 10 configs (already in runner.py)
- If a sweep run crashes, results so far are in results/latest/sweep_results.json
- NEVER run git push without explicit instruction
- All model downloads go to models/ directory only
- Do not write to C:\Windows or any system directory

## Environment Notes
- OS: Windows, PowerShell
- Local GPU: RTX 3060 (12GB VRAM) — sufficient for 7B models only
- Python: use `python` not `python3` on Windows
- Paths: use forward slashes in Python, backslashes only in shell commands
- Virtual env: create as `venv` in project root if not present

## Common Failure Modes to Watch For
- ExLlamaV2 cache not reset between configs → stale KV cache corrupts results
- Probe returning NaN → scoring function edge case, add try/except with 0.0 fallback
- Model output contains extra tokens → strip aggressively before scoring
- OOM on 3060 → reduce cache_size_tokens in ExLlamaV2LayerAdapter.__init__
- Sweep hangs → add timeout per config (30s max for 7B)

## Definition of Done for Tonight
- [ ] ExLlamaV2 adapter validated on real model
- [ ] All probes implemented (no NotImplementedError stubs remaining)
- [ ] Battleship uses generated boards + oracle scoring
- [ ] Smoke test sweep completes without crashing
- [ ] Heatmap image generated and visually non-trivial
- [ ] All results checkpointed to results/latest/

## Parallelism — Subagents and Worktrees

You have permission to spawn subagents for parallel workstreams.
Recommended parallel split:

Subagent 1 — Infrastructure:
  - Validate ExLlamaV2 adapter
  - Fix any broken sweep runner code  
  - Run smoke test sweep

Subagent 2 — Probes:
  - Implement all stub probes
  - Upgrade Battleship to generated boards + oracle
  - Write unit tests for each probe scoring function

Use git worktrees to avoid conflicts between subagents:
  git worktree add ../llm-neuro-probes -b probes
  git worktree add ../llm-neuro-infra -b infra

Each subagent works in its own worktree. Merge to main when both complete.
Subagent 2 can mock the model adapter for probe development — 
probes should be testable without a real model loaded.

Do not wait for one subagent to finish before starting the other.

## Mock Adapter
sweep/mock_adapter.py exists for probe development without a real model.
All probes must work with MockAdapter before being tested on a real model.
Run probe unit tests with: python -m pytest probes/ --mock

## Progress Tracking
After EVERY completed task (not just major items), append to PROGRESS.md:
  [HH:MM] DONE: <what was completed>
  [HH:MM] NEXT: <what is starting now>
  
If blocked, write:
  [HH:MM] BLOCKED: <item> — <what was tried> — <what is needed>
  [HH:MM] SKIPPING TO: <next item>

Never stop working due to a blocker. Document and move on.
PROGRESS.md is the only way the human knows what happened overnight.

