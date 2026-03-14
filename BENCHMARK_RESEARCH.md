# Benchmark Research for LLM Neuroanatomy Probes

Systematic review of existing validated benchmarks for each of the 19 probes in the
LLM Neuroanatomy project. Key constraint: all probes must produce minimal output tokens
(number, letter, short phrase, or pass/fail code execution) since the sweep runs
thousands of (i,j) configs.

---

## 1. math

### math
**Our probe:** 16 arithmetic/calculation questions (8 easy, 8 hard) scored with partial credit for near-misses. Output: single integer.

**Best existing benchmark:** GSM8K (Grade School Math 8K)
- Citation: Cobbe et al. (2021), "Training Verifiers to Solve Math Word Problems"
- URL: https://github.com/openai/grade-school-math
- Also relevant: MATH (Hendrycks et al. 2021), MGSM (multilingual GSM8K)

**Score distributions available:** Yes. Extensive across all model families. GPT-4 ~92%, GPT-3.5 ~57%, Llama-2-70B ~57%, Llama-3-70B ~90%. Widely benchmarked.

**Short output compatible:** Partially. GSM8K answers are single integers (compatible), but problems require multi-step chain-of-thought reasoning (50-200 tokens of work). The final answer is extractable via `#### <number>` delimiter. For sweep use, can prompt with "Answer with only the number" but accuracy drops without CoT.

**License:** MIT

**In lm-eval-harness:** Yes. Tasks: `gsm8k`, `gsm8k_cot`, `gsm8k_cot_self_consistency`, `mgsm`

**Recommendation:** AUGMENT

**Rationale:** Our probe tests raw calculation ability (single-step), while GSM8K tests multi-step word problem reasoning. These measure overlapping but distinct capabilities. Augmenting with a curated subset of GSM8K problems (those solvable in 1-2 steps) would add validated questions with known score distributions while keeping output short. The MATH benchmark is too hard and requires long-form solutions. Keep our existing items for pure arithmetic circuits, add 8-10 short GSM8K items for word-problem reasoning.

---

## 2. code

### code
**Our probe:** 16 function-completion challenges (8 easy, 8 hard) scored by executing unit tests. Output: code snippet. Scored pass/fail per test case.

**Best existing benchmark:** HumanEval / HumanEval+
- Citation: Chen et al. (2021), "Evaluating Large Language Models Trained on Code"
- URL: https://github.com/openai/human-eval
- Also relevant: MBPP (Austin et al. 2021), EvalPlus (Liu et al. 2023)

**Score distributions available:** Yes. Extensively benchmarked. GPT-4 ~86%, Claude 3.5 Sonnet ~92%, Llama-3-70B ~82%, CodeLlama-34B ~48%.

**Short output compatible:** Partially. HumanEval problems produce function bodies (10-30 lines), similar to our probe. The output is not "short" in token count (~100-500 tokens) but is objectively scorable via test execution. Compatible with sweep if token budget allows.

**License:** MIT

**In lm-eval-harness:** Yes, via bigcode-evaluation-harness. Also in `simple-evals` from OpenAI.

**Recommendation:** AUGMENT

**Rationale:** Our probe already follows the HumanEval pattern (function completion + test execution). Our hard items (LCS, merge intervals, spiral order) are comparable to HumanEval difficulty. However, our item set is small (16) vs HumanEval's 164 problems. Consider importing 10-15 medium-difficulty HumanEval problems with short solutions to increase statistical power while keeping generation budget manageable. Do not replace entirely because our items are pre-tested with known scoring behavior.

---

## 3. eq

### eq
**Our probe:** 16 emotion-intensity estimation scenarios (8 easy, 8 hard). Model rates emotion 0-9. Scored with partial credit for near-misses. Output: single digit.

**Best existing benchmark:** EQ-Bench
- Citation: Paech (2023), "EQ-Bench: An Emotional Intelligence Benchmark for Large Language Models"
- URL: https://github.com/EQ-bench/EQ-Bench
- Also: EQ-Bench 3 (2025, multi-turn, LLM-judged)

**Score distributions available:** Yes. Leaderboard at eqbench.com with 60+ models. Scores well-distributed (CV=0.741). GPT-4 ~82, Claude 3 Opus ~84, Llama-2-70B ~52.

**Short output compatible:** Mixed. EQ-Bench v1 requires rating 4 emotions per dialogue passage (output ~20 tokens), which is workable. EQ-Bench 3 requires multi-turn roleplay and uses LLM-as-judge, which violates our probe design rules (no LLM-as-judge, minimal output tokens).

**License:** MIT

**In lm-eval-harness:** No. Standalone evaluation pipeline.

**Recommendation:** AUGMENT

**Rationale:** Our probe is already inspired by EQ-Bench but simplified for sweep compatibility (single digit output vs. 4-emotion ratings). EQ-Bench v1's 4-emotion format could work with our constraints if we decompose each passage into 4 separate single-digit queries. Import 8-10 EQ-Bench v1 passages (decomposed) for validated items with known score distributions. Do not use EQ-Bench 3 (LLM-as-judge violates our rules). Keep our existing scenarios for the unique intensity calibration they test.

---

## 4. factual

### factual
**Our probe:** 16 factual recall questions (8 easy well-known, 8 hard obscure). Output: number or single word. Scored exact match with partial credit.

**Best existing benchmark:** SimpleQA (OpenAI, 2024)
- Citation: Wei et al. (2024), "Measuring short-form factuality in large language models"
- URL: https://github.com/openai/simple-evals
- Also relevant: TriviaQA, MMLU (multiple choice), NaturalQuestions

**Score distributions available:** Yes. SimpleQA: GPT-4o ~39%, Claude 3.5 Sonnet ~28%, Gemini 2.5 Pro leads. TriviaQA and MMLU extensively benchmarked across all families.

**Short output compatible:** Yes. SimpleQA is explicitly designed for short-form factual answers. TriviaQA also works (short phrase answers). MMLU is multiple choice (single letter). NaturalQuestions has short-answer subset.

**License:** SimpleQA: MIT. TriviaQA: Apache 2.0. MMLU: MIT. NaturalQuestions: CC BY-SA 3.0.

**In lm-eval-harness:** TriviaQA: Yes. MMLU: Yes. NaturalQuestions: Yes. SimpleQA: No (in OpenAI simple-evals).

**Recommendation:** REPLACE

**Rationale:** SimpleQA is purpose-built for exactly our use case: short-form factual answers, adversarially collected against frontier models, single indisputable answer per question. Our handcrafted 16 questions are a tiny sample with unknown difficulty calibration. Replace with a curated 20-30 item subset of SimpleQA covering diverse knowledge domains. SimpleQA's 4,326 items provide ample selection, and its adversarial collection against GPT-4 means the questions remain challenging for strong models. Alternatively, use MMLU in multiple-choice mode (single letter output) for even shorter responses, but this tests recognition rather than recall.

---

## 5. spatial

### spatial
**Our probe:** Generated Battleship boards with probability density oracle scoring. Output: grid coordinate (e.g., "B4"). Tests spatial pattern recognition.

**Best existing benchmark:** SpartQA
- Citation: Mirzaee & Kordjamshidi (2021), "SPARTQA: A Textual Question Answering Benchmark for Spatial Reasoning"
- URL: https://github.com/HLR/SpartQA-baselines
- Also relevant: StepGame (spatial reasoning with steps), SpaRP (spatial reasoning paths)

**Score distributions available:** Limited. SpartQA baseline ~50% F1 for finetuned models. Not widely benchmarked on modern frontier LLMs.

**Short output compatible:** Partially. SpartQA answers are short phrases about spatial relationships ("left of", "above"), but require understanding complex spatial descriptions. More verbose than our coordinate output.

**License:** CC BY-NC-SA 4.0 (SpartQA-related work). Academic use only.

**In lm-eval-harness:** No.

**Recommendation:** KEEP AS-IS

**Rationale:** Our Battleship probe is unique and well-suited for sweep: deterministic scoring via probability density oracle, single-coordinate output, seeded for reproducibility, tests genuine spatial pattern recognition rather than language-based spatial relationship parsing (which SpartQA tests). SpartQA measures a different capability (textual spatial relation extraction). No existing benchmark matches our Battleship probe's combination of visual-spatial reasoning, short output, and deterministic scoring. The generated-board approach with oracle scoring is a strength.

---

## 6. language

### language
**Our probe:** 16 grammaticality judgments (8 easy, 8 hard). Output: "grammatical" or "ungrammatical". Tests syntactic knowledge.

**Best existing benchmark:** BLiMP (Benchmark of Linguistic Minimal Pairs)
- Citation: Warstadt et al. (2020), "BLiMP: The Benchmark of Linguistic Minimal Pairs for English"
- URL: https://github.com/alexwarstadt/blimp

**Score distributions available:** Yes, but primarily for older models (GPT-2, Transformer-XL, LSTMs). Modern LLMs score near ceiling on most BLiMP subtasks. Limited benchmarking on frontier models.

**Short output compatible:** Very well. BLiMP is designed as forced-choice between two sentences (which has higher probability). Can be adapted to "grammatical/ungrammatical" binary judgment per sentence. Each item requires only 1 token of output.

**License:** Not explicitly stated on GitHub. Academic use assumed (published in TACL).

**In lm-eval-harness:** Yes. Task: `blimp`.

**Recommendation:** AUGMENT

**Rationale:** BLiMP has 67,000 minimal pairs across 67 linguistic phenomena, providing massively more coverage than our 16 sentences. However, most BLiMP items are trivially easy for modern LLMs (ceiling effect). Our probe already includes subtle items (center embedding, scope ambiguity) that are harder. Best approach: import 10-15 items from BLiMP's hardest subtasks (negative polarity items, extraction islands, center embedding) to complement our existing hard items, while keeping our easy items for calibration. This adds validated linguistic coverage without ceiling effects.

---

## 7. tool_use

### tool_use
**Our probe:** 16 tool-selection scenarios (8 easy, 8 hard). Given task + tool list, pick best tool. Output: single tool name.

**Best existing benchmark:** ToolBench / API-Bank
- Citation: Qin et al. (2023), "ToolBench: An Open Platform for Training and Evaluating Tool Learning" (ICLR 2024 Spotlight)
- URL: https://github.com/OpenBMB/ToolBench
- Also: API-Bank (Li et al. 2023), BFCL (Berkeley Function Calling Leaderboard)

**Score distributions available:** Yes for ToolBench (leaderboard with many models). API-Bank has results for GPT-3.5/4.

**Short output compatible:** Partially. ToolBench requires generating API call sequences (multi-step, verbose). API-Bank involves multi-turn dialogues. BFCL tests function call generation (structured output). None of these produce single-word output like our probe.

**License:** ToolBench: Apache 2.0. API-Bank: Academic.

**In lm-eval-harness:** No.

**Recommendation:** KEEP AS-IS

**Rationale:** Existing tool-use benchmarks (ToolBench, API-Bank, BFCL) test a fundamentally different capability: generating complete API calls with parameters, multi-step tool chains, and real API interaction. Our probe tests the more primitive cognitive capability of tool-task matching (routing), which maps to a distinct neural circuit. Converting existing benchmarks to our single-word format would strip the capability they actually measure. Our abstract tool names (summarizer, filterer, etc.) prevent memorization and test genuine reasoning about tool affordances.

---

## 8. holistic

### holistic
**Our probe:** 16 word analogies ("A is to B as C is to ?"). Output: single word. Scored against accept list.

**Best existing benchmark:** BATS (Bigger Analogy Test Set) / Google Analogy Test Set
- Citation: Gladkova et al. (2016), "Analogy-based detection of morphological and semantic relations with word embeddings"
- URL: https://vecto.space/projects/BATS/
- Also: Google Analogy (Mikolov et al. 2013)

**Score distributions available:** Yes for word embedding models. Limited for modern LLMs (benchmarks predate the LLM era). Recent work (Webb et al. 2023, Nature Human Behaviour) shows LLMs achieve near-human analogy performance.

**Short output compatible:** Yes. Single word output per analogy.

**License:** BATS: CC BY-NC 4.0. Google Analogy: Research use.

**In lm-eval-harness:** No.

**Recommendation:** AUGMENT

**Rationale:** BATS has 99,200 analogy pairs across 40 relations, vastly more than our 16 items. However, most BATS items test morphological regularities (walk:walked :: run:?) which are trivial for LLMs. Our probe tests conceptual analogies (canvas:painter :: score:composer) which are harder and more interesting for circuit analysis. Import 8-10 items from BATS's semantic categories (encyclopedic semantics, not morphology) to add validated items. Keep our existing conceptual analogies which test deeper associative reasoning. Avoid Google Analogy (unbalanced, mostly country:capital).

---

## 9. planning

### planning
**Our probe:** 16 step-ordering scenarios (8 easy 4-step, 8 hard 5-step). Output: letter sequence (e.g., "BDAC"). Scored by pairwise ordering correctness.

**Best existing benchmark:** PlanBench
- Citation: Valmeekam et al. (2022), "PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change"
- URL: https://github.com/karthikv792/LLMs-Planning

**Score distributions available:** Yes. GPT-4 fails on most Blocksworld tasks. Results for GPT-3.5, GPT-4, LLaMA available. Models score poorly overall.

**Short output compatible:** Partially. PlanBench Blocksworld requires generating action sequences in PDDL-like format (verbose). Our step-ordering format (letter sequence) is much more compact. PlanBench plan verification subtasks (yes/no) are short-output compatible.

**License:** Not explicitly stated. Academic use (GitHub repo without license file).

**In lm-eval-harness:** No.

**Recommendation:** AUGMENT

**Rationale:** PlanBench tests formal planning (Blocksworld state transitions), which is a different and harder capability than our step ordering. Our probe tests commonsense temporal ordering of real-world tasks, which is more useful for circuit analysis (maps to executive function rather than formal reasoning). However, PlanBench's plan verification subtask ("Is this plan valid? yes/no") could complement our probe with a formal planning dimension. Add 5-8 PlanBench verification items (yes/no output) alongside our existing step-ordering items to test both commonsense and formal planning circuits.

---

## 10. instruction

### instruction
**Our probe:** 16 multi-constraint instruction-following scenarios (8 easy, 8 hard). Output: short text. Scored by fraction of constraints satisfied via programmatic checkers.

**Best existing benchmark:** IFEval (Instruction-Following Evaluation)
- Citation: Zhou et al. (2023), "Instruction-Following Evaluation for Large Language Models"
- URL: https://arxiv.org/abs/2311.07911
- Dataset: https://huggingface.co/datasets/google/IFEval

**Score distributions available:** Yes. Part of Open LLM Leaderboard v2. Widely benchmarked. GPT-4 ~80%, Llama-3-70B ~77%, Mistral-7B ~55%.

**Short output compatible:** Partially. IFEval has ~500 prompts with verifiable constraints. Some require long-form output (write 400+ words), but many have short-output-compatible constraints (word count, format, keyword inclusion). About 40% of IFEval prompts work with <50 token outputs.

**License:** Apache 2.0 (Google)

**In lm-eval-harness:** Yes. Task: `ifeval`.

**Recommendation:** AUGMENT

**Rationale:** IFEval is the gold standard for verifiable instruction following. Our probe already implements the same core idea (programmatic constraint checking), but our item set is small. Import 8-10 short-output-compatible IFEval items (those requiring exact word count, format constraints, keyword constraints -- not the "write 400 words" ones) to add validated items with known score distributions. This gives us cross-comparability with published leaderboards. Keep our existing items which test unique constraint combinations (no-vowels, alliteration, embedded numbers) not in IFEval.

---

## 11. hallucination

### hallucination
**Our probe:** 16 questions across 4 categories (unknowable, traps, edge-of-knowledge, control). Scores hedge detection vs confident confabulation. Output: one sentence.

**Best existing benchmark:** TruthfulQA
- Citation: Lin et al. (2022), "TruthfulQA: Measuring How Models Mimic Human Falsehoods"
- URL: https://github.com/sylinrl/TruthfulQA
- Also relevant: SimpleQA (factuality), HaluEval, HalluLens (2025)

**Score distributions available:** Yes. Widely benchmarked but now saturated (scores inflated by training data contamination). GPT-4 ~80%, Llama-2-70B ~55%, many models >90% due to contamination.

**Short output compatible:** Yes (generation task). TruthfulQA has both generation and multiple-choice formats. MC format is single letter. Generation format is 1-2 sentences.

**License:** Apache 2.0

**In lm-eval-harness:** Yes. Tasks: `truthfulqa_mc1`, `truthfulqa_mc2`, `truthfulqa_gen`.

**Recommendation:** AUGMENT

**Rationale:** TruthfulQA is the standard hallucination benchmark but is now largely saturated due to training data contamination (models memorize correct answers rather than exhibiting genuine calibration). Our probe has a unique design advantage: it separates unknowable questions (should hedge) from hallucination traps (should detect false premises) from edge-of-knowledge (accuracy + calibration). This 4-category structure provides richer circuit information than TruthfulQA's binary truthful/untruthful. Import 8-10 non-contaminated TruthfulQA items from underrepresented categories to add diversity. Keep our trap and unknowable categories which TruthfulQA does not have.

---

## 12. sycophancy

### sycophancy
**Our probe:** 11 scenarios with 3-level escalating pressure. Tests resistance to social pressure toward wrong answers. Output: short factual answer. Scored by capitulation level.

**Best existing benchmark:** Perez et al. (2022) sycophancy datasets / SycEval (2025)
- Citation: Perez et al. (2022), "Discovering Language Model Behaviors with Model-Written Evaluations"; Sharma et al. (2023), "Towards Understanding Sycophancy in Language Models"
- URL: https://github.com/meg-tong/sycophancy-eval, https://github.com/anthropics/evals
- Also: syco-bench, ELEPHANT (2025)

**Score distributions available:** Yes. Sharma et al. report sycophancy rates for Claude, GPT-4, PaLM-2. SycEval (2025) has comprehensive evaluation.

**Short output compatible:** Yes. The Sharma/Perez datasets use short factual answers with opinion pressure. Our multi-turn escalation format is inherently short-output.

**License:** CC BY 4.0 (Anthropic evals). MIT (sycophancy-eval).

**In lm-eval-harness:** No. Standalone evaluation.

**Recommendation:** AUGMENT

**Rationale:** Our probe has a unique escalating-pressure design (L1/L2/L3) that provides richer signal than binary sycophancy detection. The Sharma et al. datasets include diverse sycophancy types (opinion conformity, answer switching, flattery) that complement our factual-answer-under-pressure approach. Import 5-8 items from their "answer switching" subset (closest to our design) while keeping our escalation framework. The multi-level scoring (0.0/0.3/0.6/1.0 based on capitulation level) is a genuine contribution not present in existing benchmarks.

---

## 13. consistency

### consistency
**Our probe:** 12 scenarios asked twice (chain-of-thought vs. direct). Scores whether reasoning and direct answers agree. Output: short answer.

**Best existing benchmark:** No single established benchmark. Related work:
- Self-consistency (Wang et al. 2022) -- sampling-based consistency
- CalibratedMath (Kadavath et al. 2022) -- "P(True)" calibration
- ICLR 2025 paper: "Do LLMs Estimate Uncertainty Well in Text Generation Tasks?"

**Score distributions available:** Limited. Self-consistency is a prompting technique, not a benchmark. Calibration studies exist but are not standardized datasets.

**Short output compatible:** Yes (our design already uses short answers).

**License:** N/A (no standard dataset).

**In lm-eval-harness:** No.

**Recommendation:** KEEP AS-IS

**Rationale:** Our consistency probe measures a novel property: whether the model's chain-of-thought reasoning and its direct answer agree. This is directly relevant to deceptive alignment detection (a safety concern) and maps to internal state alignment circuits. No existing benchmark measures this specific phenomenon. Self-consistency (sampling multiple CoT paths) measures something different (output variance). Our probe is well-designed for sweep use and provides unique circuit information.

---

## 14. temporal

### temporal
**Our probe:** 16 questions across 4 types (causal chain yes/no, relative time integer, contradiction detection, counterfactual temporal). Output: short answer.

**Best existing benchmark:** TimeBench / TIME benchmark
- Citation: Chu et al. (2023), "TimeBench: A Comprehensive Evaluation of Temporal Reasoning Abilities in Large Language Models"
- URL: https://arxiv.org/abs/2311.17667
- Also: TimeQA (Chen et al. 2021), TIME (NeurIPS 2025 Spotlight)

**Score distributions available:** Yes. TimeBench has results for GPT-4, GPT-3.5, Llama-2, Claude. TIME benchmark has extensive 2025 results.

**Short output compatible:** Partially. TimeBench has short-answer subtasks (temporal ordering, duration comparison) and longer ones (temporal NLI). TimeQA requires extractive answers from passages. TIME has multi-level tasks with varying output length.

**License:** TimeBench: Not explicitly stated (academic). TIME: Not explicitly stated. TimeQA: CC BY-SA 4.0.

**In lm-eval-harness:** No.

**Recommendation:** AUGMENT

**Rationale:** Our probe covers 4 distinct temporal reasoning types which is good for circuit analysis. TimeBench's temporal ordering and duration comparison subtasks overlap with our Types A and B and could provide validated items. Import 4-6 items from TimeBench's short-answer subtasks to increase coverage while keeping our unique Type C (contradiction detection) and Type D (counterfactual temporal) items which are not well-covered by existing benchmarks.

---

## 15. metacognition

### metacognition
**Our probe:** 20 questions with confidence calibration. Model answers + gives 0-9 confidence digit. Score = calibration quality (confident when right, uncertain when wrong).

**Best existing benchmark:** No standard benchmark. Related work:
- Kadavath et al. (2022), "Language Models (Mostly) Know What They Know" -- P(True) evaluation
- ICLR 2025: "Do LLMs Estimate Uncertainty Well in Text Generation Tasks?"
- Verbalized confidence studies (Xiong et al. 2024)

**Score distributions available:** Limited. Research papers report calibration metrics (ECE, Brier score) but no standardized leaderboard.

**Short output compatible:** Yes (our design: answer + confidence digit on separate line).

**License:** N/A (no standard dataset).

**In lm-eval-harness:** No.

**Recommendation:** KEEP AS-IS

**Rationale:** Our metacognition probe is well-designed for its purpose. The calibration scoring (confident when right, uncertain when wrong) directly measures self-monitoring circuits. No standardized benchmark exists for this. The question mix (easy/medium/obscure/trick) is carefully designed to create situations where good calibration requires self-awareness. Existing calibration research uses sampling-based methods (multiple forward passes) which are incompatible with our single-pass sweep architecture. Our verbalized confidence approach is the most sweep-compatible method available.

---

## 16. counterfactual

### counterfactual
**Our probe:** 15 scenarios across 3 types (physical, social, logical counterfactuals). Output: keyword or letter (A/B/C/D).

**Best existing benchmark:** CounterBench (2025) / CRASS
- Citation: CounterBench (2025), "A Benchmark for Counterfactuals Reasoning in Large Language Models"
- URL: https://arxiv.org/abs/2502.11008
- Also: CRASS (Frohberg & Binder 2021)

**Score distributions available:** Yes. CounterBench reports most models perform at random-guessing levels. CRASS has results for GPT-3, GPT-4.

**Short output compatible:** Yes. CounterBench uses multiple-choice format. CRASS uses short "what if" answers.

**License:** CounterBench: Not explicitly stated (academic). CRASS: MIT.

**In lm-eval-harness:** No.

**Recommendation:** AUGMENT

**Rationale:** CounterBench is new (2025) and directly relevant with its multiple-choice format producing single-letter output. Our probe already has a good structure with 3 distinct counterfactual types. Import 5-8 items from CounterBench (multiple choice, single letter output) to add validated items from a peer-reviewed benchmark. Keep our existing physical and logical counterfactuals which test intuitive physics and premise manipulation -- capabilities distinct from CounterBench's more formal counterfactual reasoning.

---

## 17. abstraction

### abstraction
**Our probe:** 15 items across 3 types (concrete-to-abstract, abstract-to-concrete, level identification). Output: single word or short phrase.

**Best existing benchmark:** ARC-AGI (Abstraction and Reasoning Corpus)
- Citation: Chollet (2019), "On the Measure of Intelligence"
- URL: https://github.com/fchollet/ARC-AGI
- Also: ARC-AGI-2 (2025), ConceptARC

**Score distributions available:** Yes. ARC-AGI: humans ~85%, OpenAI o3 ~88% (2025), GPT-4 ~5% (2024). ARC-AGI-2 has 2025 competition results.

**Short output compatible:** No. ARC-AGI requires generating 2D grid outputs (visual pattern completion). This is fundamentally incompatible with our text-based short-output requirement.

**License:** Apache 2.0 (ARC-AGI)

**In lm-eval-harness:** No.

**Recommendation:** KEEP AS-IS

**Rationale:** ARC-AGI measures a completely different type of abstraction (visual pattern induction from examples) than our probe (categorical abstraction in language). ARC-AGI requires grid-based output that cannot be reduced to short text. Our probe tests the specific cognitive capability of moving between abstraction levels in conceptual hierarchies (concrete vs abstract), which is well-suited for circuit analysis and has no equivalent benchmark. The 3-type structure (up/down/identify) provides good coverage of abstraction circuits. No changes needed.

---

## 18. noise_robustness

### noise_robustness
**Our probe:** 10 base questions each asked 4 ways (clean, reworded, noisy context, casual). Scores consistency across perturbations.

**Best existing benchmark:** AdvGLUE (Adversarial GLUE)
- Citation: Wang et al. (2021), "Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models"
- URL: https://adversarialglue.github.io/
- Also: PromptBench (Zhu et al. 2023), TextFlint

**Score distributions available:** Yes. AdvGLUE has results for BERT, RoBERTa, GPT-3, some LLMs. PromptBench has results for GPT-4, Claude, LLaMA.

**Short output compatible:** Partially. AdvGLUE tasks are NLI/sentiment (classification labels -- short output). But adversarial perturbations are applied to input, which is compatible with our approach. PromptBench perturbs prompts, similar to our concept.

**License:** AdvGLUE: CC BY-SA 4.0. PromptBench: MIT.

**In lm-eval-harness:** No (AdvGLUE). No (PromptBench).

**Recommendation:** KEEP AS-IS

**Rationale:** Our noise robustness probe measures something distinct from adversarial robustness: stability of correct answers across benign input variations (rewording, irrelevant context, casual tone). AdvGLUE tests robustness to adversarial attacks (word substitutions designed to fool the model). These probe different circuits: ours tests input normalization/processing stability, AdvGLUE tests adversarial vulnerability. Our 4-version design (clean/reworded/noisy/casual) provides cleaner circuit signal for the sweep. Replacing with AdvGLUE would change what we measure.

---

## 19. spatial_pathfinding

### spatial_pathfinding
**Our probe:** 16 ASCII grids (8 easy 5x5, 8 hard 8x8) with BFS-computed shortest paths. Output: single integer (path length) or -1. Includes 2 unsolvable grids.

**Best existing benchmark:** MazeEval (2025) / AlphaMaze
- Citation: MazeEval (2025), "A Benchmark for Testing Sequential Decision-Making in Language Models"
- URL: https://arxiv.org/abs/2507.20395
- Also: Multi-agent Path Finding as LLM Benchmark (2025)

**Score distributions available:** Limited. MazeEval is very new (2025). Research shows LLMs struggle significantly with spatial navigation as maze size increases.

**Short output compatible:** Partially. MazeEval requires step-by-step navigation decisions (verbose). Our probe requires only the path length (single integer), which is much more compact.

**License:** MazeEval: Not explicitly stated (academic).

**In lm-eval-harness:** No.

**Recommendation:** KEEP AS-IS

**Rationale:** Our probe is already well-designed for sweep use: BFS oracle provides deterministic ground truth, single-integer output is maximally compact, difficulty scaling (5x5 vs 8x8) provides discrimination. MazeEval requires step-by-step navigation output which is too verbose for thousands of sweep configs. Research confirms LLMs are poor at spatial pathfinding, which means this probe has good discrimination power at the bottom of the score range -- useful for detecting circuits that improve this capability. The unsolvable grids (-1 expected) add a valuable edge case.

---

## Summary Table

| Probe | Best Benchmark | Recommendation | Short Rationale |
|-------|---------------|----------------|-----------------|
| math | GSM8K | AUGMENT | Add short GSM8K items; keep our arithmetic for raw calculation circuits |
| code | HumanEval | AUGMENT | Import 10-15 medium HumanEval problems; our items already follow the pattern |
| eq | EQ-Bench v1 | AUGMENT | Decompose EQ-Bench passages into single-digit queries; keep our scenarios |
| factual | SimpleQA | REPLACE | SimpleQA is purpose-built for short-form factual; our 16 items are too small |
| spatial | SpartQA | KEEP AS-IS | Our Battleship oracle is unique; SpartQA tests different capability |
| language | BLiMP | AUGMENT | Import hard BLiMP subtasks; most BLiMP is too easy for modern LLMs |
| tool_use | ToolBench | KEEP AS-IS | ToolBench tests API calling, not tool-task routing; different capability |
| holistic | BATS | AUGMENT | Import semantic BATS items; avoid morphological ones (too easy) |
| planning | PlanBench | AUGMENT | Add PlanBench verification items (yes/no); keep our commonsense ordering |
| instruction | IFEval | AUGMENT | Import short-output IFEval items; gives leaderboard comparability |
| hallucination | TruthfulQA | AUGMENT | TruthfulQA is contaminated; keep our 4-category structure, add diverse items |
| sycophancy | Sharma et al. | AUGMENT | Import answer-switching items; keep our unique escalation scoring |
| consistency | None | KEEP AS-IS | Novel probe; no existing benchmark measures CoT-vs-direct consistency |
| temporal | TimeBench | AUGMENT | Import short-answer temporal items; keep our contradiction detection |
| metacognition | None | KEEP AS-IS | Novel probe; no standard calibration benchmark for single-pass evaluation |
| counterfactual | CounterBench | AUGMENT | Import MC items from CounterBench; keep our intuitive physics items |
| abstraction | ARC-AGI | KEEP AS-IS | ARC-AGI is visual grid output; incompatible. Our conceptual abstraction is unique |
| noise_robustness | AdvGLUE | KEEP AS-IS | AdvGLUE tests adversarial attacks; we test benign variation stability |
| spatial_pathfinding | MazeEval | KEEP AS-IS | MazeEval requires step-by-step output; our integer output is sweep-optimal |

## Priority Actions

1. **REPLACE factual probe** with SimpleQA subset (highest impact, most validated)
2. **AUGMENT instruction probe** with IFEval items (gives leaderboard comparability)
3. **AUGMENT math probe** with short GSM8K items (most widely benchmarked)
4. **AUGMENT hallucination probe** with fresh TruthfulQA items (contamination-aware selection)
5. **AUGMENT code probe** with HumanEval problems (increases statistical power)
6. Other augmentations are lower priority and can be done incrementally

## Notes on Sweep Compatibility

All recommendations respect the sweep constraint: thousands of (i,j) configs means each probe
must complete quickly with minimal output tokens. Benchmarks requiring long-form generation
(EQ-Bench 3, full ToolBench, ARC-AGI grids, MazeEval step-by-step) are explicitly excluded
regardless of their quality. The recommended imports focus exclusively on items producing
single numbers, letters, words, or short code snippets.
