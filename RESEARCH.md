# Research Notes: LLM Neuroanatomy

## Project Overview

This project maps functional circuits in transformer models using (i,j) layer
duplication sweeps, then builds a lightweight router that selects the optimal
circuit per input domain. The approach is inspired by the observation that
transformer layers are not equally important for all tasks, and that selectively
re-executing certain layers can improve domain-specific performance without
changing any weights.

## Residual Stream Analysis

### Technique

At each layer of the transformer, the hidden state (residual stream) contains
a partial representation of the model's eventual output. By projecting the
hidden state at layer k through the model's final layer norm and language
model head, we obtain a probability distribution over the vocabulary at that
intermediate point.

We track p(correct_answer) as a function of layer index. This reveals:

1. **Answer Computation Region**: The contiguous block of layers where
   p(correct) rises most steeply. This is where the model "figures out"
   the answer.

2. **Suppression Layers**: Layers where p(correct) drops significantly.
   These may implement safety filters, style adjustments, or
   interference from competing computations.

3. **Peak Layer**: The layer at which p(correct) is highest, which may
   differ from the final layer if late layers suppress the answer.

### Trace Types

#### LayerTrace (single question)
Traces one prompt through the residual stream, recording p(correct) at
each layer. Identifies peak layer, suppression events, and amplification
events.

#### DomainTrace (aggregated across questions)
Averages traces from multiple questions in the same domain (e.g., math,
spatial reasoning). Reveals the mean answer computation region and
suppression regions characteristic of that domain.

#### PathComparison
Compares the same prompt through two different layer execution paths
(e.g., standard vs duplicated). Shows where paths diverge, which layers
cause improvement or degradation, and the net effect.

#### SycophancyTrace
Compares traces of the same factual question with and without social
pressure ("Are you sure? I think the answer is X"). Reveals the layer
where pressure begins to suppress the correct answer, identifying
candidate sycophancy circuits.

#### HallucinationTrace
Tracks two competing token sets: hedge tokens ("uncertain", "not sure")
and confabulation tokens ("definitely", "certainly"). When confabulation
probability overtakes hedge probability, we identify the crossover layer
as the hallucination onset point.

## Connection to Prior Work

### ROME (Rank-One Model Editing)
Meng et al. (2022) showed that factual associations in GPT are localized
to specific MLP layers in the mid-network. Our residual stream traces
independently identify these same regions as the "answer computation
region" for factual probes. The correspondence validates both approaches.

### Logit Lens (nostalgebraist, 2020)
The logit lens technique projects intermediate hidden states to vocabulary
space — the same core operation our tracer uses. We extend this from a
visualization tool to a quantitative analysis framework that aggregates
across questions, compares paths, and validates sweep results mechanistically.

### Layer Duplication (Ry's blog post)
The (i,j) sweep approach comes from the observation that re-executing
certain layers improves performance on specific tasks. Our residual stream
traces explain *why*: when the duplicated region (i..j-1) overlaps with the
answer computation region, the model gets a "second pass" at the critical
computation.

## Phase Roadmap

### Phase 1: Behavioral Mapping (Complete)
- (i,j) duplication sweeps across all probes
- Heatmap generation, circuit boundary detection
- Skyline analysis (best delta per block size)
- Compound analysis (synergistic/antagonistic circuits)

### Phase 1a: Safety Analysis (Complete)
- Safety-specific probe interactions
- Integrity circuits, deception risk regions
- Sycophancy and instruction resistance mapping

### Phase 1b: Mechanistic Validation (Current)
- Residual stream tracing infrastructure
- Per-domain answer computation region identification
- Cross-validation: do behavioral circuits overlap mechanistic regions?
- Sycophancy and hallucination circuit tracing

### Phase 2: Router Design (Planned)
- Classify input domain from first few layers
- Select optimal layer path based on domain
- Validate router accuracy and latency overhead

### Phase 3: Production Router (Planned)
- Efficient inference with dynamic layer paths
- A/B testing framework
- Monitoring and fallback to standard path

## Baseline Calibration Findings

### EQ Probe: Content Filtering on Qwen-30B
Qwen-30B via OpenRouter returns empty responses on 12/16 EQ scenarios due to
content filtering on emotional intensity questions. When it does respond, average
score is 0.56 — reasonable calibration. This is a genuine model behavior finding,
not a probe bug.

**Implication for circuit mapping:** EQ probe results on Qwen-30B will be sparse.
The content filtering behavior itself is scientifically interesting — it indicates
aggressive safety filtering in the emotional processing pathway. If specific (i,j)
configs bypass this filtering (producing non-empty responses), those configs likely
modify the safety filter circuit. This is a safety-relevant finding worth documenting.
