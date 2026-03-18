#!/usr/bin/env python
"""
Harvest full generative responses (including thinking) from logprob probes.

Runs generative probes on the model with thinking enabled, saves full responses
for later replay analysis.

Usage:
    python scripts/harvest_responses.py --model models/Qwen3-30B-A3B-exl2 --probes causal_logprob,logic_logprob
    python scripts/harvest_responses.py --mock --probes causal_logprob
    python scripts/harvest_responses.py --mock --probes all

Options:
    --model PATH        Model path (required unless --mock)
    --probes NAMES      Comma-separated probe names, or "all" for all logprob probes
    --output-dir DIR    Output directory (default: results/harvested/)
    --mock              Use MockAdapter for testing
    --max-items N       Max items per probe (default: 0 = all)
    --max-tokens N      Max tokens for generation (default: 200)
    --mode MODE         MockAdapter mode (default: perfect)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def extract_thinking(response: str) -> tuple[str, str]:
    """Extract thinking and answer from a response.

    Returns (thinking, answer_text) where thinking is the content inside
    <think>...</think> tags and answer_text is everything after.
    """
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        # Everything after the closing think tag
        after_think = response[think_match.end():].strip()
        return thinking, after_think
    # No thinking tags — entire response is the answer
    return "", response.strip()


def extract_answer(response: str) -> str:
    """Extract the final answer from a response, stripping thinking tags.

    Handles:
      - "<think>...</think>\\nyes" -> "yes"
      - "yes" -> "yes"
      - "Yes." -> "yes"
      - Multi-word: take first word only
    """
    _, answer_text = extract_thinking(response)
    if not answer_text:
        return ""
    # Strip punctuation and take first word
    answer_text = answer_text.strip()
    # Remove leading/trailing punctuation
    answer_text = re.sub(r'^[^a-zA-Z0-9]+', '', answer_text)
    answer_text = re.sub(r'[^a-zA-Z0-9]+$', '', answer_text)
    # Take first word
    first_word = answer_text.split()[0] if answer_text.split() else ""
    return first_word.lower()


def check_correct(extracted: str, expected: str) -> bool:
    """Check if extracted answer matches expected, with fuzzy matching.

    Handles yes/no/true/false variants.
    """
    extracted = extracted.lower().strip()
    expected = expected.lower().strip()

    if extracted == expected:
        return True

    # yes/true and no/false equivalences
    yes_variants = {"yes", "true", "correct", "right"}
    no_variants = {"no", "false", "incorrect", "wrong"}

    if expected in yes_variants and extracted in yes_variants:
        return True
    if expected in no_variants and extracted in no_variants:
        return True

    return False


def get_logprob_probe_names() -> list[str]:
    """Get all registered logprob probe names."""
    from probes.registry import list_probes
    all_probes = list_probes()
    return [name for name in all_probes if name.endswith("_logprob")]


def harvest_probe(model, probe, probe_name: str, max_tokens: int,
                  is_mock: bool) -> dict:
    """Harvest responses for a single probe.

    Returns a harvest dict with probe_name, model info, and items.
    """
    items = probe._limit(probe.ITEMS)
    choices = probe.CHOICES

    # Build the thinking template for each item
    # We want the model to think, so use _CHAT_TEMPLATE_THINK
    THINK_TEMPLATE = (
        "<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    harvested_items = []
    correct_count = 0

    for i, item in enumerate(items):
        prompt_text = item["prompt"]
        expected = item["answer"].lower()

        # Format with thinking template (no /no_think)
        formatted_prompt = THINK_TEMPLATE.format(prompt=prompt_text)

        # Generate with thinking enabled
        # Pass the already-formatted prompt so adapter doesn't re-wrap
        if is_mock:
            # Mock adapter: simulate a thinking response
            raw_answer = model.generate_short(prompt_text, max_new_tokens=max_tokens)
            # Wrap in fake thinking for mock
            full_response = f"<think>Let me think about this. {prompt_text[:50]}...</think>\n{raw_answer}"
        else:
            full_response = model.generate_short(
                formatted_prompt, max_new_tokens=max_tokens)

        thinking, answer_part = extract_thinking(full_response)
        extracted = extract_answer(full_response)
        correct = check_correct(extracted, expected)

        if correct:
            correct_count += 1

        harvested_items.append({
            "prompt": prompt_text,
            "expected": expected,
            "choices": choices,
            "full_response": full_response,
            "extracted_answer": extracted,
            "correct": correct,
            "thinking": thinking,
            "difficulty": item.get("difficulty", "hard"),
        })

    model_name = "mock"
    if hasattr(model, 'model_path') and model.model_path:
        model_name = Path(model.model_path).name

    return {
        "probe_name": probe_name,
        "model": model_name,
        "n_items": len(harvested_items),
        "n_correct": correct_count,
        "accuracy": correct_count / len(harvested_items) if harvested_items else 0.0,
        "items": harvested_items,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Harvest full generative responses from logprob probes")
    default_model = str(project_root / "models" / "Qwen3-30B-A3B-exl2")
    parser.add_argument("--model", type=str, default=default_model,
                        help="Model path")
    parser.add_argument("--probes", type=str, default=None,
                        help="Comma-separated probe names, or 'all'")
    parser.add_argument("--output-dir", type=str, default="results/harvested/")
    parser.add_argument("--mock", action="store_true",
                        help="Use MockAdapter for testing")
    parser.add_argument("--max-items", type=int, default=0,
                        help="Max items per probe (0 = all)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max tokens for generation (default: 200)")
    parser.add_argument("--mode", type=str, default="perfect",
                        help="MockAdapter mode (default: perfect)")
    parser.add_argument("--api", type=str, default=None,
                        help="Use API model instead of local (e.g. qwen-30b, claude-sonnet)")

    args = parser.parse_args()

    # Load model
    if args.mock:
        from sweep.mock_adapter import MockAdapter
        model = MockAdapter(mode=args.mode, seed=42)
        print(f"Using MockAdapter (mode={args.mode}, {model.num_layers} layers)")
        is_mock = True
    elif args.api:
        from scripts.run_baselines import MODEL_REGISTRY, FALLBACK_REGISTRY
        from sweep.api_adapters import get_adapter
        if args.api in MODEL_REGISTRY:
            provider, model_id = MODEL_REGISTRY[args.api]
        elif args.api in FALLBACK_REGISTRY:
            provider, model_id = FALLBACK_REGISTRY[args.api]
        else:
            # Treat as provider/model_id directly (e.g. openrouter/qwen/qwen3-32b)
            parts = args.api.split("/", 1)
            if len(parts) == 2:
                provider, model_id = parts[0], parts[1]
            else:
                parser.error(f"Unknown API model: {args.api}. "
                             f"Available: {list(MODEL_REGISTRY.keys())}")
                return
        model = get_adapter(provider, model_id)
        print(f"Using API: {provider}/{model_id}")
        is_mock = True  # API models don't need special template handling
    else:
        from sweep.exllama_adapter import ExLlamaV2LayerAdapter
        model = ExLlamaV2LayerAdapter(args.model)
        print(f"Loaded model: {model.num_layers} layers")
        is_mock = False

    # Resolve probe names
    from probes.registry import get_probe
    import probes  # trigger auto-discovery

    if args.probes is None or args.probes == "all":
        probe_names = get_logprob_probe_names()
    else:
        probe_names = [p.strip() for p in args.probes.split(",")]

    if not probe_names:
        print("No probes found.")
        return

    print(f"\nHarvesting {len(probe_names)} probes: {probe_names}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Harvest each probe
    t0 = time.time()
    for probe_name in probe_names:
        try:
            probe = get_probe(probe_name)
        except KeyError:
            print(f"  Warning: probe '{probe_name}' not found, skipping")
            continue

        if not hasattr(probe, 'ITEMS') or not hasattr(probe, 'CHOICES'):
            print(f"  Warning: probe '{probe_name}' has no ITEMS/CHOICES, skipping")
            continue

        if args.max_items > 0:
            probe.max_items = args.max_items

        print(f"\n--- Harvesting {probe_name} ({len(probe._limit(probe.ITEMS))} items) ---")
        tp = time.time()

        try:
            result = harvest_probe(model, probe, probe_name, args.max_tokens,
                                   is_mock)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - tp
        print(f"  Correct: {result['n_correct']}/{result['n_items']} "
              f"({result['accuracy']:.1%})")
        print(f"  Time: {elapsed:.1f}s")

        # Save
        output_file = output_dir / f"{probe_name}_responses.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {output_file}")

    total_time = time.time() - t0
    print(f"\nHarvest complete in {total_time:.1f}s")
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
