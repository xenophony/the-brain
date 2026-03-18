"""
Token convergence logprob probe — syntactic processing speed.

Incomplete sentences where the next word is highly predictable.
Each targets a specific function word category (articles, conjunctions,
prepositions, auxiliaries, modals, quantifiers).

Per-layer analysis reveals how early each word class converges,
directly mapping to inference savings: if articles converge by layer 10
out of 48, syntactic scaffolding may not need later layers.

Scoring normalizes within each item's category choices, not across all
50+ function words. This prevents cross-category dilution.

Maps to: Broca's area / syntactic processing circuits.
"""

from probes.registry import BaseLogprobProbe, register_probe

# Per-category choice sets — each item scores only against its category
CATEGORY_CHOICES = {
    "articles": ["the", "a", "an"],
    "conjunctions": ["and", "but", "or", "nor", "yet", "so", "for"],
    "prepositions": ["in", "on", "at", "to", "from", "with", "by", "of",
                     "into", "through", "between", "among", "under", "over"],
    "auxiliaries": ["is", "are", "was", "were", "be", "been", "being",
                    "has", "have", "had", "do", "does", "did"],
    "modals": ["will", "would", "could", "should", "can", "may",
               "might", "shall", "must"],
    "quantifiers": ["some", "any", "many", "much", "few", "several",
                    "most", "each", "both", "either"],
}

# Items — prompts where the next token is strongly predictable.
# Calibrated: the expected answer should be the clear #1 choice
# within its category.
ITEMS = [
    # --- Articles ---
    # Strong "the" contexts (definite reference back to known entity)
    {"prompt": "The cat jumped over", "answer": "the", "difficulty": "easy", "category": "articles"},
    {"prompt": "She walked into", "answer": "the", "difficulty": "easy", "category": "articles"},
    {"prompt": "He returned to", "answer": "the", "difficulty": "easy", "category": "articles"},
    {"prompt": "They left", "answer": "the", "difficulty": "easy", "category": "articles"},
    # Strong "a" contexts (introducing new entity)
    {"prompt": "She is", "answer": "a", "difficulty": "easy", "category": "articles"},
    {"prompt": "He bought", "answer": "a", "difficulty": "easy", "category": "articles"},
    # Harder
    {"prompt": "Inside", "answer": "the", "difficulty": "hard", "category": "articles"},
    {"prompt": "Once upon", "answer": "a", "difficulty": "hard", "category": "articles"},
    # Added
    {"prompt": "Open", "answer": "the", "difficulty": "easy", "category": "articles"},
    {"prompt": "I saw", "answer": "a", "difficulty": "easy", "category": "articles"},
    {"prompt": "Close", "answer": "the", "difficulty": "easy", "category": "articles"},
    {"prompt": "There is", "answer": "a", "difficulty": "easy", "category": "articles"},
    {"prompt": "Under", "answer": "the", "difficulty": "hard", "category": "articles"},
    {"prompt": "It was", "answer": "a", "difficulty": "hard", "category": "articles"},
    {"prompt": "Behind", "answer": "the", "difficulty": "hard", "category": "articles"},
    {"prompt": "What", "answer": "a", "difficulty": "hard", "category": "articles"},

    # --- Conjunctions ---
    {"prompt": "bread", "answer": "and", "difficulty": "easy", "category": "conjunctions"},
    {"prompt": "black", "answer": "and", "difficulty": "easy", "category": "conjunctions"},
    {"prompt": "salt", "answer": "and", "difficulty": "easy", "category": "conjunctions"},
    {"prompt": "pros", "answer": "and", "difficulty": "easy", "category": "conjunctions"},
    # Harder — "but" contexts
    {"prompt": "He tried his best", "answer": "but", "difficulty": "hard", "category": "conjunctions"},
    {"prompt": "I wanted to go", "answer": "but", "difficulty": "hard", "category": "conjunctions"},
    {"prompt": "It looked easy", "answer": "but", "difficulty": "hard", "category": "conjunctions"},
    {"prompt": "yes", "answer": "or", "difficulty": "hard", "category": "conjunctions"},
    # Added
    {"prompt": "peanut butter", "answer": "and", "difficulty": "easy", "category": "conjunctions"},
    {"prompt": "boys", "answer": "and", "difficulty": "easy", "category": "conjunctions"},
    {"prompt": "left", "answer": "and", "difficulty": "easy", "category": "conjunctions"},
    {"prompt": "rock", "answer": "and", "difficulty": "easy", "category": "conjunctions"},
    {"prompt": "She was tired", "answer": "but", "difficulty": "hard", "category": "conjunctions"},
    {"prompt": "The plan seemed perfect", "answer": "but", "difficulty": "hard", "category": "conjunctions"},
    {"prompt": "now", "answer": "or", "difficulty": "hard", "category": "conjunctions"},
    {"prompt": "He ran fast", "answer": "but", "difficulty": "hard", "category": "conjunctions"},

    # --- Prepositions ---
    {"prompt": "The cat sat", "answer": "on", "difficulty": "easy", "category": "prepositions"},
    {"prompt": "She lives", "answer": "in", "difficulty": "easy", "category": "prepositions"},
    {"prompt": "Welcome", "answer": "to", "difficulty": "easy", "category": "prepositions"},
    {"prompt": "Made", "answer": "of", "difficulty": "easy", "category": "prepositions"},
    # Harder
    {"prompt": "The letter was written", "answer": "by", "difficulty": "hard", "category": "prepositions"},
    {"prompt": "She walked", "answer": "through", "difficulty": "hard", "category": "prepositions"},
    {"prompt": "Shared", "answer": "between", "difficulty": "hard", "category": "prepositions"},
    {"prompt": "He hid", "answer": "under", "difficulty": "hard", "category": "prepositions"},
    # Added
    {"prompt": "Born", "answer": "in", "difficulty": "easy", "category": "prepositions"},
    {"prompt": "Point", "answer": "of", "difficulty": "easy", "category": "prepositions"},
    {"prompt": "Listen", "answer": "to", "difficulty": "easy", "category": "prepositions"},
    {"prompt": "Put it", "answer": "on", "difficulty": "easy", "category": "prepositions"},
    {"prompt": "Sent", "answer": "by", "difficulty": "hard", "category": "prepositions"},
    {"prompt": "Jumped", "answer": "over", "difficulty": "hard", "category": "prepositions"},
    {"prompt": "Fell", "answer": "into", "difficulty": "hard", "category": "prepositions"},
    {"prompt": "Chosen", "answer": "from", "difficulty": "hard", "category": "prepositions"},

    # --- Auxiliaries ---
    {"prompt": "She", "answer": "is", "difficulty": "easy", "category": "auxiliaries"},
    {"prompt": "They", "answer": "are", "difficulty": "easy", "category": "auxiliaries"},
    {"prompt": "He", "answer": "was", "difficulty": "easy", "category": "auxiliaries"},
    {"prompt": "It", "answer": "is", "difficulty": "easy", "category": "auxiliaries"},
    # Harder
    {"prompt": "The results", "answer": "were", "difficulty": "hard", "category": "auxiliaries"},
    {"prompt": "All the evidence", "answer": "has", "difficulty": "hard", "category": "auxiliaries"},
    {"prompt": "Nothing", "answer": "has", "difficulty": "hard", "category": "auxiliaries"},
    {"prompt": "The work", "answer": "has", "difficulty": "hard", "category": "auxiliaries"},
    # Added
    {"prompt": "We", "answer": "are", "difficulty": "easy", "category": "auxiliaries"},
    {"prompt": "I", "answer": "was", "difficulty": "easy", "category": "auxiliaries"},
    {"prompt": "You", "answer": "are", "difficulty": "easy", "category": "auxiliaries"},
    {"prompt": "The dog", "answer": "is", "difficulty": "easy", "category": "auxiliaries"},
    {"prompt": "The children", "answer": "were", "difficulty": "hard", "category": "auxiliaries"},
    {"prompt": "Everyone", "answer": "has", "difficulty": "hard", "category": "auxiliaries"},
    {"prompt": "The doors", "answer": "were", "difficulty": "hard", "category": "auxiliaries"},
    {"prompt": "My patience", "answer": "has", "difficulty": "hard", "category": "auxiliaries"},

    # --- Modals ---
    {"prompt": "You", "answer": "can", "difficulty": "easy", "category": "modals"},
    {"prompt": "I think we", "answer": "should", "difficulty": "easy", "category": "modals"},
    {"prompt": "It", "answer": "would", "difficulty": "easy", "category": "modals"},
    {"prompt": "We", "answer": "can", "difficulty": "easy", "category": "modals"},
    # Harder
    {"prompt": "All employees", "answer": "must", "difficulty": "hard", "category": "modals"},
    {"prompt": "The suspect", "answer": "may", "difficulty": "hard", "category": "modals"},
    {"prompt": "Under no circumstances", "answer": "shall", "difficulty": "hard", "category": "modals"},
    {"prompt": "Given the evidence, it", "answer": "could", "difficulty": "hard", "category": "modals"},
    # Added
    {"prompt": "I", "answer": "can", "difficulty": "easy", "category": "modals"},
    {"prompt": "She", "answer": "would", "difficulty": "easy", "category": "modals"},
    {"prompt": "They", "answer": "will", "difficulty": "easy", "category": "modals"},
    {"prompt": "He", "answer": "could", "difficulty": "easy", "category": "modals"},
    {"prompt": "Visitors", "answer": "must", "difficulty": "hard", "category": "modals"},
    {"prompt": "The defendant", "answer": "may", "difficulty": "hard", "category": "modals"},
    {"prompt": "Students", "answer": "should", "difficulty": "hard", "category": "modals"},
    {"prompt": "One", "answer": "must", "difficulty": "hard", "category": "modals"},

    # --- Quantifiers ---
    {"prompt": "There are", "answer": "some", "difficulty": "easy", "category": "quantifiers"},
    {"prompt": "Are there", "answer": "any", "difficulty": "easy", "category": "quantifiers"},
    {"prompt": "Very", "answer": "few", "difficulty": "easy", "category": "quantifiers"},
    {"prompt": "How", "answer": "many", "difficulty": "easy", "category": "quantifiers"},
    # Harder
    {"prompt": "Only a", "answer": "few", "difficulty": "hard", "category": "quantifiers"},
    {"prompt": "On", "answer": "both", "difficulty": "hard", "category": "quantifiers"},
    {"prompt": "Without", "answer": "any", "difficulty": "hard", "category": "quantifiers"},
    {"prompt": "Not", "answer": "many", "difficulty": "hard", "category": "quantifiers"},
    # Added
    {"prompt": "I have", "answer": "some", "difficulty": "easy", "category": "quantifiers"},
    {"prompt": "Too", "answer": "many", "difficulty": "easy", "category": "quantifiers"},
    {"prompt": "So", "answer": "many", "difficulty": "easy", "category": "quantifiers"},
    {"prompt": "Is there", "answer": "any", "difficulty": "easy", "category": "quantifiers"},
    {"prompt": "Quite a", "answer": "few", "difficulty": "hard", "category": "quantifiers"},
    {"prompt": "For", "answer": "each", "difficulty": "hard", "category": "quantifiers"},
    {"prompt": "In", "answer": "some", "difficulty": "hard", "category": "quantifiers"},
    {"prompt": "Hardly", "answer": "any", "difficulty": "hard", "category": "quantifiers"},
]

# All choices merged — needed for cross-probe batching in the sweep runner.
# But scoring normalizes within each item's category.
CHOICES = sorted(set(
    tok for choices in CATEGORY_CHOICES.values() for tok in choices
))


@register_probe
class ConvergenceLogprobProbe(BaseLogprobProbe):
    name = "convergence_logprob"
    description = "Function word convergence speed — syntactic processing circuits"
    ITEMS = ITEMS
    CHOICES = CHOICES
    max_items = None  # always run all items — need all categories

    def score_logprobs(self, items, all_logprobs):
        """Score within each item's category, not across all 50+ choices."""
        import math
        from collections import defaultdict

        easy_argmax = []
        hard_argmax = []
        easy_pcorrect = []
        hard_pcorrect = []
        cat_argmax = defaultdict(list)
        cat_pcorrect = defaultdict(list)
        item_results = []

        for item, logprobs in zip(items, all_logprobs):
            expected = item["answer"].lower()
            category = item.get("category", "unknown")
            difficulty = item.get("difficulty", "hard")

            # Score against category-specific choices only
            cat_choices = CATEGORY_CHOICES.get(category, self.CHOICES)

            probs = {}
            for choice in cat_choices:
                lp = logprobs.get(choice, float('-inf'))
                probs[choice] = math.exp(lp) if lp > -100 else 0.0
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}

            best = max(probs, key=probs.get) if probs else ""
            argmax_correct = 1.0 if best == expected else 0.0
            p_correct = probs.get(expected, 0.0)

            if difficulty == "easy":
                easy_argmax.append(argmax_correct)
                easy_pcorrect.append(p_correct)
            else:
                hard_argmax.append(argmax_correct)
                hard_pcorrect.append(p_correct)

            cat_argmax[category].append(argmax_correct)
            cat_pcorrect[category].append(p_correct)

            if self.log_responses:
                item_results.append({
                    "difficulty": difficulty,
                    "category": category,
                    "prompt": item["prompt"][:100],
                    "expected": expected,
                    "argmax": best,
                    "argmax_correct": argmax_correct,
                    "p_correct": round(p_correct, 4),
                })

        all_argmax = easy_argmax + hard_argmax
        all_pcorrect = easy_pcorrect + hard_pcorrect

        result = {
            "score": sum(all_argmax) / len(all_argmax) if all_argmax else 0.0,
            "easy_score": sum(easy_argmax) / len(easy_argmax) if easy_argmax else 0.0,
            "hard_score": sum(hard_argmax) / len(hard_argmax) if hard_argmax else 0.0,
            "p_correct": sum(all_pcorrect) / len(all_pcorrect) if all_pcorrect else 0.0,
            "p_correct_easy": sum(easy_pcorrect) / len(easy_pcorrect) if easy_pcorrect else 0.0,
            "p_correct_hard": sum(hard_pcorrect) / len(hard_pcorrect) if hard_pcorrect else 0.0,
            "n_easy": len(easy_argmax),
            "n_hard": len(hard_argmax),
        }

        for cat in sorted(cat_argmax.keys()):
            vals = cat_argmax[cat]
            pvals = cat_pcorrect[cat]
            result[f"cat_{cat}_score"] = sum(vals) / len(vals) if vals else 0.0
            result[f"cat_{cat}_pcorrect"] = sum(pvals) / len(pvals) if pvals else 0.0

        if item_results:
            result["item_results"] = item_results

        return result
