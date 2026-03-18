"""
Token convergence logprob probe — syntactic processing speed.

Incomplete sentences where the next word is highly predictable.
Each targets a specific function word category (articles, conjunctions,
prepositions, auxiliaries, modals, quantifiers).

Per-layer analysis reveals how early each word class converges,
directly mapping to inference savings: if articles converge by layer 10
out of 48, syntactic scaffolding may not need later layers.

Maps to: Broca's area / syntactic processing circuits.
"""

from probes.registry import BaseLogprobProbe, register_probe

# Each item's answer is the highly-predictable next function word.
# Grouped by function word category for analysis.
ITEMS = [
    # --- Articles ---
    # "the" is overwhelmingly likely after these fragments
    {"prompt": "She opened the door to", "answer": "the", "difficulty": "easy", "category": "articles"},
    {"prompt": "He walked across", "answer": "the", "difficulty": "easy", "category": "articles"},
    {"prompt": "They arrived at", "answer": "the", "difficulty": "easy", "category": "articles"},
    {"prompt": "The winner of", "answer": "the", "difficulty": "easy", "category": "articles"},
    # "a" expected
    {"prompt": "She picked up", "answer": "a", "difficulty": "easy", "category": "articles"},
    {"prompt": "It was", "answer": "a", "difficulty": "easy", "category": "articles"},
    # Harder — context-dependent article choice
    {"prompt": "He saw", "answer": "a", "difficulty": "hard", "category": "articles"},
    {"prompt": "There was", "answer": "a", "difficulty": "hard", "category": "articles"},

    # --- Conjunctions ---
    {"prompt": "She bought apples", "answer": "and", "difficulty": "easy", "category": "conjunctions"},
    {"prompt": "He tried hard", "answer": "but", "difficulty": "easy", "category": "conjunctions"},
    {"prompt": "You can have tea", "answer": "or", "difficulty": "easy", "category": "conjunctions"},
    {"prompt": "The food was good", "answer": "and", "difficulty": "easy", "category": "conjunctions"},
    # Harder
    {"prompt": "He studied all night", "answer": "but", "difficulty": "hard", "category": "conjunctions"},
    {"prompt": "The plan was risky", "answer": "but", "difficulty": "hard", "category": "conjunctions"},
    {"prompt": "She neither agreed", "answer": "nor", "difficulty": "hard", "category": "conjunctions"},
    {"prompt": "He was tired", "answer": "yet", "difficulty": "hard", "category": "conjunctions"},

    # --- Prepositions ---
    {"prompt": "The cat sat", "answer": "on", "difficulty": "easy", "category": "prepositions"},
    {"prompt": "She lives", "answer": "in", "difficulty": "easy", "category": "prepositions"},
    {"prompt": "He went", "answer": "to", "difficulty": "easy", "category": "prepositions"},
    {"prompt": "The book is", "answer": "on", "difficulty": "easy", "category": "prepositions"},
    # Harder
    {"prompt": "The agreement was reached", "answer": "between", "difficulty": "hard", "category": "prepositions"},
    {"prompt": "She walked", "answer": "through", "difficulty": "hard", "category": "prepositions"},
    {"prompt": "The letter was written", "answer": "by", "difficulty": "hard", "category": "prepositions"},
    {"prompt": "He hid", "answer": "under", "difficulty": "hard", "category": "prepositions"},

    # --- Auxiliaries ---
    {"prompt": "She", "answer": "is", "difficulty": "easy", "category": "auxiliaries"},
    {"prompt": "They", "answer": "are", "difficulty": "easy", "category": "auxiliaries"},
    {"prompt": "He", "answer": "was", "difficulty": "easy", "category": "auxiliaries"},
    {"prompt": "The work", "answer": "has", "difficulty": "easy", "category": "auxiliaries"},
    # Harder
    {"prompt": "The project", "answer": "was", "difficulty": "hard", "category": "auxiliaries"},
    {"prompt": "All the evidence", "answer": "has", "difficulty": "hard", "category": "auxiliaries"},
    {"prompt": "The results", "answer": "were", "difficulty": "hard", "category": "auxiliaries"},
    {"prompt": "Nothing", "answer": "has", "difficulty": "hard", "category": "auxiliaries"},

    # --- Modals ---
    {"prompt": "You", "answer": "can", "difficulty": "easy", "category": "modals"},
    {"prompt": "I think we", "answer": "should", "difficulty": "easy", "category": "modals"},
    {"prompt": "This", "answer": "could", "difficulty": "easy", "category": "modals"},
    {"prompt": "That", "answer": "would", "difficulty": "easy", "category": "modals"},
    # Harder
    {"prompt": "All employees", "answer": "must", "difficulty": "hard", "category": "modals"},
    {"prompt": "The suspect", "answer": "may", "difficulty": "hard", "category": "modals"},
    {"prompt": "Under no circumstances", "answer": "shall", "difficulty": "hard", "category": "modals"},
    {"prompt": "Given the evidence, it", "answer": "could", "difficulty": "hard", "category": "modals"},

    # --- Quantifiers ---
    {"prompt": "There are", "answer": "some", "difficulty": "easy", "category": "quantifiers"},
    {"prompt": "Not", "answer": "many", "difficulty": "easy", "category": "quantifiers"},
    {"prompt": "Are there", "answer": "any", "difficulty": "easy", "category": "quantifiers"},
    {"prompt": "Very", "answer": "few", "difficulty": "easy", "category": "quantifiers"},
    # Harder
    {"prompt": "Only a", "answer": "few", "difficulty": "hard", "category": "quantifiers"},
    {"prompt": "On", "answer": "both", "difficulty": "hard", "category": "quantifiers"},
    {"prompt": "Nearly all of the", "answer": "most", "difficulty": "hard", "category": "quantifiers"},
    {"prompt": "Without", "answer": "any", "difficulty": "hard", "category": "quantifiers"},
]

# All possible target tokens across all categories
CHOICES = [
    # Articles
    "the", "a", "an",
    # Conjunctions
    "and", "but", "or", "nor", "yet", "so", "for",
    # Prepositions
    "in", "on", "at", "to", "from", "with", "by", "of",
    "into", "through", "between", "among", "under", "over",
    # Auxiliaries
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    # Modals
    "will", "would", "could", "should", "can", "may",
    "might", "shall", "must",
    # Quantifiers
    "some", "any", "many", "much", "few", "several",
    "most", "each", "both", "either",
]


@register_probe
class ConvergenceLogprobProbe(BaseLogprobProbe):
    name = "convergence_logprob"
    description = "Function word convergence speed — syntactic processing circuits"
    ITEMS = ITEMS
    CHOICES = CHOICES

    def score_logprobs(self, items, all_logprobs):
        """Score with per-category breakdown in addition to standard scoring."""
        import math

        # Standard scoring via parent
        result = super().score_logprobs(items, all_logprobs)

        # Per-category breakdown
        from collections import defaultdict
        cat_argmax = defaultdict(list)
        cat_pcorrect = defaultdict(list)

        for item, logprobs in zip(items, all_logprobs):
            expected = item["answer"].lower()
            category = item.get("category", "unknown")

            probs = {}
            for choice in self.CHOICES:
                lp = logprobs.get(choice, float('-inf'))
                probs[choice] = math.exp(lp) if lp > -100 else 0.0
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}

            best = max(probs, key=probs.get) if probs else ""
            cat_argmax[category].append(1.0 if best == expected else 0.0)
            cat_pcorrect[category].append(probs.get(expected, 0.0))

        for cat in sorted(cat_argmax.keys()):
            vals = cat_argmax[cat]
            pvals = cat_pcorrect[cat]
            result[f"cat_{cat}_score"] = sum(vals) / len(vals) if vals else 0.0
            result[f"cat_{cat}_pcorrect"] = sum(pvals) / len(pvals) if pvals else 0.0

        return result
