"""
Hierarchical router — classifies prompts and selects optimal circuit paths.

Architecture:
  L1: Domain classifier (REASONING, SOCIAL, LANGUAGE, SPATIAL, MEMORY, EXECUTION)
  L2: Per-domain subtype classifier (e.g. REASONING -> math vs planning)

Classification cascade:
  1. L1 determines broad domain
  2. L2 determines specific probe/circuit
  3. Circuit config is looked up from sweep results
  4. Layer path is returned for the model adapter

Without trained classifiers, falls back to keyword heuristic matching.
"""

import re
from collections import Counter

from analysis.taxonomy import ALL_DOMAINS, get_probes_in_domain


# Default keyword sets for heuristic classification
_DOMAIN_KEYWORDS = {
    "REASONING": [
        "calculate", "compute", "solve", "math", "equation", "proof",
        "logic", "deduce", "if.*then", "therefore", "plan", "steps",
        "sequence", "order", "cause", "effect", "why", "reason",
    ],
    "SOCIAL": [
        "feel", "emotion", "empathy", "perspective", "social",
        "relationship", "tone", "sentiment", "mood", "analogy",
        "metaphor", "meaning", "interpret",
    ],
    "LANGUAGE": [
        "grammar", "syntax", "word", "sentence", "translate",
        "rephrase", "summarize", "abstract", "definition",
        "instruction", "follow", "constraint", "format",
    ],
    "SPATIAL": [
        "grid", "board", "position", "coordinate", "map",
        "direction", "north", "south", "east", "west",
        "left", "right", "above", "below", "adjacent",
    ],
    "MEMORY": [
        "fact", "who", "when", "where", "capital", "president",
        "history", "date", "name", "recall", "remember",
        "know", "knowledge", "confident", "certain",
    ],
    "EXECUTION": [
        "code", "program", "function", "implement", "debug",
        "test", "compile", "run", "execute", "tool",
        "api", "call", "search", "lookup", "query",
    ],
}


class HierarchicalRouter:
    """Two-level domain classifier with circuit path lookup."""

    def __init__(self, n_layers: int, taxonomy=None):
        """
        Args:
            n_layers: Number of layers in the target model.
            taxonomy: Optional custom taxonomy dict. Defaults to ALL_DOMAINS.
        """
        self.n_layers = n_layers
        self.taxonomy = taxonomy or ALL_DOMAINS
        self.l1_classifier = None  # trained later
        self.l2_classifiers = {}   # per-domain, trained later
        self.circuit_configs = {}  # from sweep results: {probe_name: (i, j)}

    def classify(self, prompt: str, activation_vector=None) -> dict:
        """Returns domain classification and recommended path.

        Without trained classifiers, uses keyword heuristic.

        Args:
            prompt: Input text to classify.
            activation_vector: Optional hidden state vector for trained classifier.

        Returns:
            Dict with keys:
              - domain: str (top domain)
              - domain_scores: dict[str, float] (all domain scores)
              - probe: str (recommended probe/circuit)
              - path: list[int] (recommended layer execution path)
              - method: str ("trained" or "heuristic")
        """
        if self.l1_classifier is not None and activation_vector is not None:
            return self._classify_trained(prompt, activation_vector)
        return self._classify_heuristic(prompt)

    def _classify_heuristic(self, prompt: str) -> dict:
        """Keyword-based domain classification fallback."""
        prompt_lower = prompt.lower()
        scores = {}

        for domain, keywords in _DOMAIN_KEYWORDS.items():
            score = 0.0
            for kw in keywords:
                if re.search(r'\b' + kw + r'\b', prompt_lower):
                    score += 1.0
            # Normalize by number of keywords
            scores[domain] = score / max(len(keywords), 1)

        # Default to REASONING if no keywords match
        top_domain = max(scores, key=scores.get) if any(v > 0 for v in scores.values()) else "REASONING"

        # Pick first probe in the domain that has a circuit config
        probes = get_probes_in_domain(top_domain)
        recommended_probe = probes[0] if probes else "math"
        for p in probes:
            if p in self.circuit_configs:
                recommended_probe = p
                break

        # Build path from circuit config if available
        path = list(range(self.n_layers))
        if recommended_probe in self.circuit_configs:
            config = self.circuit_configs[recommended_probe]
            # Simple duplication of the best circuit
            i, j = config
            path = list(range(j)) + list(range(i, self.n_layers))

        return {
            "domain": top_domain,
            "domain_scores": scores,
            "probe": recommended_probe,
            "path": path,
            "method": "heuristic",
        }

    def _classify_trained(self, prompt: str, activation_vector) -> dict:
        """Classification using trained L1/L2 classifiers."""
        # L1 prediction
        domain = self.l1_classifier.predict([activation_vector])[0]
        domain_probs = {}
        if hasattr(self.l1_classifier, "predict_proba"):
            probs = self.l1_classifier.predict_proba([activation_vector])[0]
            domain_probs = dict(zip(self.l1_classifier.classes_, probs.tolist()))

        # L2 prediction if available
        probes = get_probes_in_domain(domain)
        recommended_probe = probes[0] if probes else "math"
        if domain in self.l2_classifiers:
            recommended_probe = self.l2_classifiers[domain].predict([activation_vector])[0]

        path = list(range(self.n_layers))
        if recommended_probe in self.circuit_configs:
            i, j = self.circuit_configs[recommended_probe]
            path = list(range(j)) + list(range(i, self.n_layers))

        return {
            "domain": domain,
            "domain_scores": domain_probs,
            "probe": recommended_probe,
            "path": path,
            "method": "trained",
        }

    def train_l1(self, labeled_prompts: list, labels: list):
        """Train L1 domain classifier.

        Uses sklearn LogisticRegression on TF-IDF features as baseline.

        Args:
            labeled_prompts: List of prompt strings.
            labels: List of domain labels (e.g. "REASONING", "SOCIAL").
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(labeled_prompts)
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
        clf.fit(X, labels)

        self.l1_classifier = clf
        self._l1_vectorizer = vectorizer

    def train_l2(self, domain: str, labeled_prompts: list, labels: list):
        """Train per-domain L2 subtype classifier.

        Args:
            domain: Domain name (e.g. "REASONING").
            labeled_prompts: List of prompt strings for this domain.
            labels: List of probe/subtype labels (e.g. "math", "planning").
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        vectorizer = TfidfVectorizer(max_features=3000)
        X = vectorizer.fit_transform(labeled_prompts)
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
        clf.fit(X, labels)

        self.l2_classifiers[domain] = clf
        if not hasattr(self, "_l2_vectorizers"):
            self._l2_vectorizers = {}
        self._l2_vectorizers[domain] = vectorizer


class FuzzyDomainMatcher:
    """Lightweight domain matcher using keyword overlap / TF-IDF similarity."""

    def __init__(self, probe_questions: dict[str, list[str]]):
        """Build domain prototypes from probe question sets.

        Args:
            probe_questions: {probe_name: [list of example questions]}.
                Used to build keyword profiles for each probe domain.
        """
        self.probe_questions = probe_questions
        self._probe_keywords = {}

        for probe_name, questions in probe_questions.items():
            # Build word frequency profile
            word_counts = Counter()
            for q in questions:
                words = re.findall(r'\b[a-z]+\b', q.lower())
                word_counts.update(words)
            # Keep top 50 keywords, excluding stop words
            stop_words = {
                "the", "a", "an", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "shall", "can",
                "to", "of", "in", "for", "on", "with", "at", "by", "from",
                "it", "this", "that", "these", "those", "and", "or", "but",
                "not", "no", "if", "then", "than", "so", "as", "what",
                "which", "who", "whom", "how", "when", "where", "why",
            }
            filtered = {w: c for w, c in word_counts.items() if w not in stop_words and len(w) > 2}
            self._probe_keywords[probe_name] = dict(
                sorted(filtered.items(), key=lambda x: -x[1])[:50]
            )

    def match(self, prompt: str) -> dict[str, float]:
        """Return weighted domain mixture via keyword similarity.

        Args:
            prompt: Input text to match against domain prototypes.

        Returns:
            {probe_name: similarity_score} normalized to sum to 1.0.
            Returns uniform distribution if no keywords match.
        """
        prompt_words = set(re.findall(r'\b[a-z]+\b', prompt.lower()))
        scores = {}

        for probe_name, keywords in self._probe_keywords.items():
            overlap = sum(
                keywords[w] for w in prompt_words if w in keywords
            )
            total = sum(keywords.values()) if keywords else 1
            scores[probe_name] = overlap / total

        # Normalize
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        else:
            # Uniform fallback
            n = len(self._probe_keywords)
            if n > 0:
                scores = {k: 1.0 / n for k in self._probe_keywords}

        return scores
