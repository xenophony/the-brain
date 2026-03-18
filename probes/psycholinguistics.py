"""
Psycholinguistic vocabulary scoring for logprob probes.

Measures probability mass across psychological signal categories at the
answer decision point. LLMs are trained on human data containing hedging,
confidence, uncertainty signals. RLHF suppresses these in generation but
the signals persist in the logit distribution.

At the moment the model "decides" its answer, the relative probability
of hedging vs confidence vocab reflects internal certainty — regardless
of what the model is trained to output.

Tokenization notes:
  - Qwen3 uses SentencePiece-style tokenization with leading space as
    part of the token (▁maybe vs maybe)
  - Words at sentence start vs mid-sentence tokenize differently
  - Our capture is at the answer decision point, so mid-sentence
    tokenization (with leading space) is more representative
  - Multi-token words are noted but not excluded — their first token
    still carries signal
  - The vocab list should be treated as approximate signal, not exact
    measurement
"""

import math

# Psychological signal categories → word lists
PSYCH_VOCAB = {
    "hedging": [
        "perhaps", "maybe", "might", "possibly", "unclear", "uncertain",
        "seems", "appears", "suggest", "indicate",
    ],
    "confidence": [
        "definitely", "certainly", "clearly", "obviously", "undoubtedly",
        "always", "never", "absolutely",
    ],
    "epistemic_uncertain": [
        "guess", "assume", "suppose", "believe", "think", "feel", "suspect",
    ],
    "epistemic_certain": [
        "know", "sure", "certain", "fact", "proven", "confirmed",
    ],
    "causation": [
        "because", "therefore", "thus", "hence", "since", "consequently",
    ],
    "approximators": [
        "about", "around", "roughly", "approximately", "nearly", "almost",
    ],
    "negation": [
        "not", "never", "no", "neither", "nor", "without", "except",
    ],
    "absolutes": [
        "always", "never", "every", "all", "none", "impossible", "guaranteed",
    ],
    "first_person": [
        "I", "my", "me", "myself", "mine",
    ],
    "distancing": [
        "one", "they", "people", "someone", "it", "this",
    ],
    "urgency": [
        "critical", "urgent", "important", "emergency", "immediately",
        "crucial", "essential", "vital", "priority", "now",
    ],
    "stress": [
        "dangerous", "risk", "threat", "warning", "careful",
        "caution", "harm", "fatal", "serious", "severe",
    ],
    "multilingual_yes": [
        "yes", "oui", "sí", "ja", "да", "sim", "はい", "evet",
        "tak", "igen", "kyllä",
    ],
    "multilingual_no": [
        "no", "non", "nein", "не", "нет", "não", "いいえ",
        "hayır", "nie", "nem", "ei",
    ],
    # --- Function words: structural/syntactic processing signals ---
    # These track grammatical scaffolding rather than semantic content.
    # In the brain, function words activate Broca's area (syntax) while
    # content words activate Wernicke's area (semantics). Tracking these
    # per-layer reveals where the model builds syntactic structure.
    "articles": [
        "the", "a", "an",
    ],
    "conjunctions": [
        "and", "but", "or", "nor", "yet", "so", "for",
    ],
    "prepositions": [
        "in", "on", "at", "to", "from", "with", "by", "of",
        "into", "through", "between", "among", "under", "over",
    ],
    "auxiliaries": [
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did",
    ],
    "modals": [
        "will", "would", "could", "should", "can", "may",
        "might", "shall", "must",
    ],
    "quantifiers": [
        "some", "any", "many", "much", "few", "several",
        "most", "each", "both", "either",
    ],
}


def build_psych_token_map(tokenizer) -> dict:
    """Build mapping from psych categories to token IDs.

    Tries both bare word and space-prefixed variant for each word.
    Returns:
        {
            "categories": {name: [first_token_ids]},  # backward compat
            "token_sequences": {name: {word: [all_token_ids]}},
            "multi_token_words": set,
        }

    For multi-token words, token_sequences stores the full subword token
    list so that score_psych_vocab_from_logits can compute a geometric mean
    across all subword probabilities instead of using only the first token.
    """
    categories = {}
    token_sequences = {}
    multi_token_words = set()

    for cat_name, words in PSYCH_VOCAB.items():
        token_ids = set()
        cat_sequences = {}
        for word in words:
            best_ids = None
            for variant in [word, f" {word}", word.lower(), f" {word.lower()}"]:
                try:
                    encoded = tokenizer.encode(variant, add_bos=False)
                    if hasattr(encoded, 'shape'):
                        # Tensor
                        if encoded.dim() == 2:
                            encoded = encoded[0]
                        ids = encoded.tolist()
                    elif isinstance(encoded, list):
                        ids = encoded
                    else:
                        continue

                    if ids:
                        if best_ids is None or len(ids) < len(best_ids):
                            best_ids = ids  # prefer shortest tokenization
                except Exception:
                    continue

            if best_ids:
                token_ids.add(best_ids[0])  # first token for backward compat
                if len(best_ids) > 1:
                    multi_token_words.add(word)
                cat_sequences[word] = best_ids

        categories[cat_name] = sorted(token_ids)
        token_sequences[cat_name] = cat_sequences

    return {
        "categories": categories,
        "token_sequences": token_sequences,
        "multi_token_words": multi_token_words,
    }


def score_psych_vocab(log_probs_dict: dict, token_map: dict) -> dict:
    """Score psycholinguistic categories from logprob results.

    Takes the log_probs dict (token_str -> log_prob) from get_logprobs
    and the token_map from build_psych_token_map. Since we only have
    logprobs for target tokens (not full vocab), this version works
    with whatever tokens are available.

    For full-vocab scoring, use score_psych_vocab_from_logits instead.
    """
    # This is a placeholder — the real scoring uses logits directly
    return {}


def score_psych_vocab_from_logits(logits_tensor, token_map: dict) -> dict:
    """Score psycholinguistic categories from raw logit tensor.

    Takes the full vocabulary logit tensor and computes probability mass
    per category. Fast — single softmax + index gather.

    For multi-token words, uses geometric mean of subword probabilities
    (= exp(mean(log_probs))) instead of only the first token probability.
    This gives the "independent marginal" probability — not the true
    conditional p(t2|t1), but the best approximation from a single forward
    pass's full vocab logits.

    Args:
        logits_tensor: (vocab_size,) float tensor of logits
        token_map: output of build_psych_token_map

    Returns:
        dict mapping category_name -> probability mass (float, 0-1)
    """
    import torch

    probs = torch.softmax(logits_tensor.float(), dim=-1)
    log_probs = torch.log_softmax(logits_tensor.float(), dim=-1)
    token_sequences = token_map.get("token_sequences", {})

    scores = {}

    if token_sequences:
        # New path: use full token sequences for proper multi-token scoring
        for cat_name, sequences in token_sequences.items():
            cat_mass = 0.0
            for word, ids in sequences.items():
                valid_ids = [i for i in ids if i < probs.shape[0]]
                if not valid_ids:
                    continue
                if len(valid_ids) == 1:
                    # Single-token word: use probability directly
                    cat_mass += probs[valid_ids[0]].item()
                else:
                    # Multi-token word: geometric mean of subword probs
                    mean_lp = sum(log_probs[i].item() for i in valid_ids) / len(valid_ids)
                    cat_mass += math.exp(mean_lp)
            scores[cat_name] = cat_mass
    else:
        # Fallback for old-format token_maps without token_sequences
        categories = token_map.get("categories", {})
        for cat_name, token_ids in categories.items():
            if not token_ids:
                scores[cat_name] = 0.0
                continue
            ids_tensor = torch.tensor(token_ids, device=probs.device, dtype=torch.long)
            valid = ids_tensor[ids_tensor < probs.shape[0]]
            if valid.numel() == 0:
                scores[cat_name] = 0.0
            else:
                scores[cat_name] = probs[valid].sum().item()

    return scores
