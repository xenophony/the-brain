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
}


def build_psych_token_map(tokenizer) -> dict:
    """Build mapping from psych categories to token IDs.

    Tries both bare word and space-prefixed variant for each word.
    Returns {"categories": {name: [token_ids]}, "multi_token_words": set}.
    """
    categories = {}
    multi_token_words = set()

    for cat_name, words in PSYCH_VOCAB.items():
        token_ids = set()
        for word in words:
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

                    if len(ids) == 1:
                        token_ids.add(ids[0])
                    elif len(ids) > 1:
                        multi_token_words.add(word)
                        token_ids.add(ids[0])  # first token still carries signal
                except Exception:
                    continue

        categories[cat_name] = sorted(token_ids)

    return {"categories": categories, "multi_token_words": multi_token_words}


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

    Args:
        logits_tensor: (vocab_size,) float tensor of logits
        token_map: output of build_psych_token_map

    Returns:
        dict mapping category_name -> probability mass (float, 0-1)
    """
    import torch

    probs = torch.softmax(logits_tensor.float(), dim=-1)
    categories = token_map["categories"]

    scores = {}
    for cat_name, token_ids in categories.items():
        if not token_ids:
            scores[cat_name] = 0.0
            continue
        # Gather probabilities for this category's tokens
        ids_tensor = torch.tensor(token_ids, device=probs.device, dtype=torch.long)
        # Filter out any IDs beyond vocab size
        valid = ids_tensor[ids_tensor < probs.shape[0]]
        if valid.numel() == 0:
            scores[cat_name] = 0.0
        else:
            scores[cat_name] = probs[valid].sum().item()

    return scores
