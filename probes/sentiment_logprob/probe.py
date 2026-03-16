"""
Sentiment analysis logprob probe — social/emotional tone circuits.

Measures P("positive") vs P("negative") via logprobs.
Generation-based scoring failed (model returns emotion words, not labels).
Logprob scoring bypasses this — directly measures the model's internal
classification regardless of what it would generate.

Maps to: limbic / social cognition circuits.
"""

from probes.registry import BaseLogprobProbe, register_probe

ITEMS = [
    # Easy (obvious sentiment)
    {"prompt": "Is this positive or negative: 'I love this beautiful sunny day!'", "answer": "positive", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'This is the worst experience of my life.'", "answer": "negative", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'What a wonderful surprise!'", "answer": "positive", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'I'm so grateful for your help.'", "answer": "positive", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'This product is terrible and broke immediately.'", "answer": "negative", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'I'm devastated by the news.'", "answer": "negative", "difficulty": "easy"},
    # Hard (sarcasm, double negatives, mixed signals)
    {"prompt": "Is this positive or negative: 'Oh great, another meeting that could have been an email.'", "answer": "negative", "difficulty": "hard"},
    {"prompt": "Is this positive or negative: 'Well, that went exactly as planned.' (said after a disaster)", "answer": "negative", "difficulty": "hard"},
    {"prompt": "Is this positive or negative: 'I'm not unhappy with the results.'", "answer": "positive", "difficulty": "hard"},
    {"prompt": "Is this positive or negative: 'Sure, because what I really needed was more work.'", "answer": "negative", "difficulty": "hard"},
    {"prompt": "Is this positive or negative: 'The surgery was successful but recovery will be long.'", "answer": "positive", "difficulty": "hard"},
    {"prompt": "Is this positive or negative: 'At least it can't get any worse.'", "answer": "negative", "difficulty": "hard"},
]


@register_probe
class SentimentLogprobProbe(BaseLogprobProbe):
    name = "sentiment_logprob"
    description = "Sentiment classification via logprobs — limbic/social circuits"
    ITEMS = ITEMS
    CHOICES = ["positive", "negative"]
