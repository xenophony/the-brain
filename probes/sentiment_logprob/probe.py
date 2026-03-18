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
    # --- Added items for convergence reliability ---
    # Easy
    {"prompt": "Is this positive or negative: 'This is the best meal I've ever had!'", "answer": "positive", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'I'm so disappointed with the service.'", "answer": "negative", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'My family makes me so happy.'", "answer": "positive", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'This movie was a complete waste of time.'", "answer": "negative", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'I got promoted today!'", "answer": "positive", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'The food was cold and tasteless.'", "answer": "negative", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'What an amazing concert!'", "answer": "positive", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'I regret buying this product.'", "answer": "negative", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'The sunset was breathtaking.'", "answer": "positive", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'The hotel room was filthy and overpriced.'", "answer": "negative", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'I'm thrilled with the results.'", "answer": "positive", "difficulty": "easy"},
    {"prompt": "Is this positive or negative: 'This is an absolute disaster.'", "answer": "negative", "difficulty": "easy"},
    # Hard
    {"prompt": "Is this positive or negative: 'The movie was not without its merits.'", "answer": "positive", "difficulty": "hard"},
    {"prompt": "Is this positive or negative: 'Thanks a lot for ruining my evening.'", "answer": "negative", "difficulty": "hard"},
    {"prompt": "Is this positive or negative: 'Could this day get any better?' (said genuinely after good news)", "answer": "positive", "difficulty": "hard"},
    {"prompt": "Is this positive or negative: 'It was a bittersweet victory — we won but lost our best player.'", "answer": "negative", "difficulty": "hard"},
    {"prompt": "Is this positive or negative: 'He's not the worst speaker I've heard.'", "answer": "positive", "difficulty": "hard"},
    {"prompt": "Is this positive or negative: 'Wow, you really outdid yourself this time.' (said after a mistake)", "answer": "negative", "difficulty": "hard"},
    {"prompt": "Is this positive or negative: 'Despite the setbacks, we managed to finish on time.'", "answer": "positive", "difficulty": "hard"},
    {"prompt": "Is this positive or negative: 'The customer service was impeccable, as always.' (from a loyal customer)", "answer": "positive", "difficulty": "hard"},
    {"prompt": "Is this positive or negative: 'I suppose it could have been worse.'", "answer": "negative", "difficulty": "hard"},
    {"prompt": "Is this positive or negative: 'The presentation was adequate but uninspiring.'", "answer": "negative", "difficulty": "hard"},
]


@register_probe
class SentimentLogprobProbe(BaseLogprobProbe):
    name = "sentiment_logprob"
    description = "Sentiment classification via logprobs — limbic/social circuits"
    ITEMS = ITEMS
    CHOICES = ["positive", "negative"]
