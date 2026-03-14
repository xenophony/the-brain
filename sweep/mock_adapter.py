"""
Mock adapter for probe development and unit testing.
Returns canned or random responses without loading a real model.
Use this to develop and test all probes without GPU/model dependency.
"""

import random

class MockAdapter:
    def __init__(self, mode="random"):
        self.mode = mode  # "random", "perfect", "terrible"
        self.num_layers = 32
        self._layer_path = list(range(32))
    
    def set_layer_path(self, path):
        self._layer_path = path
    
    def generate_short(self, prompt, max_new_tokens=20, temperature=0.0):
        if self.mode == "perfect":
            # Probes should hardcode expected answers for perfect mode
            return "42"
        if self.mode == "terrible":
            return "banana"
        return str(random.randint(0, 9999))
    
    def get_logprobs(self, prompt, target_tokens=None):
        import math
        tokens = target_tokens or [str(i) for i in range(10)]
        raw = {t: random.random() for t in tokens}
        total = sum(raw.values())
        return {t: math.log(v/total) for t, v in raw.items()}