"""
ExLlamaV2 adapter with layer path injection support.

The key trick: ExLlamaV2 exposes the transformer layer list directly.
We override forward() to execute layers in an arbitrary order
rather than the default sequential 0..N-1.

No weight copying — layers referenced multiple times in the path
just execute twice using the same weight tensors.
"""

from typing import Optional
import torch


class ExLlamaV2LayerAdapter:
    """
    Wraps an ExLlamaV2 model to support arbitrary layer execution paths.
    
    Usage:
        adapter = ExLlamaV2LayerAdapter("path/to/model")
        adapter.set_layer_path([0,1,2,3,4,3,4,5,6,...])  # duplicate layers 3,4
        logits = adapter.forward(input_ids)
    """
    
    def __init__(self, model_path: str, cache_size_tokens: int = 512):
        self.model_path = model_path
        self.cache_size_tokens = cache_size_tokens
        self._layer_path: Optional[list[int]] = None
        self._model = None
        self._tokenizer = None
        self._cache = None
        self._load()
        
    def _load(self):
        from exllamav2 import (
            ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
        )
        
        config = ExLlamaV2Config(self.model_path)
        config.arch_compat_overrides()
        
        self._model = ExLlamaV2(config)
        self._model.load(lazy=False)
        
        self._tokenizer = ExLlamaV2Tokenizer(config)
        self._cache = ExLlamaV2Cache(
            self._model, 
            max_seq_len=self.cache_size_tokens,
            lazy=False
        )
        
        self.num_layers = len(self._model.modules_dict.get("model.layers", []))
        print(f"Model loaded: {self.num_layers} transformer layers")
        
    def set_layer_path(self, path: list[int]):
        """Set the layer execution order for subsequent forward passes."""
        self._layer_path = path
        
    def forward_with_path(self, input_ids: torch.Tensor, layer_path: list[int]) -> torch.Tensor:
        """
        Run a forward pass using an explicit layer execution path.
        
        ExLlamaV2 exposes model.modules which is an ordered list.
        We identify the transformer layer modules by index and 
        re-execute them in the specified order.
        
        The non-layer modules (embedding, norm, lm_head) always run 
        in their standard positions.
        """
        model = self._model
        
        # Identify layer module indices in the full module list
        # ExLlamaV2 modules: [embed, layer0, layer1, ..., norm, lm_head]
        all_modules = model.modules
        layer_module_start = 1  # after embedding
        layer_module_end = layer_module_start + self.num_layers
        
        # Build the custom module execution list
        # Pre-layer modules (embedding)
        pre_modules = all_modules[:layer_module_start]
        # Post-layer modules (norm + lm_head)
        post_modules = all_modules[layer_module_end:]
        # All layer modules indexed by layer number
        layer_modules = all_modules[layer_module_start:layer_module_end]
        
        # Execute
        self._cache.current_seq_len = 0
        
        # Embedding
        hidden = input_ids
        for module in pre_modules:
            hidden, _, _ = module.forward(hidden, self._cache, None, None)
        
        # Layers in custom order
        for layer_idx in layer_path:
            module = layer_modules[layer_idx]
            hidden, _, _ = module.forward(hidden, self._cache, None, None)
            
        # Norm + LM head
        for module in post_modules:
            hidden, _, _ = module.forward(hidden, self._cache, None, None)
            
        return hidden  # logits
    
    def get_logprobs(
        self, 
        prompt: str, 
        target_tokens: Optional[list[str]] = None
    ) -> dict[str, float]:
        """
        Get log probabilities for specific tokens at the final position.
        Used by probes that need logit distributions (e.g. EQ scoring).
        
        Returns dict of token -> log probability.
        """
        input_ids = self._tokenizer.encode(prompt, add_bos=True)
        
        layer_path = self._layer_path or list(range(self.num_layers))
        logits = self.forward_with_path(input_ids, layer_path)
        
        # Last token logits
        last_logits = logits[0, -1, :]
        log_probs = torch.log_softmax(last_logits, dim=-1)
        
        if target_tokens is None:
            return {"raw_logits": last_logits}
        
        result = {}
        for token_str in target_tokens:
            token_ids = self._tokenizer.encode(token_str, add_bos=False)
            if len(token_ids) == 1:
                result[token_str] = log_probs[token_ids[0]].item()
            else:
                # Multi-token: sum log probs along the path
                # (simplified — good enough for digit tokens 0-9)
                result[token_str] = log_probs[token_ids[0]].item()
                
        return result
    
    def generate_short(
        self, 
        prompt: str, 
        max_new_tokens: int = 20,
        temperature: float = 0.0
    ) -> str:
        """
        Generate a short completion. Temperature=0 for deterministic probes.
        Uses the currently set layer path.
        """
        from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
        
        layer_path = self._layer_path or list(range(self.num_layers))
        
        # Monkey-patch the forward pass for this generation
        # Store original, replace with path-aware version, restore after
        original_forward = self._model.forward
        
        def patched_forward(input_ids, cache, input_mask, preprocess_only, **kwargs):
            return self.forward_with_path(input_ids, layer_path)
        
        self._model.forward = patched_forward
        
        try:
            generator = ExLlamaV2BaseGenerator(self._model, self._cache, self._tokenizer)
            settings = ExLlamaV2Sampler.Settings()
            settings.temperature = temperature
            settings.top_k = 1 if temperature == 0 else 50
            
            output = generator.generate_simple(
                prompt, 
                settings, 
                max_new_tokens,
                seed=42
            )
            # Strip the prompt from output
            return output[len(prompt):]
        finally:
            self._model.forward = original_forward
            
    def tokenize(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_bos=True).tolist()[0]
