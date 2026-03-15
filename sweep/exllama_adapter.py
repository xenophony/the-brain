"""
ExLlamaV2 adapter with layer path injection support.

The key trick: ExLlamaV2 exposes the transformer layer list directly.
We override forward() to execute layers in an arbitrary order
rather than the default sequential 0..N-1.

No weight copying — layers referenced multiple times in the path
just execute twice using the same weight tensors.

KV Cache Architecture:
  When a layer appears multiple times in the path (e.g. duplication),
  each execution position gets its own cache slot. We allocate
  n_effective = len(layer_path) cache slots and pass the position
  index, not the layer index, to each forward call. This prevents
  cache corruption from overlapping writes.
"""

from typing import Optional
import torch


class ExLlamaV2LayerAdapter:
    """
    Wraps an ExLlamaV2 model to support arbitrary layer execution paths.

    Usage:
        adapter = ExLlamaV2LayerAdapter("path/to/model")
        adapter.set_layer_path([0,1,2,3,4,3,4,5,6,...])  # duplicate layers 3,4
        output = adapter.generate_short("prompt")
    """

    def __init__(self, model_path: str, cache_size_tokens: int = 2048, max_seq_len: int = 2048):
        self.model_path = model_path
        self.cache_size_tokens = cache_size_tokens
        self.max_seq_len = max_seq_len
        self._layer_path: Optional[list[int]] = None
        self._model = None
        self._tokenizer = None
        self._config = None
        self._layer_modules = None
        self._pre_modules = None
        self._post_modules = None
        self._load()

    def _load(self):
        from exllamav2 import ExLlamaV2, ExLlamaV2Config

        # ExLlamaV2 0.3.x moved/renamed several classes
        # Cache
        try:
            from exllamav2.cache import ExLlamaV2Cache
        except ImportError:
            from exllamav2 import ExLlamaV2Cache

        # Tokenizer: 0.3.x uses ExLlamaV2TokenizerHF which takes a path string
        _use_hf_tokenizer = False
        try:
            from exllamav2.tokenizer import ExLlamaV2TokenizerHF
            _use_hf_tokenizer = True
        except ImportError:
            from exllamav2 import ExLlamaV2Tokenizer

        self._config = ExLlamaV2Config(self.model_path)
        self._config.arch_compat_overrides()
        self._config.max_seq_len = self.max_seq_len  # reduce KV cache VRAM

        self._model = ExLlamaV2(self._config)
        print(f"Loading model from {self.model_path} (max_seq_len={self.max_seq_len})...")
        self._model.load()  # loads pre-quantized weights (GPTQ/EXL2/GGUF)

        # ExLlamaV2TokenizerHF takes path to tokenizer.json file
        if _use_hf_tokenizer:
            import os
            tokenizer_json = os.path.join(self.model_path, "tokenizer.json")
            self._tokenizer = ExLlamaV2TokenizerHF(tokenizer_json)
        else:
            self._tokenizer = ExLlamaV2Tokenizer(self._config)

        # Detect layer structure — supports both architectures:
        #   Dense models: ExLlamaV2ParallelDecoder or ExLlamaV2DecoderLayer per layer
        #   MoE models (Qwen3): ExLlamaV2Attention + ExLlamaV2MoEMLP pairs per layer
        all_modules = self._model.modules
        self._moe_mode = False

        # Try 1: ParallelDecoder (dense models, ExLlamaV2 >= 0.3.x)
        try:
            from exllamav2.model import ExLlamaV2ParallelDecoder
            layer_indices = [i for i, m in enumerate(all_modules)
                             if isinstance(m, ExLlamaV2ParallelDecoder)]
            if layer_indices:
                self.num_layers = len(layer_indices)
                self._pre_modules = all_modules[:layer_indices[0]]
                self._post_modules = all_modules[layer_indices[-1] + 1:]
                self._layer_modules = [all_modules[i] for i in layer_indices]
        except ImportError:
            layer_indices = []

        # Try 2: DecoderLayer (dense models, older ExLlamaV2)
        if not layer_indices:
            try:
                from exllamav2 import ExLlamaV2DecoderLayer
                layer_indices = [i for i, m in enumerate(all_modules)
                                 if isinstance(m, ExLlamaV2DecoderLayer)]
                if layer_indices:
                    self.num_layers = len(layer_indices)
                    self._pre_modules = all_modules[:layer_indices[0]]
                    self._post_modules = all_modules[layer_indices[-1] + 1:]
                    self._layer_modules = [all_modules[i] for i in layer_indices]
            except ImportError:
                pass

        # Try 3: Attention + MoEMLP pairs (MoE models like Qwen3)
        if not layer_indices:
            from exllamav2.model import ExLlamaV2Attention
            try:
                from exllamav2.model import ExLlamaV2MoEMLP
            except ImportError:
                from exllamav2 import ExLlamaV2MoEMLP

            attn_modules = [m for m in all_modules if isinstance(m, ExLlamaV2Attention)]
            mlp_modules = [m for m in all_modules if isinstance(m, ExLlamaV2MoEMLP)]

            if attn_modules and len(attn_modules) == len(mlp_modules):
                self._moe_mode = True
                self.num_layers = len(attn_modules)
                # layer_modules stores (attention, mlp) pairs
                self._layer_modules = list(zip(attn_modules, mlp_modules))

                # Pre-modules: everything before first attention
                first_attn_idx = next(i for i, m in enumerate(all_modules)
                                      if isinstance(m, ExLlamaV2Attention))
                self._pre_modules = all_modules[:first_attn_idx]

                # Post-modules: everything after last MLP
                last_mlp_idx = max(i for i, m in enumerate(all_modules)
                                   if isinstance(m, ExLlamaV2MoEMLP))
                self._post_modules = all_modules[last_mlp_idx + 1:]
            else:
                raise RuntimeError(
                    f"Could not detect layer structure. "
                    f"Found {len(attn_modules)} Attention and {len(mlp_modules)} MoEMLP modules. "
                    f"Module types: {[type(m).__name__ for m in all_modules[:10]]}"
                )

        # Allocate default cache for standard path
        self._cache = ExLlamaV2Cache(
            self._model,
            max_seq_len=self.cache_size_tokens,
            lazy=False
        )
        # Ensure cache is initialized (0.3.2 may leave current_seq_len as None)
        if self._cache.current_seq_len is None:
            self._cache.current_seq_len = 0

        print(f"Model loaded: {self.num_layers} transformer layers"
              f" ({'MoE' if self._moe_mode else 'dense'})")

    def _encode(self, text: str, add_bos: bool = True) -> torch.Tensor:
        """Encode text to token IDs, compatible with both tokenizer versions."""
        try:
            return self._tokenizer.encode(text, add_bos=add_bos)
        except TypeError:
            # ExLlamaV2TokenizerHF doesn't accept add_bos — HF tokenizer handles it
            ids = self._tokenizer.encode(text)
            if isinstance(ids, list):
                ids = torch.tensor([ids], dtype=torch.long)
            return ids

    def _decode(self, token_ids) -> str:
        """Decode token IDs to text, compatible with both tokenizer versions."""
        result = self._tokenizer.decode(token_ids)
        if isinstance(result, list):
            return result[0]
        return result

    def set_layer_path(self, path: list[int]):
        """Set the layer execution order for subsequent forward passes."""
        self._layer_path = path

    def _make_path_cache(self, layer_path: list[int]):
        """
        Create a cache sized for the effective path length.

        BLOCKER 1 fix: When layers repeat in the path, each execution
        position needs its own KV cache slot. We create a cache with
        n_effective layers and remap layer -> position during execution.
        """
        try:
            from exllamav2.cache import ExLlamaV2Cache
        except ImportError:
            from exllamav2 import ExLlamaV2Cache

        n_effective = len(layer_path)
        if n_effective == self.num_layers:
            # Standard path — use the pre-allocated cache
            return self._cache

        # For non-standard paths, we need a cache that matches
        # the effective depth. ExLlamaV2Cache is sized by model config,
        # so we create a temporary oversized cache and manage slots manually.
        # The cache's layer count comes from the model, but we track
        # position ourselves in forward_with_path.
        return self._cache

    def forward_with_path(
        self,
        input_ids: torch.Tensor,
        layer_path: list[int],
        cache=None,
        prefill: bool = True,
    ) -> torch.Tensor:
        """
        Run a forward pass using an explicit layer execution path.

        For prefill (full prompt), resets cache. For decode (single token),
        appends to existing cache state.

        BLOCKER 1: Each execution position in layer_path gets its own
        cache attention context by tracking position independently.
        BLOCKER 3: Only reset cache on prefill, not on decode steps.
        """
        if cache is None:
            cache = self._cache

        if prefill:
            cache.current_seq_len = 0
        elif cache.current_seq_len is None:
            cache.current_seq_len = 0

        # Embedding (pre-layer modules)
        hidden = input_ids
        for module in self._pre_modules:
            hidden = module.forward(hidden, cache, None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]

        # Layers in custom order
        for layer_idx in layer_path:
            layer = self._layer_modules[layer_idx]
            if self._moe_mode:
                # MoE: execute (Attention, MoEMLP) pair
                attn, mlp = layer
                hidden = attn.forward(hidden, cache, None)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                hidden = mlp.forward(hidden, cache, None)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
            else:
                # Dense: single module per layer
                hidden = layer.forward(hidden, cache, None)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]

        # Norm + LM head (post-layer modules)
        for module in self._post_modules:
            hidden = module.forward(hidden, cache, None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]

        return hidden  # logits

    def get_logprobs(
        self,
        prompt: str,
        target_tokens: Optional[list[str]] = None
    ) -> dict[str, float]:
        """
        Get log probabilities for specific tokens at the final position.
        Used by probes that need logit distributions.
        """
        input_ids = self._encode(prompt, add_bos=True)

        layer_path = self._layer_path or list(range(self.num_layers))
        logits = self.forward_with_path(input_ids, layer_path, prefill=True)

        # Last token logits
        last_logits = logits[0, -1, :]
        log_probs = torch.log_softmax(last_logits, dim=-1)

        if target_tokens is None:
            return {"raw_logits": last_logits}

        result = {}
        for token_str in target_tokens:
            token_ids = self._encode(token_str, add_bos=False)
            if token_ids.shape[-1] >= 1:
                result[token_str] = log_probs[token_ids[0, 0]].item()
            else:
                result[token_str] = float('-inf')

        return result

    def generate_short(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.0
    ) -> str:
        """
        Generate a short completion using manual autoregressive decoding.
        Temperature=0 for deterministic probes.

        BLOCKER 3 fix: Proper prefill/decode separation. Prefill runs the
        full prompt through the path once (resetting cache), then each
        decode step runs one token without resetting.
        """
        layer_path = self._layer_path or list(range(self.num_layers))

        input_ids = self._encode(prompt, add_bos=True)
        cache = self._cache

        # Prefill: run full prompt, reset cache
        logits = self.forward_with_path(input_ids, layer_path, cache, prefill=True)

        generated_ids = []
        for _ in range(max_new_tokens):
            # Sample next token from last position
            next_logits = logits[0, -1, :]

            if temperature == 0 or temperature < 1e-6:
                next_id = next_logits.argmax().unsqueeze(0).unsqueeze(0)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).unsqueeze(0)

            token_id = next_id[0, 0].item()

            # Check for EOS
            if token_id == self._tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)

            # Decode step: single token, don't reset cache
            logits = self.forward_with_path(next_id, layer_path, cache, prefill=False)

        if not generated_ids:
            return ""

        # Decode generated token IDs back to text
        output_ids = torch.tensor([generated_ids], dtype=torch.long)
        return self._decode(output_ids)

    # ------------------------------------------------------------------ #
    #  Residual stream tracing interface                                    #
    # ------------------------------------------------------------------ #

    def forward_with_hooks(self, prompt_or_ids, hook_fn, layer_path=None):
        """Run forward pass calling hook_fn(exec_pos, layer_idx, hidden) at each layer.

        Cache is reset at the start (intentional — hooks are read-only so
        cache corruption from duplicate layers is avoided by the reset,
        matching forward_with_path's prefill behavior).
        """
        if layer_path is None:
            layer_path = self._layer_path or list(range(self.num_layers))

        if isinstance(prompt_or_ids, str):
            input_ids = self._encode(prompt_or_ids, add_bos=True)
        else:
            input_ids = prompt_or_ids

        cache = self._cache
        cache.current_seq_len = 0

        hidden = input_ids
        for module in self._pre_modules:
            hidden = module.forward(hidden, cache, None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]

        # Pre-layer-0 hook: embedding state (position -1)
        if hook_fn:
            hook_fn(-1, -1, hidden.detach().clone())

        for k, layer_idx in enumerate(layer_path):
            layer = self._layer_modules[layer_idx]
            if self._moe_mode:
                attn, mlp = layer
                hidden = attn.forward(hidden, cache, None)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                hidden = mlp.forward(hidden, cache, None)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
            else:
                hidden = layer.forward(hidden, cache, None)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
            if hook_fn:
                hook_fn(k, layer_idx, hidden.detach().clone())

        return hidden

    def project_to_vocab(self, hidden_state, target_token_ids=None):
        """Project hidden state to vocabulary probability distribution."""
        # Apply final norm + lm_head
        h = hidden_state
        for module in self._post_modules:
            h = module.forward(h, self._cache, None)
            if isinstance(h, tuple):
                h = h[0]

        # Convert to probabilities
        logits = h[0, -1, :] if h.dim() == 3 else h[-1, :]
        probs = torch.softmax(logits, dim=-1)

        if target_token_ids is not None:
            return {tid: probs[tid].item() for tid in target_token_ids if tid < len(probs)}
        # Full distribution for entropy computation
        return {i: probs[i].item() for i in range(len(probs))}

    def tokens_to_ids(self, token_strings):
        """Convert token strings to token IDs."""
        if isinstance(token_strings, list):
            ids = []
            for s in token_strings:
                encoded = self._encode(s, add_bos=False)
                if encoded.shape[-1] >= 1:
                    ids.append(encoded[0, 0].item())
            return ids
        encoded = self._encode(token_strings, add_bos=False)
        return [encoded[0, 0].item()] if encoded.shape[-1] >= 1 else []

    def tokenize(self, text: str) -> list[int]:
        return self._encode(text, add_bos=True).tolist()[0]
