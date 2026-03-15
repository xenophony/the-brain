"""
ExLlamaV2 adapter with layer path injection support.

The key trick: ExLlamaV2 exposes the transformer layer list directly.
We override forward() to execute layers in an arbitrary order
rather than the default sequential 0..N-1.

No weight copying — layers referenced multiple times in the path
just execute twice using the same weight tensors.

Cache note:
  For sweep scoring (forward_with_path with use_cache=False), no KV cache
  is used — clean stateless execution.
  For text generation (generate_short), we use manual autoregressive
  decoding with forward_with_path so the custom layer path is respected.
  Each attention layer writes to its own cache slot (indexed by layer_idx),
  so duplicated layers overwrite the same slot — acceptable for short
  probe generations.

API target: ExLlamaV2 0.3.2
Source verified against: github.com/turboderp/exllamav2 master branch
"""

from typing import Optional
import torch


def _to_device(tensor: torch.Tensor, device_idx) -> torch.Tensor:
    """Move tensor to the device indicated by a module's device_idx."""
    if device_idx is not None and device_idx >= 0:
        target = f"cuda:{device_idx}"
        if tensor.device != torch.device(target):
            return tensor.to(target, non_blocking=True)
    return tensor


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
        self._eos_token_ids: list[int] = []
        self._load()

    def _load(self):
        from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer, ExLlamaV2Cache

        # ---- Config ----
        self._config = ExLlamaV2Config(self.model_path)
        self._config.arch_compat_overrides()
        self._config.max_seq_len = self.max_seq_len

        # ---- Model ----
        self._model = ExLlamaV2(self._config)
        print(f"Loading model from {self.model_path} (max_seq_len={self.max_seq_len})...")
        self._model.load()

        # ---- Tokenizer ----
        # ExLlamaV2Tokenizer(config) is the full tokenizer:
        #   - encode() returns torch.Tensor (batch_size, seq_len)
        #   - decode() accepts torch.Tensor, returns str
        #   - has eos_token_id attribute (int)
        self._tokenizer = ExLlamaV2Tokenizer(self._config)

        # Collect EOS token IDs for stop conditions
        self._eos_token_ids = []
        if hasattr(self._tokenizer, 'eos_token_id') and self._tokenizer.eos_token_id is not None:
            eid = self._tokenizer.eos_token_id
            self._eos_token_ids.append(eid if isinstance(eid, int) else int(eid))
        # Qwen3 uses <|im_end|> and <|endoftext|> as stop tokens
        for special in ["<|im_end|>", "<|endoftext|>"]:
            try:
                sid = self._tokenizer.single_id(special)
                if sid is not None and sid not in self._eos_token_ids:
                    self._eos_token_ids.append(sid)
            except Exception:
                pass

        # ---- Detect layer structure ----
        # ExLlamaV2.modules is a flat list:
        #   [Embedding, (PosEmbedding)?, layer0_modules..., layerN_modules..., RMSNorm, Linear]
        #
        # Dense models:  ExLlamaV2ParallelDecoder per layer (wraps attn+mlp)
        # MoE models:    ExLlamaV2Attention + ExLlamaV2MoEMLP per layer (separate)
        all_modules = self._model.modules
        self._moe_mode = False
        layer_indices = []

        # Strategy 1: ParallelDecoder (dense models, ExLlamaV2 >= 0.3.x)
        try:
            from exllamav2.parallel_decoder import ExLlamaV2ParallelDecoder
            layer_indices = [i for i, m in enumerate(all_modules)
                            if isinstance(m, ExLlamaV2ParallelDecoder)]
            if layer_indices:
                self.num_layers = len(layer_indices)
                self._pre_modules = all_modules[:layer_indices[0]]
                self._post_modules = all_modules[layer_indices[-1] + 1:]
                self._layer_modules = [all_modules[i] for i in layer_indices]
        except ImportError:
            pass

        # Strategy 2: Attention + MoEMLP pairs (MoE models like Qwen3)
        if not layer_indices:
            from exllamav2.attn import ExLlamaV2Attention
            from exllamav2.moe_mlp import ExLlamaV2MoEMLP

            attn_indices = [i for i, m in enumerate(all_modules)
                           if isinstance(m, ExLlamaV2Attention)]
            mlp_indices = [i for i, m in enumerate(all_modules)
                          if isinstance(m, ExLlamaV2MoEMLP)]

            if attn_indices and len(attn_indices) == len(mlp_indices):
                self._moe_mode = True
                self.num_layers = len(attn_indices)
                self._layer_modules = [
                    (all_modules[ai], all_modules[mi])
                    for ai, mi in zip(attn_indices, mlp_indices)
                ]
                self._pre_modules = all_modules[:attn_indices[0]]
                self._post_modules = all_modules[mlp_indices[-1] + 1:]
                layer_indices = attn_indices  # just to mark success

        # Strategy 3: Attention + MLP pairs (dense without ParallelDecoder)
        if not layer_indices:
            from exllamav2.attn import ExLlamaV2Attention
            from exllamav2.mlp import ExLlamaV2MLP

            attn_indices = [i for i, m in enumerate(all_modules)
                           if isinstance(m, ExLlamaV2Attention)]
            mlp_indices = [i for i, m in enumerate(all_modules)
                          if isinstance(m, ExLlamaV2MLP)]

            if attn_indices and len(attn_indices) == len(mlp_indices):
                self.num_layers = len(attn_indices)
                self._layer_modules = [
                    (all_modules[ai], all_modules[mi])
                    for ai, mi in zip(attn_indices, mlp_indices)
                ]
                self._pre_modules = all_modules[:attn_indices[0]]
                self._post_modules = all_modules[mlp_indices[-1] + 1:]
                self._moe_mode = True  # same execution pattern as MoE (attn, mlp pair)
                layer_indices = attn_indices

        if not layer_indices:
            raise RuntimeError(
                f"Could not detect layer structure. "
                f"Module types: {[type(m).__name__ for m in all_modules]}"
            )

        # ---- Cache ----
        # ExLlamaV2Cache(model, batch_size=1, max_seq_len=-1, lazy=False)
        # max_seq_len=-1 uses model config default
        self._cache = ExLlamaV2Cache(
            self._model,
            batch_size=1,
            max_seq_len=self.cache_size_tokens,
            lazy=False,
        )

        # Detect GPU device from first layer module
        self._device = torch.device("cpu")
        first_layer = self._layer_modules[0]
        if self._moe_mode:
            first_module = first_layer[0]  # attention module from pair
        else:
            first_module = first_layer
        try:
            self._device = next(first_module.parameters()).device
        except (StopIteration, AttributeError):
            if hasattr(first_module, 'device_idx') and first_module.device_idx is not None:
                self._device = torch.device(f"cuda:{first_module.device_idx}")

        print(f"Model loaded: {self.num_layers} transformer layers"
              f" ({'MoE' if self._moe_mode else 'dense'})"
              f", device={self._device}"
              f", EOS IDs: {self._eos_token_ids}")

    # ------------------------------------------------------------------ #
    #  Encode / Decode                                                     #
    # ------------------------------------------------------------------ #

    def _encode(self, text: str, add_bos: bool = True) -> torch.Tensor:
        """Encode text to token IDs. Returns shape (1, seq_len) tensor on model device."""
        ids = self._tokenizer.encode(text, add_bos=add_bos)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids.to(self._device)

    def _decode(self, token_ids) -> str:
        """Decode token IDs to text. Accepts tensor or list."""
        if isinstance(token_ids, list):
            token_ids = torch.tensor([token_ids], dtype=torch.long)
        elif token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        # ExLlamaV2Tokenizer.decode(tensor) -> str
        return self._tokenizer.decode(token_ids)

    # ------------------------------------------------------------------ #
    #  Layer path control                                                  #
    # ------------------------------------------------------------------ #

    def set_layer_path(self, path: list[int]):
        """Set the layer execution order for subsequent forward passes."""
        self._layer_path = path

    # ------------------------------------------------------------------ #
    #  Core forward pass                                                   #
    # ------------------------------------------------------------------ #

    def _run_module(self, module, x, cache, attn_params, past_len):
        """Run a single module's forward pass with device transfer."""
        x = _to_device(x, module.device_idx)
        return module.forward(x, cache=cache, attn_params=attn_params, past_len=past_len)

    def forward_with_path(
        self,
        input_ids: torch.Tensor,
        layer_path: list[int],
        use_cache: bool = False,
        past_len: int = 0,
    ) -> torch.Tensor:
        """
        Run a forward pass using an explicit layer execution path.

        Mirrors ExLlamaV2.forward_chunk() from the actual source:
        - Creates Params(batch_size, seq_len, past_len, input_mask, position_offsets)
        - Calls module.forward(hidden_states, cache=cache, attn_params=params, past_len=past_len)
        - Moves tensors to correct device via module.device_idx
        - Advances cache.current_seq_len after all modules

        Args:
            use_cache: False (sweep/tracing) = no KV cache, clean execution.
                       True (generation) = KV cache for autoregressive decode.
            past_len: Number of previously cached tokens (use_cache=True only).
        """
        from exllamav2.attn import ExLlamaV2Attention

        cache = self._cache if use_cache else None
        if use_cache and past_len == 0:
            self._cache.reset()

        p_len = past_len if use_cache else 0
        batch_size, seq_len = input_ids.shape

        attn_params = ExLlamaV2Attention.Params(batch_size, seq_len, p_len)

        x = _to_device(input_ids, self._pre_modules[0].device_idx)

        # Pre-layer modules (embedding)
        try:
            for module in self._pre_modules:
                x = self._run_module(module, x, cache, attn_params, p_len)
            # Only log on non-standard paths to avoid noise
            if len(layer_path) != self.num_layers:
                print(f"DEBUG fwp: embedding OK, shape={x.shape}, device={x.device}, "
                      f"path_len={len(layer_path)}, path[:5]={layer_path[:5]}")
        except Exception as e:
            print(f"DEBUG fwp: embedding FAILED: {type(e).__name__}: {e}")
            raise

        # Transformer layers in custom order
        for k, layer_idx in enumerate(layer_path):
            layer = self._layer_modules[layer_idx]
            try:
                if self._moe_mode:
                    attn, mlp = layer
                    x = self._run_module(attn, x, cache, attn_params, p_len)
                    x = self._run_module(mlp, x, cache, attn_params, p_len)
                else:
                    x = self._run_module(layer, x, cache, attn_params, p_len)
                if k < 3 and len(layer_path) != self.num_layers:
                    print(f"DEBUG fwp: layer {k} (idx={layer_idx}) OK, shape={x.shape}")
            except Exception as e:
                print(f"DEBUG fwp: layer {k} (idx={layer_idx}) FAILED: {type(e).__name__}: {e}")
                raise

        # Post-layer modules (final norm + lm_head)
        for module in self._post_modules:
            x = self._run_module(module, x, cache, attn_params, p_len)

        # Advance cache position (mirrors forward_chunk: cache.current_seq_len += seq_len)
        if use_cache:
            self._cache.current_seq_len += seq_len

        return x  # logits: (batch_size, seq_len, vocab_size)

    # ------------------------------------------------------------------ #
    #  Text generation (respects custom layer path)                        #
    # ------------------------------------------------------------------ #

    # Qwen3 chat templates
    _CHAT_TEMPLATE_NO_THINK = (
        "<|im_start|>user\n{prompt} /no_think<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    _CHAT_TEMPLATE_THINK = (
        "<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    def generate_short(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a short completion using KV-cached autoregressive decoding
        with forward_with_path(), so the custom layer path is respected.

        Uses use_cache=True for O(n) generation (not O(n²)).
        KV cache contamination from duplicate layers is acceptable for
        probe scoring — it only affects sweep circuit analysis.
        """
        import re

        # Reset cache to prevent contamination from previous questions/configs
        self._cache.reset()

        # Wrap in chat template if not already formatted
        if "<|im_start|>" not in prompt:
            prompt = self._CHAT_TEMPLATE_NO_THINK.format(prompt=prompt)

        input_ids = self._encode(prompt, add_bos=True)  # (1, seq_len)
        layer_path = self._layer_path or list(range(self.num_layers))

        generated_ids = []

        with torch.inference_mode():
            # Prefill: process full prompt, past_len=0 resets cache
            logits = self.forward_with_path(
                input_ids, layer_path, use_cache=True, past_len=0
            )
            current_past_len = input_ids.shape[1]

            for step in range(max_new_tokens):
                # Sample from last position
                next_logits = logits[0, -1, :].float()

                if temperature <= 0.0 or temperature < 1e-7:
                    next_id = next_logits.argmax().item()
                else:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()

                # Check stop conditions
                if next_id in self._eos_token_ids:
                    break

                generated_ids.append(next_id)

                # Decode step: single token, KV cached
                next_tensor = torch.tensor([[next_id]], dtype=torch.long,
                                           device=input_ids.device)
                logits = self.forward_with_path(
                    next_tensor, layer_path, use_cache=True, past_len=current_past_len
                )
                current_past_len += 1

                # Check string stop conditions
                partial = self._decode(generated_ids)
                if isinstance(partial, list):
                    partial = ''.join(partial)
                if "<|im_end|>" in partial:
                    break

        # Decode only the generated tokens
        if not generated_ids:
            return ""

        output = self._decode(generated_ids)
        if isinstance(output, list):
            output = ''.join(output)

        # Clean output
        if "<|im_end|>" in output:
            output = output.split("<|im_end|>")[0]
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
        output = output.strip().split('\n')[0]
        return output.strip()

    # ------------------------------------------------------------------ #
    #  Logprob interface                                                   #
    # ------------------------------------------------------------------ #

    def get_logprobs(
        self,
        prompt: str,
        target_tokens: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """
        Get log probabilities for specific tokens at the final position.
        Used by probes that need logit distributions.
        """
        input_ids = self._encode(prompt, add_bos=True)
        layer_path = self._layer_path or list(range(self.num_layers))

        with torch.inference_mode():
            logits = self.forward_with_path(input_ids, layer_path, use_cache=False)

        last_logits = logits[0, -1, :].float()
        log_probs = torch.log_softmax(last_logits, dim=-1)

        if target_tokens is None:
            return {"raw_logits": last_logits}

        result = {}
        for token_str in target_tokens:
            token_ids = self._encode(token_str, add_bos=False)
            if token_ids.shape[-1] >= 1:
                tid = token_ids[0, 0].item()
                result[token_str] = log_probs[tid].item()
            else:
                result[token_str] = float('-inf')

        return result

    # ------------------------------------------------------------------ #
    #  Residual stream tracing interface                                   #
    # ------------------------------------------------------------------ #

    def forward_with_hooks(self, prompt_or_ids, hook_fn, layer_path=None, trace_thinking=False):
        """Run forward pass calling hook_fn(exec_pos, layer_idx, hidden) at each layer.

        Args:
            prompt_or_ids: Text string or (1, seq_len) token ID tensor.
            hook_fn: Called as hook_fn(exec_position, layer_index, hidden_state_clone).
                     exec_position=-1 for the post-embedding state.
            layer_path: Override layer path. Defaults to self._layer_path or [0..N-1].
            trace_thinking: If False (default), uses /no_think to trace answer
                production only. If True, uses normal prompt to trace full
                thinking + answer.
        """
        if layer_path is None:
            layer_path = self._layer_path or list(range(self.num_layers))

        from exllamav2.attn import ExLlamaV2Attention

        if isinstance(prompt_or_ids, str):
            if "<|im_start|>" not in prompt_or_ids:
                template = self._CHAT_TEMPLATE_THINK if trace_thinking else self._CHAT_TEMPLATE_NO_THINK
                prompt_or_ids = template.format(prompt=prompt_or_ids)
            input_ids = self._encode(prompt_or_ids, add_bos=True)
        else:
            input_ids = prompt_or_ids

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size, seq_len = input_ids.shape
        attn_params = ExLlamaV2Attention.Params(batch_size, seq_len, 0)

        # Embedding
        x = _to_device(input_ids, self._pre_modules[0].device_idx)
        for module in self._pre_modules:
            x = self._run_module(module, x, None, attn_params, 0)

        # Post-embedding hook
        if hook_fn:
            hook_fn(-1, -1, x.detach().clone())

        # Layers
        for k, layer_idx in enumerate(layer_path):
            layer = self._layer_modules[layer_idx]
            if self._moe_mode:
                attn, mlp = layer
                x = self._run_module(attn, x, None, attn_params, 0)
                x = self._run_module(mlp, x, None, attn_params, 0)
            else:
                x = self._run_module(layer, x, None, attn_params, 0)
            if hook_fn:
                hook_fn(k, layer_idx, x.detach().clone())

        return x

    def project_to_vocab(self, hidden_state, target_token_ids=None):
        """Project hidden state through final norm + lm_head to vocabulary logits."""
        from exllamav2.attn import ExLlamaV2Attention

        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)

        batch_size, seq_len = hidden_state.shape[0], hidden_state.shape[1]
        attn_params = ExLlamaV2Attention.Params(batch_size, seq_len, 0)

        h = hidden_state
        for module in self._post_modules:
            h = self._run_module(module, h, None, attn_params, 0)

        logits = h[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)

        if target_token_ids is not None:
            return {tid: probs[tid].item() for tid in target_token_ids if tid < len(probs)}
        return {i: probs[i].item() for i in range(len(probs))}

    # ------------------------------------------------------------------ #
    #  Token utilities                                                     #
    # ------------------------------------------------------------------ #

    def tokens_to_ids(self, token_strings):
        """Convert token strings to token IDs."""
        if isinstance(token_strings, str):
            token_strings = [token_strings]
        ids = []
        for s in token_strings:
            encoded = self._encode(s, add_bos=False)
            if encoded.shape[-1] >= 1:
                ids.append(encoded[0, 0].item())
        return ids

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text, returning flat list of token IDs (with BOS)."""
        ids = self._encode(text, add_bos=True)
        return ids[0].tolist()
