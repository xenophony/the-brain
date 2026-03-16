"""
ExLlamaV2 adapter with layer path injection support.

API target: ExLlamaV2 0.3.2
Source verified against: github.com/turboderp/exllamav2 master branch

ALL tensors are kept on GPU (self._device) after _load() completes.
ExLlamaV2Tokenizer.encode() returns CPU tensors — _encode() moves them.
"""

from typing import Optional
import torch


class ExLlamaV2LayerAdapter:
    """
    Wraps an ExLlamaV2 model to support arbitrary layer execution paths.

    Usage:
        adapter = ExLlamaV2LayerAdapter("path/to/model")
        adapter.set_layer_path([0,1,2,3,4,3,4,5,6,...])
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
        self._device = torch.device("cuda:0")
        self._moe_mode = False
        self._think_token_id: int | None = None
        self._think_close_count = 0     # per generate_short call
        self._think_close_total = 0     # accumulated, reset by runner
        self._load()

    def _load(self):
        from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer, ExLlamaV2Cache

        # ---- Config ----
        self._config = ExLlamaV2Config(self.model_path)
        self._config.arch_compat_overrides()
        self._config.max_seq_len = self.max_seq_len
        # Disable CUDA graph caching. ExLlamaV2 caches CUDA graphs keyed
        # by (q_len, batch_size) in QAttn.graph_map. Graphs hardcode tensor
        # memory addresses at capture time. Our sweep calls generate_short()
        # hundreds of times with fresh tensor allocations — replaying a
        # cached graph writes to stale addresses, causing std::bad_alloc
        # in ext_c.q_attn_forward_1 after ~30 calls.
        self._config.no_graphs = True

        # ---- Model ----
        self._model = ExLlamaV2(self._config)
        print(f"Loading model from {self.model_path} (max_seq_len={self.max_seq_len})...")
        self._model.load()

        # ---- Tokenizer ----
        self._tokenizer = ExLlamaV2Tokenizer(self._config)

        # Collect EOS token IDs
        self._eos_token_ids = []
        if hasattr(self._tokenizer, 'eos_token_id') and self._tokenizer.eos_token_id is not None:
            eid = self._tokenizer.eos_token_id
            self._eos_token_ids.append(eid if isinstance(eid, int) else int(eid))
        for special in ["<|im_end|>", "<|endoftext|>"]:
            try:
                sid = self._tokenizer.single_id(special)
                if sid is not None and sid not in self._eos_token_ids:
                    self._eos_token_ids.append(sid)
            except Exception:
                pass

        # Detect <think> token ID for force-close (avoid per-token string decode)
        try:
            self._think_token_id = self._tokenizer.single_id("<think>")
        except Exception:
            self._think_token_id = None

        # Pre-encode </think> for force-close injection
        self._think_close_ids: list[int] = []
        if self._think_token_id is not None:
            try:
                close_enc = self._tokenizer.encode("</think>", add_bos=False)
                if close_enc.dim() == 2:
                    close_enc = close_enc[0]
                self._think_close_ids = close_enc.tolist()
            except Exception:
                pass

        # ---- Detect layer structure ----
        all_modules = self._model.modules
        layer_indices = []

        # Strategy 1: ParallelDecoder (dense models)
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
                layer_indices = attn_indices

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
                self._moe_mode = True
                layer_indices = attn_indices

        if not layer_indices:
            raise RuntimeError(
                f"Could not detect layer structure. "
                f"Module types: {[type(m).__name__ for m in all_modules]}"
            )

        # ---- Cache ----
        # ExLlamaV2 attention ignores the past_len parameter and reads
        # cache.current_seq_len instead (confirmed from attn.py source).
        # We manage current_seq_len ourselves in generate_short().
        # For duplicate layers: same cache slot gets overwritten (the
        # second execution's KV replaces the first's). This is semantically
        # imperfect but does not crash and preserves relative scoring.
        from exllamav2.cache import ExLlamaV2Cache
        self._cache = ExLlamaV2Cache(
            self._model, batch_size=1,
            max_seq_len=self.cache_size_tokens, lazy=False,
        )

        # ---- Device ----
        # ExLlamaV2 modules have device_idx attribute (int >= 0 for CUDA)
        first_module = self._layer_modules[0]
        if self._moe_mode:
            first_module = first_module[0]
        d = getattr(first_module, 'device_idx', None)
        if d is not None and d >= 0:
            self._device = torch.device(f"cuda:{d}")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

        print(f"Model loaded: {self.num_layers} layers "
              f"({'MoE' if self._moe_mode else 'dense'}), "
              f"device={self._device}, EOS={self._eos_token_ids}")

    # ------------------------------------------------------------------ #
    #  Encode / Decode — ALL tensors on self._device                      #
    # ------------------------------------------------------------------ #

    def _encode(self, text: str, add_bos: bool = True) -> torch.Tensor:
        """Encode text → (1, seq_len) tensor. Stays on CPU — _run_module moves per module."""
        ids = self._tokenizer.encode(text, add_bos=add_bos)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids

    def _decode(self, token_ids) -> str:
        """Decode token IDs → string. Moves to CPU for tokenizer."""
        if isinstance(token_ids, list):
            token_ids = torch.tensor([token_ids], dtype=torch.long)
        elif token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        # Tokenizer expects CPU tensor
        return self._tokenizer.decode(token_ids.cpu())

    def set_layer_path(self, path: list[int]):
        self._layer_path = path

    # ------------------------------------------------------------------ #
    #  Core forward pass                                                   #
    # ------------------------------------------------------------------ #

    def _run_module(self, module, x, cache, attn_params, past_len):
        """Run one module, moving x to module's device first.
        Handles GPU modules (device_idx >= 0) and CPU modules (device_idx None/-1)."""
        d = getattr(module, 'device_idx', None)
        if d is not None and d >= 0:
            target = torch.device(f"cuda:{d}")
        else:
            target = torch.device("cpu")
        if x.device != target:
            x = x.to(target, non_blocking=True)
        return module.forward(x, cache=cache, attn_params=attn_params, past_len=past_len)

    def forward_with_path(
        self, input_ids: torch.Tensor, layer_path: list[int],
        cache=None,
    ) -> torch.Tensor:
        """Forward pass with custom layer path.

        cache: ExLlamaV2Cache or None. When provided, attention uses
               cache.current_seq_len for past_len (we manage it in
               generate_short). Caller must set current_seq_len before
               calling this method.

        Single Params object reused across all modules. Safe because
        config.no_graphs=True prevents CUDA graph caching (the original
        cause of the bad_alloc that motivated per-module fresh Params).
        """
        from exllamav2.attn import ExLlamaV2Attention

        past_len = cache.current_seq_len if cache is not None else 0
        batch_size, seq_len = input_ids.shape
        attn_params = ExLlamaV2Attention.Params(batch_size, seq_len, past_len)

        # Embedding
        x = input_ids
        for module in self._pre_modules:
            x = self._run_module(module, x, cache, attn_params, past_len)

        # Transformer layers in custom order
        for layer_idx in layer_path:
            layer = self._layer_modules[layer_idx]
            if self._moe_mode:
                attn, mlp = layer
                x = self._run_module(attn, x, cache, attn_params, past_len)
                x = self._run_module(mlp, x, cache, attn_params, past_len)
            else:
                x = self._run_module(layer, x, cache, attn_params, past_len)

        # Post-layer modules (norm + lm_head)
        for module in self._post_modules:
            x = self._run_module(module, x, cache, attn_params, past_len)

        return x

    # ------------------------------------------------------------------ #
    #  Text generation                                                     #
    # ------------------------------------------------------------------ #

    _CHAT_TEMPLATE_NO_THINK = (
        "<|im_start|>user\n{prompt} /no_think<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    _CHAT_TEMPLATE_THINK = (
        "<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    def generate_short(self, prompt: str, max_new_tokens: int = 20,
                       temperature: float = 0.0) -> str:
        """Generate with KV cache reuse.

        Reuses a single persistent cache (allocated in __init__) and
        resets current_seq_len to 0 each call. No cache
        re-allocation, no torch.cuda.empty_cache() — let PyTorch's
        caching allocator handle memory reuse to avoid CUDA heap
        fragmentation.

        We manage cache.current_seq_len ourselves:
          - After reset: current_seq_len = 0
          - After prefill: current_seq_len = prompt_len
          - After each decode step: current_seq_len += 1
        """
        import re

        if "<|im_start|>" not in prompt:
            prompt = self._CHAT_TEMPLATE_NO_THINK.format(prompt=prompt)

        input_ids = self._encode(prompt, add_bos=True)
        layer_path = self._layer_path or list(range(self.num_layers))
        generated_ids = []

        self._cache.reset()
        self._think_close_count = 0

        with torch.inference_mode():
            # Prefill: cache.current_seq_len is 0 (fresh cache)
            logits = self.forward_with_path(input_ids, layer_path,
                                            cache=self._cache)
            # Advance current_seq_len past the prompt tokens
            self._cache.current_seq_len = input_ids.shape[1]

            think_detected = False

            for _ in range(max_new_tokens):
                next_logits = logits[0, -1, :].float()

                if temperature <= 0.0 or temperature < 1e-7:
                    next_id = next_logits.argmax().item()
                else:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()

                if next_id in self._eos_token_ids:
                    break

                generated_ids.append(next_id)

                # Force-close thinking by token ID — safety net.
                # Should rarely fire with the hard no-think template
                # (pre-filled empty think block). If it does, thinking
                # is leaking through disrupted layer configs.
                if (not think_detected
                        and self._think_token_id is not None
                        and next_id == self._think_token_id
                        and self._think_close_ids):
                    think_detected = True
                    self._think_close_count += 1
                    self._think_close_total += 1
                    print(f"WARNING: think leak detected, force-closing")
                    # Inject </think> tokens
                    for cid in self._think_close_ids:
                        generated_ids.append(cid)
                        next_tensor = torch.tensor([[cid]], dtype=torch.long)
                        logits = self.forward_with_path(
                            next_tensor, layer_path, cache=self._cache)
                        self._cache.current_seq_len += 1
                    continue  # skip the normal forward — we already advanced

                # Single-token decode: cache.current_seq_len tells
                # attention where to read/write in the KV cache
                next_tensor = torch.tensor([[next_id]], dtype=torch.long)
                logits = self.forward_with_path(next_tensor, layer_path,
                                                cache=self._cache)
                self._cache.current_seq_len += 1

        if not generated_ids:
            return ""

        output = self._decode(generated_ids)
        if isinstance(output, list):
            output = ''.join(output)
        if "<|im_end|>" in output:
            output = output.split("<|im_end|>")[0]
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
        output = output.strip().split('\n')[0]
        return output.strip()

    # ------------------------------------------------------------------ #
    #  Logprob interface                                                   #
    # ------------------------------------------------------------------ #

    def get_logprobs(self, prompt: str,
                     target_tokens: Optional[list[str]] = None) -> dict[str, float]:
        input_ids = self._encode(prompt, add_bos=True)  # GPU
        layer_path = self._layer_path or list(range(self.num_layers))

        with torch.inference_mode():
            logits = self.forward_with_path(input_ids, layer_path)

        last_logits = logits[0, -1, :].float()
        log_probs = torch.log_softmax(last_logits, dim=-1)

        if target_tokens is None:
            return {"raw_logits": last_logits}

        result = {}
        for token_str in target_tokens:
            token_ids = self._encode(token_str, add_bos=False)
            if token_ids.shape[-1] >= 1:
                result[token_str] = log_probs[token_ids[0, 0].item()].item()
            else:
                result[token_str] = float('-inf')
        return result

    # ------------------------------------------------------------------ #
    #  Residual stream tracing                                             #
    # ------------------------------------------------------------------ #

    def forward_with_hooks(self, prompt_or_ids, hook_fn, layer_path=None,
                           trace_thinking=False):
        """Forward pass with per-layer hooks. No KV cache (clean execution)."""
        if layer_path is None:
            layer_path = self._layer_path or list(range(self.num_layers))

        from exllamav2.attn import ExLlamaV2Attention

        if isinstance(prompt_or_ids, str):
            if "<|im_start|>" not in prompt_or_ids:
                tmpl = self._CHAT_TEMPLATE_THINK if trace_thinking else self._CHAT_TEMPLATE_NO_THINK
                prompt_or_ids = tmpl.format(prompt=prompt_or_ids)
            input_ids = self._encode(prompt_or_ids, add_bos=True)  # GPU
        else:
            input_ids = prompt_or_ids

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size, seq_len = input_ids.shape
        attn_params = ExLlamaV2Attention.Params(batch_size, seq_len, 0)

        x = input_ids
        for module in self._pre_modules:
            x = self._run_module(module, x, None, attn_params, 0)

        if hook_fn:
            hook_fn(-1, -1, x.detach().clone())

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
        """Project hidden state through norm + lm_head → vocab probs."""
        from exllamav2.attn import ExLlamaV2Attention

        h = hidden_state
        if h.dim() == 2:
            h = h.unsqueeze(0)

        batch_size, seq_len = h.shape[0], h.shape[1]
        attn_params = ExLlamaV2Attention.Params(batch_size, seq_len, 0)

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
        if isinstance(token_strings, str):
            token_strings = [token_strings]
        ids = []
        for s in token_strings:
            encoded = self._encode(s, add_bos=False)
            if encoded.shape[-1] >= 1:
                ids.append(encoded[0, 0].item())
        return ids

    def tokenize(self, text: str) -> list[int]:
        ids = self._encode(text, add_bos=True)
        return ids[0].tolist()
