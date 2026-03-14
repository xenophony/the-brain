"""
API adapters for baseline comparison against cloud LLM providers.

Same interface as MockAdapter / ExLlamaV2LayerAdapter so all probes work unchanged.
Each adapter wraps one provider SDK with:
  - Exponential backoff on rate limits (max 3 retries)
  - API call logging to results/api_logs/
  - Import guards with helpful install messages
  - API key validation at construction time

Usage:
    adapter = ClaudeAdapter(model="claude-sonnet-4-20250514")
    response = adapter.generate_short("What is 2+2?", max_new_tokens=10)
"""

import json
import os
import time
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
#  Logging helper
# ---------------------------------------------------------------------------

_LOG_DIR = Path(__file__).parent.parent / "results" / "api_logs"


def _log_api_call(
    provider: str,
    model: str,
    prompt_length: int,
    response_length: int,
    latency_ms: float,
    error: str | None = None,
) -> None:
    """Append one JSON-lines entry to the daily log file."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = _LOG_DIR / f"{provider}_{today}.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "latency_ms": round(latency_ms, 1),
        "error": error,
    }
    # Atomic append via tmp+rename is overkill for JSONL; just append.
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
#  Retry helper
# ---------------------------------------------------------------------------

def _retry_with_backoff(fn, max_retries: int = 3, base_delay: float = 1.0,
                        timeout_seconds: float = 20.0):
    """Call *fn*; retry on rate-limit (429) or transient errors with exp backoff.

    Each individual call is wrapped in a thread with a timeout to prevent hangs.
    """
    import threading

    last_exc = None
    for attempt in range(max_retries + 1):
        result_holder = [None]
        error_holder = [None]

        def _target():
            try:
                result_holder[0] = fn()
            except Exception as exc:
                error_holder[0] = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            # Call timed out
            last_exc = TimeoutError(f"API call timed out after {timeout_seconds}s (attempt {attempt + 1}/{max_retries + 1})")
            if attempt == max_retries:
                raise last_exc
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
            continue

        if error_holder[0] is not None:
            last_exc = error_holder[0]
            exc_str = str(last_exc).lower()
            status = getattr(last_exc, "status_code", None) or getattr(last_exc, "status", None)
            is_rate_limit = (status == 429) or ("429" in exc_str) or ("rate" in exc_str and "limit" in exc_str)
            is_transient = (
                is_rate_limit
                or (isinstance(status, int) and status >= 500)
                or "timeout" in exc_str
                or "connection" in exc_str
            )
            if not is_transient or attempt == max_retries:
                raise last_exc
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
            continue

        return result_holder[0]

    raise last_exc  # should not reach here
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
#  Base class
# ---------------------------------------------------------------------------

class BaseAPIAdapter(ABC):
    """Base class for all API adapters.  Same interface as MockAdapter."""

    num_layers = None  # API models don't expose internal layers
    rate_limit_workers: int = 5  # max concurrent requests (override per adapter)

    def __init__(self, model: str, provider: str):
        self.model = model
        self._provider = provider

    def set_layer_path(self, path) -> None:  # noqa: ARG002
        """No-op for API models (no layer-level control)."""
        pass

    @abstractmethod
    def generate_short(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.0) -> str:
        ...

    def get_logprobs(self, prompt: str, target_tokens=None) -> dict:  # noqa: ARG002
        """Most API providers don't expose logprobs; fall back to empty dict."""
        return {}


# ---------------------------------------------------------------------------
#  Claude (Anthropic)
# ---------------------------------------------------------------------------

class ClaudeAdapter(BaseAPIAdapter):
    """Adapter for Anthropic Claude models via the Messages API."""

    rate_limit_workers = 5  # conservative — Anthropic rate limits
    request_timeout = 30

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        super().__init__(model=model, provider="claude")
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ImportError(
                "anthropic SDK not installed. Run: pip install anthropic"
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Get your key at https://console.anthropic.com/"
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate_short(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.0) -> str:
        t0 = time.perf_counter()
        error_msg = None
        response_text = ""
        try:
            def _call():
                return self._client.messages.create(
                    model=self.model,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
            msg = _retry_with_backoff(_call, timeout_seconds=self.request_timeout)
            response_text = msg.content[0].text if msg.content else ""
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _log_api_call(
                provider="claude",
                model=self.model,
                prompt_length=len(prompt),
                response_length=len(response_text),
                latency_ms=elapsed_ms,
                error=error_msg,
            )
        return response_text

    def get_logprobs(self, prompt: str, target_tokens=None) -> dict:
        """Claude Messages API does not support logprobs; return empty dict."""
        return {}


# ---------------------------------------------------------------------------
#  Gemini (Google)
# ---------------------------------------------------------------------------

class GeminiAdapter(BaseAPIAdapter):
    """Adapter for Google Gemini models via google-generativeai SDK."""

    rate_limit_workers = 5  # conservative — Google rate limits
    request_timeout = 30

    def __init__(self, model: str = "gemini-2.5-pro-preview-05-06"):
        super().__init__(model=model, provider="gemini")
        try:
            import google.generativeai as genai  # noqa: F401
        except ImportError:
            raise ImportError(
                "google-generativeai SDK not installed. Run: pip install google-generativeai"
            )
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Get your key at https://aistudio.google.dev/"
            )
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_obj = genai.GenerativeModel(model)

    def generate_short(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.0) -> str:
        t0 = time.perf_counter()
        error_msg = None
        response_text = ""
        try:
            # Gemini 2.5+ uses thinking tokens that count against max_output_tokens
            effective_max = max(max_new_tokens * 8, 1024)

            def _call():
                generation_config = self._genai.types.GenerationConfig(
                    max_output_tokens=effective_max,
                    temperature=temperature,
                )
                # Disable safety filters — our probes contain benign academic content
                # but Gemini's default filters can block math/science/emotional questions
                safety_settings = {
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                }
                return self._model_obj.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
            resp = _retry_with_backoff(_call, timeout_seconds=self.request_timeout)
            response_text = (resp.text or "") if hasattr(resp, 'text') else ""
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _log_api_call(
                provider="gemini",
                model=self.model,
                prompt_length=len(prompt),
                response_length=len(response_text),
                latency_ms=elapsed_ms,
                error=error_msg,
            )
        return response_text


# ---------------------------------------------------------------------------
#  Groq
# ---------------------------------------------------------------------------

class GroqAdapter(BaseAPIAdapter):
    """Adapter for Groq inference API (OpenAI-compatible chat completions)."""

    rate_limit_workers = 10  # Groq has generous rate limits
    request_timeout = 20

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        super().__init__(model=model, provider="groq")
        try:
            import groq  # noqa: F401
        except ImportError:
            raise ImportError(
                "groq SDK not installed. Run: pip install groq"
            )
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Get your key at https://console.groq.com/"
            )
        self._client = groq.Groq(api_key=api_key)

    def generate_short(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.0) -> str:
        t0 = time.perf_counter()
        error_msg = None
        response_text = ""
        try:
            def _call():
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
            resp = _retry_with_backoff(_call, timeout_seconds=self.request_timeout)
            response_text = (resp.choices[0].message.content or "") if resp.choices else ""
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _log_api_call(
                provider="groq",
                model=self.model,
                prompt_length=len(prompt),
                response_length=len(response_text),
                latency_ms=elapsed_ms,
                error=error_msg,
            )
        return response_text


# ---------------------------------------------------------------------------
#  Together AI
# ---------------------------------------------------------------------------

class TogetherAdapter(BaseAPIAdapter):
    """Adapter for Together AI inference (OpenAI-compatible chat completions)."""

    request_timeout = 30

    def __init__(self, model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        super().__init__(model=model, provider="together")
        try:
            import together  # noqa: F401
        except ImportError:
            raise ImportError(
                "together SDK not installed. Run: pip install together"
            )
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError(
                "TOGETHER_API_KEY environment variable is not set. "
                "Get your key at https://api.together.xyz/"
            )
        self._client = together.Together(api_key=api_key)

    def generate_short(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.0) -> str:
        t0 = time.perf_counter()
        error_msg = None
        response_text = ""
        try:
            def _call():
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
            resp = _retry_with_backoff(_call, timeout_seconds=self.request_timeout)
            response_text = (resp.choices[0].message.content or "") if resp.choices else ""
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _log_api_call(
                provider="together",
                model=self.model,
                prompt_length=len(prompt),
                response_length=len(response_text),
                latency_ms=elapsed_ms,
                error=error_msg,
            )
        return response_text


# ---------------------------------------------------------------------------
#  OpenRouter (OpenAI-compatible, single key for multiple providers)
# ---------------------------------------------------------------------------

class OpenRouterAdapter(BaseAPIAdapter):
    """OpenRouter API adapter — single key for multiple providers."""

    rate_limit_workers = 10  # OpenRouter has generous rate limits
    request_timeout = 30

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct"):
        super().__init__(model=model, provider="openrouter")
        self.model_id = model
        self.num_layers = None
        self.request_timeout = 45 if "qwen" in model.lower() else 30
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Get your key at https://openrouter.ai/keys"
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai SDK not installed. Run: pip install openai"
            )
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def generate_short(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.0) -> str:
        t0 = time.perf_counter()
        error_msg = None
        response_text = ""
        try:
            # Gemini 2.5 Pro uses "thinking" mode — thinking tokens count against
            # max_tokens, so we need much more headroom for the actual answer.
            effective_max = max_new_tokens
            if "gemini" in self.model.lower():
                effective_max = max(max_new_tokens * 8, 1024)

            def _call():
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=effective_max,
                    temperature=temperature,
                )
            resp = _retry_with_backoff(_call, timeout_seconds=self.request_timeout)
            response_text = (resp.choices[0].message.content or "") if resp.choices else ""
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _log_api_call(
                provider="openrouter",
                model=self.model,
                prompt_length=len(prompt),
                response_length=len(response_text),
                latency_ms=elapsed_ms,
                error=error_msg,
            )
        return response_text

    def set_layer_path(self, path) -> None:  # noqa: ARG002
        """No-op for API models (no layer-level control)."""
        pass

    def get_logprobs(self, prompt: str, target_tokens=None) -> dict:  # noqa: ARG002
        """OpenRouter does not reliably expose logprobs; return empty dict."""
        return {}


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

ADAPTER_MAP = {
    "claude": ClaudeAdapter,
    "gemini": GeminiAdapter,
    "groq": GroqAdapter,
    "together": TogetherAdapter,
    "openrouter": OpenRouterAdapter,
}

_ENV_KEYS = {
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def get_adapter(provider: str, model: str) -> BaseAPIAdapter:
    """Instantiate an adapter by provider name."""
    if provider not in ADAPTER_MAP:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(ADAPTER_MAP.keys())}")
    return ADAPTER_MAP[provider](model=model)


def available_providers() -> list[str]:
    """Return list of providers whose API keys are currently set."""
    return [p for p, key in _ENV_KEYS.items() if os.environ.get(key)]
