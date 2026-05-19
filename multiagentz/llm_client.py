# multiagentz/llm_client.py
"""
Unified LLM client supporting multiple providers.

Supports per-instance configuration:
    client = LLMClient()  # Uses global defaults
    client = LLMClient(model="claude-sonnet-4-20250514")  # Custom model
    client = LLMClient(provider="xai", model="grok-4-fast-reasoning")  # Custom provider
"""

from __future__ import annotations

import random
import time
from typing import Any, Optional

from multiagentz.llm_config import llm_config, _infer_provider_from_model

# ── Retry constants ───────────────────────────────────────────────────
_APP_RETRIES = 3        # Additional app-level retries (on top of SDK's max_retries=3)
_BASE_DELAY = 2.0       # seconds
_MAX_DELAY = 30.0       # seconds
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 529}


def friendly_api_error(exc: Exception) -> str:
    """Return a concise, human-readable description of an LLM API error."""
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None) or {}
    err = body.get("error", {}) if isinstance(body, dict) else {}
    msg = err.get("message", "") if isinstance(err, dict) else ""
    code = err.get("code", "") if isinstance(err, dict) else ""

    # ── Permanent quota / billing errors ──────────────────────────────
    if code == "insufficient_quota" or "exceeded your current quota" in str(exc):
        return "API quota exhausted — check your plan and billing at your provider's dashboard"

    # ── Authentication errors ─────────────────────────────────────────
    if status == 401 or code == "invalid_api_key":
        return "Invalid or missing API key — run `maz setup` to configure"

    # ── Model not found ───────────────────────────────────────────────
    if status == 404 or code == "model_not_found":
        model = getattr(exc, "param", None) or ""
        return f"Model not found{f': {model}' if model else ''} — check your stack YAML"

    # ── Rate limit (transient) ────────────────────────────────────────
    if status == 429:
        return "Rate limited by provider — retrying"

    # ── Server errors ─────────────────────────────────────────────────
    if status and status >= 500:
        return f"Provider server error (HTTP {status}) — retrying"

    # ── Timeout errors ────────────────────────────────────────────────
    exc_name = type(exc).__name__
    if "Timeout" in exc_name:
        return "Request timed out — retrying"

    # ── Fallback: use the SDK message if available, else repr ─────────
    if msg:
        return msg[:200]
    return str(exc)[:200]

# ── Token budget constants ─────────────────────────────────────────────
# Conservative estimate: ~3.5 chars per token for mixed English/code.
# Used for pre-flight prompt size validation to avoid 400 errors.
CHARS_PER_TOKEN_ESTIMATE: float = 3.5
MAX_INPUT_TOKENS: int = 195_000  # Safe margin below Anthropic's 200K limit


def _get_raw_client(
    provider: str,
    api_key: str,
    base_url: Optional[str] = None
) -> Any:
    """
    Instantiate the appropriate SDK client.
    
    Args:
        provider: Provider ID (anthropic, openai, mistral, xai, etc.)
        api_key: API key for the provider
        base_url: Base URL for OpenAI-compatible providers
    """
    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )
        import httpx
        return anthropic.Anthropic(
            api_key=api_key,
            timeout=httpx.Timeout(600.0, connect=30.0),
            max_retries=3,
        )

    else:
        # All other providers use OpenAI-compatible API
        try:
            import openai
        except ImportError:
            raise RuntimeError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
        import httpx
        return openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(600.0, connect=30.0),
            max_retries=3,
        )


class CompletionResult(str):
    """
    String subclass carrying truncation metadata.

    Works transparently in json.dumps, f-strings, isinstance checks.
    Access `.truncated` and `.stop_reason` when you need them.
    """

    def __new__(cls, text: str, truncated: bool = False, stop_reason: str = ""):
        inst = super().__new__(cls, text)
        inst.truncated = truncated
        inst.stop_reason = stop_reason
        return inst

    def __repr__(self) -> str:
        flag = " [TRUNCATED]" if self.truncated else ""
        return f"CompletionResult({len(self)} chars{flag})"


class LLMClient:
    """
    Unified completion interface supporting multiple providers.
    
    Provider selection via model name:
        - Model contains "claude" → Anthropic (native SDK)
        - Model starts with "gpt" → OpenAI
        - Model starts with "grok" → xAI (OpenAI-compatible)
        - Model starts with "mistral" → Mistral AI (OpenAI-compatible)
        - Model starts with "gemini" → Google (OpenAI-compatible)
        - Model starts with "command" → Cohere (OpenAI-compatible)
        - Model contains "nemotron" → NVIDIA NIM (OpenAI-compatible)
        - Model contains ":" → Ollama (OpenAI-compatible)
    
    Usage:
        client = LLMClient()  # Global defaults
        client = LLMClient(model="grok-4-fast-reasoning")  # Auto-detects xAI
        client = LLMClient(provider="mistral", model="mistral-large-latest")
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM client with optional custom configuration.
        
        Args:
            provider: Provider ID (anthropic, openai, mistral, xai, google, etc.)
                     If not provided, inferred from model name
            model: Model name (e.g., "grok-4-fast-reasoning")
            api_key: API key (defaults to appropriate env var based on provider)
        """
        # Infer provider from model if not explicitly provided
        if model and not provider:
            provider = _infer_provider_from_model(model)
        else:
            provider = provider or llm_config.llm_provider
        
        self._provider = provider
        self._model = (
            model
            or llm_config.get_default_model_for_provider(provider)
            or llm_config.llm_model
        )
        
        # Get appropriate API key and base URL for provider
        if api_key:
            self._api_key = api_key
        else:
            self._api_key = llm_config.get_api_key_for_provider(self._provider)
        
        self._base_url = llm_config.get_base_url_for_provider(self._provider)

        # Validate before constructing SDK client
        if not self._api_key:
            configured = [
                p for p in ["anthropic", "openai", "xai", "google", "mistral", "cohere", "nvidia"]
                if llm_config.get_api_key_for_provider(p)
            ]
            if self._provider:
                provider_label = self._provider
                env_var_hint = f"  Set {self._provider.upper()}_API_KEY in your .env file or providers.py."
            else:
                provider_label = "unknown"
                env_var_hint = "  Set at least one provider API key in your .env file or providers.py."

            hint = f"\n  Configured providers: {configured}" if configured else ""
            import pathlib
            providers_path = pathlib.Path(__file__).resolve().parent / "providers.py"
            raise ValueError(
                f"\n"
                f"  No API key found for provider '{provider_label}'"
                f" (model: {self._model}).\n"
                f"{env_var_hint}\n"
                f"  Or edit: {providers_path}\n"
                f"{hint}"
            )

        # Initialize SDK client
        self._client = _get_raw_client(
            self._provider,
            self._api_key,
            self._base_url
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def raw_client(self) -> Any:
        """Escape hatch for tool-calling code that needs the native client."""
        return self._client

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token count estimate (conservative — may slightly overcount)."""
        return int(len(text) / CHARS_PER_TOKEN_ESTIMATE)

    def _guard_prompt_size(
        self, prompt: str, system: Optional[str]
    ) -> tuple[str, Optional[str]]:
        """
        Pre-flight check: truncate prompt/system if estimated tokens exceed API limit.

        Preserves system prompt (essential instructions) and truncates user prompt
        first.  If system itself is too large, truncates that too.
        """
        est_prompt = self.estimate_tokens(prompt)
        est_system = self.estimate_tokens(system) if system else 0
        est_total = est_prompt + est_system

        if est_total <= MAX_INPUT_TOKENS:
            return prompt, system

        from multiagentz import log
        log.warn(
            f"Prompt too large (~{est_total:,} est. tokens, "
            f"limit {MAX_INPUT_TOKENS:,}). Auto-truncating to fit."
        )

        # Try truncating user prompt first (system has essential instructions)
        headroom = 1_000  # tokens
        available_for_prompt = MAX_INPUT_TOKENS - est_system - headroom

        if available_for_prompt < 10_000:
            # System prompt itself is enormous — truncate it too
            max_sys_chars = int((MAX_INPUT_TOKENS - 50_000) * CHARS_PER_TOKEN_ESTIMATE)
            if system:
                system = (
                    system[:max_sys_chars]
                    + "\n\n[... system prompt truncated to fit token limit ...]"
                )
            available_for_prompt = 50_000

        max_prompt_chars = int(available_for_prompt * CHARS_PER_TOKEN_ESTIMATE)
        if len(prompt) > max_prompt_chars:
            prompt = (
                prompt[:max_prompt_chars]
                + f"\n\n[... prompt truncated from {len(prompt):,} chars "
                f"to fit {MAX_INPUT_TOKENS:,} token limit ...]"
            )

        return prompt, system

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """Return True if the exception represents a transient/retryable API error."""
        # Check for status_code attribute (both Anthropic and OpenAI SDK errors)
        status = getattr(exc, "status_code", None)
        if status and status in _TRANSIENT_STATUS_CODES:
            # OpenAI returns 429 for both rate-limits (transient) and
            # insufficient_quota (permanent).  Don't retry permanent errors.
            body = getattr(exc, "body", None) or {}
            err = body.get("error", {}) if isinstance(body, dict) else {}
            err_code = err.get("code", "") if isinstance(err, dict) else ""
            err_type = err.get("type", "") if isinstance(err, dict) else ""
            if err_code == "insufficient_quota" or err_type == "insufficient_quota":
                return False
            return True
        # httpx timeout errors
        exc_name = type(exc).__name__
        if exc_name in ("ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout"):
            return True
        return False

    def _call_with_retry(self, fn, *args, **kwargs):
        """Call fn with app-level exponential backoff retry on transient errors."""
        last_exc = None
        for attempt in range(_APP_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if not self._is_transient(exc) or attempt == _APP_RETRIES:
                    raise
                last_exc = exc
                delay = min(_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), _MAX_DELAY)
                from multiagentz import log
                log.warn(
                    f"{friendly_api_error(exc)}, "
                    f"retry {attempt + 1}/{_APP_RETRIES} in {delay:.1f}s"
                )
                time.sleep(delay)
        raise last_exc  # unreachable

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 16384,
    ) -> CompletionResult:
        """
        Generate completion from prompt.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            max_tokens: Maximum tokens to generate

        Returns:
            CompletionResult with text and metadata
        """
        # Pre-flight: prevent 400 errors from oversized prompts
        prompt, system = self._guard_prompt_size(prompt, system)

        if self._provider == "anthropic":
            return self._complete_anthropic(prompt, system, max_tokens)
        else:
            return self._complete_openai(prompt, system, max_tokens)
    
    def _complete_anthropic(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int
    ) -> CompletionResult:
        """Anthropic native SDK completion."""
        kw: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kw["system"] = system
        
        resp = self._call_with_retry(self._client.messages.create, **kw)
        text = resp.content[0].text
        stop = resp.stop_reason
        
        return CompletionResult(
            text,
            truncated=(stop == "max_tokens"),
            stop_reason=stop
        )
    
    @staticmethod
    def _uses_max_completion_tokens(model: str) -> bool:
        """Detect OpenAI models that require max_completion_tokens instead of max_tokens."""
        m = model.lower()
        # o-series: o1, o3, o4-mini, o3-mini, o1-preview, etc.
        if m.startswith("o") and len(m) > 1 and m[1:2].isdigit():
            return True
        # gpt-5+: gpt-5, gpt-5.4, gpt-5-turbo, etc.
        if m.startswith("gpt-"):
            try:
                major = int(m[4:5])
                return major >= 5
            except ValueError:
                pass
        return False

    def _complete_openai(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int
    ) -> CompletionResult:
        """OpenAI / OpenAI-compatible completion."""
        needs_new_api = self._uses_max_completion_tokens(self._model)
        is_o = self._model.lower().startswith("o") and len(self._model) > 1 and self._model.lower()[1:2].isdigit()

        msgs = []
        if system:
            # o-series models use "developer" role instead of "system"
            role = "developer" if is_o else "system"
            msgs.append({"role": role, "content": system})
        msgs.append({"role": "user", "content": prompt})

        kw: dict = {
            "model": self._model,
            "messages": msgs,
        }
        # newer OpenAI models use max_completion_tokens instead of max_tokens
        if needs_new_api:
            kw["max_completion_tokens"] = max_tokens
        else:
            kw["max_tokens"] = max_tokens

        resp = self._call_with_retry(self._client.chat.completions.create, **kw)

        text = resp.choices[0].message.content
        stop = resp.choices[0].finish_reason

        return CompletionResult(
            text,
            truncated=(stop == "length"),
            stop_reason=stop
        )