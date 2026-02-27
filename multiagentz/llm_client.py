# multiagentz/llm_client.py
"""
Unified LLM client supporting multiple providers.

Supports per-instance configuration:
    client = LLMClient()  # Uses global defaults
    client = LLMClient(model="claude-sonnet-4-20250514")  # Custom model
    client = LLMClient(provider="xai", model="grok-4-fast-reasoning")  # Custom provider
"""

from __future__ import annotations

from typing import Any, Optional

from multiagentz.llm_config import llm_config, _infer_provider_from_model


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
        self._model = model or llm_config.llm_model
        
        # Get appropriate API key and base URL for provider
        if api_key:
            self._api_key = api_key
        else:
            self._api_key = llm_config.get_api_key_for_provider(self._provider)
        
        self._base_url = llm_config.get_base_url_for_provider(self._provider)
        
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
        
        resp = self._client.messages.create(**kw)
        text = resp.content[0].text
        stop = resp.stop_reason
        
        return CompletionResult(
            text,
            truncated=(stop == "max_tokens"),
            stop_reason=stop
        )
    
    @staticmethod
    def _is_o_series(model: str) -> bool:
        """Detect OpenAI o-series reasoning models (o1, o3, o4-mini, etc.)."""
        m = model.lower()
        # Match o1, o3, o4-mini, o3-mini, o1-preview, etc.
        # but NOT "grok-..." or "command-..." or other models with 'o' in them
        return bool(m.startswith("o") and len(m) > 1 and m[1:2].isdigit())

    def _complete_openai(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int
    ) -> CompletionResult:
        """OpenAI / OpenAI-compatible completion."""
        is_o = self._is_o_series(self._model)

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
        # o-series models use max_completion_tokens instead of max_tokens
        if is_o:
            kw["max_completion_tokens"] = max_tokens
        else:
            kw["max_tokens"] = max_tokens

        resp = self._client.chat.completions.create(**kw)

        text = resp.choices[0].message.content
        stop = resp.choices[0].finish_reason

        return CompletionResult(
            text,
            truncated=(stop == "length"),
            stop_reason=stop
        )