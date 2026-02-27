# multiagentz/llm_config.py
"""
LLM provider/model configuration — reads from providers.py.

All API keys and provider settings are centralized in providers.py.
No .env files needed.

PROVIDER INFERENCE (from model name):
  1. "claude-*" -> anthropic
  2. "gpt-*", "o1", "o3", "o4" -> openai
  3. "grok-*" -> xai
  4. "gemini-*" -> google
"""

from __future__ import annotations

import os
from typing import Optional

from multiagentz.providers import PROVIDERS, DEFAULT_MODEL


def _infer_provider_from_model(model: Optional[str]) -> str:
    """Infer provider from model name."""
    if not model:
        return "unknown"

    m = model.lower().strip()

    # Anthropic
    if "claude" in m:
        return "anthropic"

    # xAI
    if m.startswith("grok"):
        return "xai"

    # Google
    if m.startswith("gemini"):
        return "google"

    # OpenAI (native)
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"

    # Default to OpenAI
    return "openai"


# Default base URLs for each provider
PROVIDER_BASE_URLS = {
    "anthropic": None,  # Native SDK
    "openai": None,  # Default OpenAI endpoint
    "xai": "https://api.x.ai/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
}


def _is_key_configured(key: Optional[str]) -> bool:
    """Check if an API key is a real key (not a placeholder)."""
    if not key:
        return False
    placeholders = {"YOUR_", "REPLACE_", "PASTE_", "INSERT_", "sk-xxx", "xai-xxx"}
    return not any(key.startswith(p) for p in placeholders)


class LLMConfig:
    """LLM configuration reading from providers.py."""

    DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "o4-mini",
        "xai": "grok-4-1-fast-non-reasoning",
        "google": "gemini-2.0-flash",
    }

    def __init__(self):
        # Load from providers.py
        self._providers = PROVIDERS

        # Extract per-provider config
        self._keys: dict[str, Optional[str]] = {}
        self._base_urls: dict[str, Optional[str]] = {}
        self._models: dict[str, Optional[str]] = {}

        for name, cfg in self._providers.items():
            self._keys[name] = cfg.get("api_key")
            self._base_urls[name] = cfg.get("base_url") or PROVIDER_BASE_URLS.get(name)
            self._models[name] = cfg.get("default_model")

        # Environment variables override providers.py (advanced users)
        env_mappings = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "xai": "XAI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        for provider, env_var in env_mappings.items():
            env_key = os.getenv(env_var)
            if env_key and _is_key_configured(env_key):
                self._keys[provider] = env_key

        # Global model override
        explicit_model = DEFAULT_MODEL or os.getenv("MAZ_LLM_MODEL")

        if explicit_model:
            self._llm_model = explicit_model
            self._llm_provider = _infer_provider_from_model(explicit_model)
        else:
            # Find first configured provider (priority order)
            self._llm_provider = None
            self._llm_model = None

            for provider in ["anthropic", "xai", "google", "openai"]:
                if _is_key_configured(self._keys.get(provider)):
                    self._llm_provider = provider
                    self._llm_model = (
                        self._models.get(provider)
                        or self.DEFAULT_MODELS.get(provider)
                    )
                    break

    @property
    def llm_model(self) -> Optional[str]:
        return self._llm_model

    @property
    def llm_provider(self) -> Optional[str]:
        return self._llm_provider

    @property
    def llm_api_key(self) -> Optional[str]:
        """Get API key for the current provider."""
        return self.get_api_key_for_provider(self._llm_provider)

    def get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        return self._keys.get(provider)

    def get_base_url_for_provider(self, provider: str) -> Optional[str]:
        """Get base URL for a specific provider."""
        if provider == "anthropic":
            return None  # Native SDK
        return self._base_urls.get(provider)

    def validate(self) -> None:
        if not self.llm_api_key or not _is_key_configured(self.llm_api_key):
            configured = [
                p for p, k in self._keys.items() if _is_key_configured(k)
            ]
            if configured:
                raise ValueError(
                    f"No API key for provider '{self.llm_provider}'. "
                    f"Configured providers: {configured}. "
                    f"Edit multiagentz/providers.py to add your key."
                )
            raise ValueError(
                "No API keys configured!\n\n"
                "Edit multiagentz/providers.py and replace the placeholder\n"
                "keys with your real API keys.\n\n"
                "Get keys from:\n"
                "  Anthropic: https://console.anthropic.com/settings/keys\n"
                "  OpenAI:    https://platform.openai.com/api-keys\n"
                "  xAI:       https://console.x.ai/\n"
                "  Google:    https://aistudio.google.com/apikey"
            )


llm_config = LLMConfig()
