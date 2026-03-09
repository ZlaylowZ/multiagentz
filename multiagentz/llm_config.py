# multiagentz/llm_config.py
"""
LLM provider/model configuration with provider-specific environment variables.

ENVIRONMENT VARIABLE PATTERN:
Each provider has its own set of env vars:
    <PROVIDER>_API_KEY
    <PROVIDER>_BASE_URL  
    <PROVIDER>_MODEL

Examples:
    XAI_API_KEY=xai-...
    XAI_BASE_URL=https://api.x.ai/v1
    XAI_MODEL=grok-4-1-fast-non-reasoning
    
    MISTRAL_API_KEY=...
    MISTRAL_BASE_URL=https://api.mistral.ai/v1
    MISTRAL_MODEL=mistral-large-latest

PROVIDER INFERENCE (from model name):
  1. "claude-*" → anthropic
  2. "gpt-*", "o1", "o3", "o4" → openai
  3. "grok-*" → xai
  4. "mistral-*", "devstral-*" → mistral
  5. "gemini-*" → google
  6. "command-*" → cohere
  7. "*nemotron*", "nvidia/*" → nvidia
  8. "*:*" (e.g. "llama3.2:latest") → ollama
  9. Otherwise → openai
"""

from __future__ import annotations

import os
from typing import Optional, Tuple
from urllib.parse import urlparse

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


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
    
    # Mistral
    if m.startswith(("mistral", "devstral", "magistral", "codestral")):
        return "mistral"
    
    # Google
    if m.startswith("gemini"):
        return "google"
    
    # Cohere
    if m.startswith("command"):
        return "cohere"
    
    # NVIDIA
    if "nvidia/" in m or "nemotron" in m:
        return "nvidia"
    
    # Ollama
    if ":" in m and not m.startswith(("http:", "https:")):
        return "ollama"
    
    # OpenAI (native)
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    
    # Default to OpenAI
    return "openai"


# Default base URLs for each provider
PROVIDER_BASE_URLS = {
    "anthropic": None,  # Native SDK
    "openai": None,  # Default OpenAI endpoint
    "mistral": "https://api.mistral.ai/v1",
    "xai": "https://api.x.ai/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "cohere": "https://api.cohere.com/compatibility/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "ollama": "http://localhost:11434/v1",
}


class LLMConfig:
    """LLM configuration with provider-specific environment variables."""

    DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "o4-mini",
        "xai": "grok-4-1-fast-non-reasoning",
        "mistral": "mistral-large-latest",
        "google": "gemini-2.0-flash",
        "cohere": "command-a-03-2025",
        "nvidia": "nvidia/llama-3.1-nemotron-70b-instruct",
        "ollama": "llama3",
    }

    def __init__(self):
        # Provider-specific environment variables
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL")
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")
        self.openai_model = os.getenv("OPENAI_MODEL")
        
        self.xai_api_key = os.getenv("XAI_API_KEY")
        self.xai_base_url = os.getenv("XAI_BASE_URL") or PROVIDER_BASE_URLS["xai"]
        self.xai_model = os.getenv("XAI_MODEL")
        
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self.mistral_base_url = os.getenv("MISTRAL_BASE_URL") or PROVIDER_BASE_URLS["mistral"]
        self.mistral_model = os.getenv("MISTRAL_MODEL")
        
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_base_url = os.getenv("GOOGLE_BASE_URL") or PROVIDER_BASE_URLS["google"]
        self.google_model = os.getenv("GOOGLE_MODEL")
        
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.cohere_base_url = os.getenv("COHERE_BASE_URL") or PROVIDER_BASE_URLS["cohere"]
        self.cohere_model = os.getenv("COHERE_MODEL")
        
        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        self.nvidia_base_url = os.getenv("NVIDIA_BASE_URL") or PROVIDER_BASE_URLS["nvidia"]
        self.nvidia_model = os.getenv("NVIDIA_MODEL")
        
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY") or "ollama"
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL") or PROVIDER_BASE_URLS["ollama"]
        self.ollama_model = os.getenv("OLLAMA_MODEL")

        # Global override via MAZ_LLM_MODEL
        explicit_model = os.getenv("MAZ_LLM_MODEL") or os.getenv("AGENTZ_LLM_MODEL")

        if explicit_model:
            # Infer provider from model name
            self._llm_model = explicit_model
            self._llm_provider = _infer_provider_from_model(explicit_model)
        elif self.anthropic_api_key:
            self._llm_provider = "anthropic"
            self._llm_model = self.anthropic_model or self.DEFAULT_MODELS["anthropic"]
        elif self.xai_api_key:
            self._llm_provider = "xai"
            self._llm_model = self.xai_model or "grok-4-1-fast-non-reasoning"
        elif self.mistral_api_key:
            self._llm_provider = "mistral"
            self._llm_model = self.mistral_model or "mistral-large-latest"
        elif self.google_api_key:
            self._llm_provider = "google"
            self._llm_model = self.google_model or "gemini-2.0-flash"
        elif self.cohere_api_key:
            self._llm_provider = "cohere"
            self._llm_model = self.cohere_model or "command-a-03-2025"
        elif self.nvidia_api_key:
            self._llm_provider = "nvidia"
            self._llm_model = self.nvidia_model or "nvidia/llama-3.1-nemotron-70b-instruct"
        elif self.openai_api_key:
            self._llm_provider = "openai"
            self._llm_model = self.openai_model or self.DEFAULT_MODELS["openai"]
        else:
            self._llm_provider = None
            self._llm_model = None

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
        if provider == "anthropic":
            return self.anthropic_api_key
        elif provider == "xai":
            return self.xai_api_key
        elif provider == "mistral":
            return self.mistral_api_key
        elif provider == "google":
            return self.google_api_key
        elif provider == "cohere":
            return self.cohere_api_key
        elif provider == "nvidia":
            return self.nvidia_api_key
        elif provider == "ollama":
            return self.ollama_api_key
        elif provider == "openai":
            return self.openai_api_key
        else:
            return None
    
    def get_default_model_for_provider(self, provider: str) -> Optional[str]:
        """Get the default model for a specific provider."""
        return self.DEFAULT_MODELS.get(provider)

    def get_base_url_for_provider(self, provider: str) -> Optional[str]:
        """Get base URL for a specific provider."""
        if provider == "anthropic":
            return None  # Native SDK
        elif provider == "xai":
            return self.xai_base_url
        elif provider == "mistral":
            return self.mistral_base_url
        elif provider == "google":
            return self.google_base_url
        elif provider == "cohere":
            return self.cohere_base_url
        elif provider == "nvidia":
            return self.nvidia_base_url
        elif provider == "ollama":
            return self.ollama_base_url
        elif provider == "openai":
            return self.openai_base_url
        else:
            return None

    def validate(self) -> None:
        if not self.llm_api_key:
            raise ValueError(
                f"No API key for provider '{self.llm_provider}'. "
                f"Set {self.llm_provider.upper()}_API_KEY in your .env file."
            )


llm_config = LLMConfig()