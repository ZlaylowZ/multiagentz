# multiagentz/providers.py
"""
Centralized API key and provider configuration.

Edit this file to add your API keys and configure your preferred LLM providers.
No .env files needed — just fill in your keys below and you're ready to go.

Supported providers: Anthropic, OpenAI, xAI, Google
Each provider needs an API key and optionally a default model.

SETUP:
  1. Get API keys from the provider dashboards:
     - Anthropic: https://console.anthropic.com/settings/keys
     - OpenAI:    https://platform.openai.com/api-keys
     - xAI:       https://console.x.ai/
     - Google:    https://aistudio.google.com/apikey

  2. Paste your keys below (replace the "YOUR_..._HERE" placeholders)

  3. That's it! Run:  maz --config stacks/example.yaml
"""

# ═══════════════════════════════════════════════════════════════════
# PROVIDER CONFIGURATION
# Fill in at least ONE provider to get started.
# ═══════════════════════════════════════════════════════════════════

PROVIDERS = {
    # ── Anthropic ──────────────────────────────────────────────────
    # Models: claude-opus-4-20250514, claude-sonnet-4-20250514,
    #         claude-haiku-3-5-20241022
    "anthropic": {
        "api_key": "YOUR_ANTHROPIC_API_KEY_HERE",
        "default_model": "claude-sonnet-4-20250514",
        # base_url: None (uses native Anthropic SDK)
    },

    # ── OpenAI ─────────────────────────────────────────────────────
    # Models: gpt-4o, gpt-4o-mini, o4-mini, o3
    "openai": {
        "api_key": "YOUR_OPENAI_API_KEY_HERE",
        "default_model": "o4-mini",
        # base_url: None (uses default OpenAI endpoint)
    },

    # ── xAI ────────────────────────────────────────────────────────
    # Models: grok-4-1-fast-reasoning, grok-4-1-fast-non-reasoning
    "xai": {
        "api_key": "YOUR_XAI_API_KEY_HERE",
        "default_model": "grok-4-1-fast-non-reasoning",
        "base_url": "https://api.x.ai/v1",
    },

    # ── Google ─────────────────────────────────────────────────────
    # Models: gemini-2.0-flash, gemini-2.5-pro
    "google": {
        "api_key": "YOUR_GOOGLE_API_KEY_HERE",
        "default_model": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    },
}


# ═══════════════════════════════════════════════════════════════════
# GLOBAL DEFAULT MODEL (optional)
# ═══════════════════════════════════════════════════════════════════
# Set this to override the default model for all agents.
# Provider is auto-detected from the model name.
# Per-agent models in YAML still take priority over this.
#
# Examples:
#   DEFAULT_MODEL = "claude-sonnet-4-20250514"   # -> Anthropic
#   DEFAULT_MODEL = "grok-4-1-fast-reasoning"    # -> xAI
#   DEFAULT_MODEL = "o4-mini"                    # -> OpenAI
#   DEFAULT_MODEL = "gemini-2.0-flash"           # -> Google

DEFAULT_MODEL = None  # Set to a model name string, or leave None for auto-detect
